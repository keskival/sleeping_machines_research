"""P2 hardware cost model: per-character memory traffic and priced energy for native and dense language models.

Counts, for one inference step (one character) at steady state, which parameters and which state values are READ and
WRITTEN, with the native model in winner-only execution (its forward pass in sleeping_machines/batched_episodes.py):

  every step, every slot:   clock_bias and the slot's CACHED key read (key + key_read . m, refreshed only when written,
                            sleeping_machines/sparse_inference.py, THEORY §414)
  every step, winner only:  key_read and key (cache refresh), raw_rate, frequency, control, input, output, gate, and the
                            winner's stored memory (read-modify-write)
  every step, shared:       embedding/content, source_gate, channel_mix, queries, transport, head

Dense references read all weights every step; the LSTM also reads/writes h and c, and the Transformer reads its KV cache
(2 x layers x context x width values) and appends one entry per layer.

Bytes are priced with Horowitz (ISSCC 2014, 45 nm) anchors already used in sleeping_machines/energy.py; DRAM uses his
640 pJ per 32-bit word. Arithmetic is priced as int8 MACs (2 FLOPs each). These are conventions for a cost MODEL, not
measurements of any chip. Stdlib + torch (for exact parameter shapes only); no training, no data.
"""
import argparse
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

PJ_PER_BYTE = {'sram_local_8KB': 1.25, 'sram_shared_1MB': 12.5, 'dram': 640 / 4}
PJ_PER_MAC_INT8 = 0.2 + 0.1   # 8-bit multiply + 32-bit accumulate (Horowitz 2014)
SLOT_ALWAYS = ('clock_bias',)
SLOT_WINNER = ('key', 'key_read', 'raw_rate', 'frequency', 'control', 'input', 'output', 'gate', 'gain')


def native_counts(payload, depth, heads, pool):
    from sleeping_machines.addressed_event_heads import AddressedEventHeads
    from sleeping_machines.fast_native_core import fast_class
    m = fast_class(AddressedEventHeads)(sources=1, content_dim=27, classes=27, payload=payload, depth=depth,
                                        heads=heads, pool=pool)
    total = shared = slot_always = slot_winner_all = 0
    for name, p in m.named_parameters():
        n = p.numel(); total += n
        if name.startswith('units.'):
            field = name.split('.')[5]
            if field in SLOT_ALWAYS:
                slot_always += n
            elif field in SLOT_WINNER:
                slot_winner_all += n
            else:
                raise ValueError('unclassified unit parameter ' + name)
        else:
            shared += n
    # gain is shared by a layer's units (units[0].gain); count it as read once per layer.
    read = shared + slot_always + slot_winner_all / pool
    state_values = 2 * depth * heads * pool * payload           # stored memories + cached key reads
    state_read = depth * heads * (pool * payload + payload)     # every cached key read + the winner's memory
    state_write = 2 * depth * heads * payload                   # winner's memory and its refreshed key read
    macs = read + depth * heads * pool * payload                # winner matvecs + U score dot products per head
    return dict(model=f'native p{payload}/d{depth}/H{heads}/U{pool}', parameters=total, weights_read=round(read),
                weights_read_fraction=read / total, state_read=state_read, state_write=state_write,
                macs_estimate=round(macs), resident_state=state_values)


def lstm_counts(size, emb=64, classes=27):
    params = 4 * size * (emb + size) + 8 * size + classes * emb + classes * size + classes
    return dict(model=f'LSTM-{size}', parameters=params, weights_read=params, weights_read_fraction=1.0,
                state_read=2 * size, state_write=2 * size, macs_estimate=params, resident_state=2 * size)


def transformer_counts(width, layers, ctx, classes=27):
    import e64_lm_baselines as M
    params = sum(p.numel() for p in M.TfLM(width, layers, ctx).parameters())
    kv = 2 * layers * ctx * width
    return dict(model=f'Transformer-{width}x{layers} (ctx {ctx})', parameters=params, weights_read=params,
                weights_read_fraction=1.0, state_read=kv, state_write=2 * layers * width,
                macs_estimate=params + kv, resident_state=kv)


def price(row, bytes_per_value=1):
    out = {}
    for wloc in ('sram_local_8KB', 'sram_shared_1MB', 'dram'):
        moved = (row['weights_read'] + row['state_read'] + row['state_write']) * bytes_per_value
        e = moved * PJ_PER_BYTE[wloc] + row['macs_estimate'] * PJ_PER_MAC_INT8
        out[f'pj_per_char_{wloc}'] = round(e)
    out['bytes_moved_per_char_int8'] = (row['weights_read'] + row['state_read'] + row['state_write']) * bytes_per_value
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', 'diagnostics',
                                                  'hardware_cost_model_20261004.json'))
    a = ap.parse_args()
    rows = [native_counts(96, 4, 2, 2), native_counts(64, 4, 2, 2), native_counts(64, 4, 2, 4),
            native_counts(64, 4, 2, 8), native_counts(64, 4, 2, 32),
            lstm_counts(256), lstm_counts(512), transformer_counts(256, 2, 256), transformer_counts(256, 4, 256)]
    for r in rows:
        r.update(price(r))
    # contract: the native p96/d4/U2 count must equal the completed result's recorded parameter count.
    assert rows[0]['parameters'] == 940875, rows[0]['parameters']
    res = dict(status='completed_cost_model', scope='Per-character steady-state inference traffic and priced energy '
               'under stated 45 nm conventions; int8 values; batch 1; not a chip measurement. Native rows use '
               'winner-only execution with cached key reads (§414, contract tests/test_sparse_inference.py).', pj_per_byte=PJ_PER_BYTE,
               pj_per_mac_int8=PJ_PER_MAC_INT8, rows=rows)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, 'w') as f:
        json.dump(res, f, indent=1)
    hdr = f"{'model':34} {'params':>9} {'w.read':>9} {'frac':>5} {'state rd':>9} {'B/char':>9} {'SRAM8K pJ':>10} {'DRAM pJ':>11}"
    print(hdr)
    for r in rows:
        print(f"{r['model']:34} {r['parameters']:9d} {r['weights_read']:9d} {r['weights_read_fraction']:5.2f} "
              f"{r['state_read']:9d} {r['bytes_moved_per_char_int8']:9d} {r['pj_per_char_sram_local_8KB']:10d} "
              f"{r['pj_per_char_dram']:11d}")


if __name__ == '__main__':
    main()
