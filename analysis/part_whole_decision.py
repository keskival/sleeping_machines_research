"""R1 part-whole hierarchy decision (theory/PART_WHOLE_COMPOSITION.md, P6a-P6d). Frozen models, no fitting.

For the one-pass 4M plan-B base (L2), the part-whole hierarchy and its no-future-target ablation (and L8 when present):
per-role NLL on the flagship protocol (8192 targets, 1024-token windows), history value beyond 64 tokens by role, and
for hierarchies the unique value of each level in the readout (u^k mean-replaced in the readout input only).
"""
import argparse, json, sys, time, hashlib
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'experiments'), str(ROOT / 'experiments/tpp'), str(ROOT / 'analysis')]
import r1_part_whole_lm as pw
from r1_deep_keyed_lm import DeepKeyedLM
from part_whole_structure import ROLES as ROLE_NAMES, target_roles, by_role, START, TARGETS, WINDOW
from r1_token_keyed_lm import load

TABLE = pw.ROLES.numpy()
# part_whole_structure roles: 0 word_initial, 1 continuation, 2 punctuation, 3 newline, 4 other.
KIND = np.select([TABLE == 0, TABLE == 1, np.isin(TABLE, (2, 3, 4)), TABLE == 5], [0, 1, 2, 3], 4).astype(np.int8)


def build(run):
    rec = json.loads((ROOT / f'experiments/results/token_language/{run}.json').read_text()); g = rec['args']
    hier = 'horizons' in g
    model = (pw.PartWholeLM(g['d'], g['modes'], g['layers'], g['heads'], g['dk'], g['dropout'], tuple(g['horizons'])) if hier
             else DeepKeyedLM(g['d'], g['modes'], g['layers'], g['heads'], g['dk'], g['dropout']))
    model.load_state_dict(torch.load(ROOT / rec['checkpoint'])); model.eval(); return model, rec, hier


@torch.no_grad()
def window_losses(model, val, cut=None, ablate=None, means=None):
    out = []
    for begin in range(START, START + TARGETS, WINDOW):
        block = torch.from_numpy(np.asarray(val[begin:begin + WINDOW + 1]).astype(np.int64)); x = block[None, :-1]
        if ablate is None:
            h, e = model.states(x)
        else:   # readout-only mean replacement of level `ablate`
            model.states(x); ups = list(model.last_levels); ups[ablate] = means[ablate].expand_as(ups[ablate])
            hb = model.embed(x); e = hb
            for b in model.blocks: hb = b(hb, e)
            y = hb + model.level_in(model.n_levels(torch.cat(ups, -1))); y = y + model.mix(model.n_mix(y)); h = model.out_norm(y)
        row = F.cross_entropy(model.logits(h, e, x, list(range(WINDOW)))[0], block[1:], reduction='none')
        if cut is not None:
            for s in range(cut - 1, WINDOW, 128):
                idx = torch.arange(s, min(s + 128, WINDOW)); lanes = torch.stack([block[t - cut + 1:t + 1] for t in idx.tolist()])
                hh, ee = model.states(lanes); row[idx] = F.cross_entropy(model.logits(hh, ee, lanes, [cut - 1])[:, 0], block[1:][idx], reduction='none')
        out.append(row)
    return torch.cat(out).double().numpy()


@torch.no_grad()
def level_means(model, val):
    acc = None; n = 0
    for begin in range(START, START + TARGETS, WINDOW):
        x = torch.from_numpy(np.asarray(val[begin:begin + WINDOW]).astype(np.int64))[None]; model.states(x)
        s = [u.sum((0, 1)) for u in model.last_levels]; acc = s if acc is None else [a + b for a, b in zip(acc, s)]; n += WINDOW
    return [a / n for a in acc]


def main():
    p = argparse.ArgumentParser(); p.add_argument('--tag', required=True); p.add_argument('--runs', nargs='+', required=True); a0 = p.parse_args()
    torch.set_num_threads(1); began = time.monotonic(); val = load(ROOT / 'data/fineweb_gpt2/fineweb_val_000000.bin')
    ids = np.asarray(val[START:START + TARGETS + 1]).astype(np.int64); roles = target_roles(ids[:-1], ids[1:], KIND); rows = {}
    for run in a0.runs:
        if not (ROOT / f'experiments/results/token_language/{run}.json').exists(): rows[run] = dict(status='missing'); continue
        model, rec, hier = build(run); full = window_losses(model, val)
        assert abs(full.mean() - rec['flagship_protocol']['nll']) < 1e-5, (full.mean(), rec['flagship_protocol']['nll'])
        short = window_losses(model, val, cut=64)
        row = dict(status='completed', hierarchy=hier, flagship_nll=float(full.mean()), prequential_last_quarter=rec['prequential_last_quarter'],
                   score_65528=rec['score_65528']['nll'], train_wall_s=rec['train_wall_s'], parameters=rec['parameters'], by_role=by_role(full, roles),
                   history_beyond_64_by_role={r: float((short - full)[roles == i].mean()) for i, r in enumerate(ROLE_NAMES)} | dict(all=float((short - full).mean())))
        if hier:
            means = level_means(model, val); row['level_unique_value_by_role'] = []
            for k in range(len(model.levels)):
                ab = window_losses(model, val, ablate=k, means=means) - full
                row['level_unique_value_by_role'].append({r: float(ab[roles == i].mean()) for i, r in enumerate(ROLE_NAMES)} | dict(all=float(ab.mean())))
        rows[run] = row; print(run, json.dumps({k: row[k] for k in ('flagship_nll', 'prequential_last_quarter')}), flush=True)
    base, hier_run = a0.runs[0], a0.runs[1]; B, H = rows.get(base, {}), rows.get(hier_run, {})
    decision = {}
    if B.get('status') == 'completed' and H.get('status') == 'completed':
        wi = lambda r: r['by_role']['word_initial']['nll']
        decision = dict(P6a_flagship_gain=B['flagship_nll'] - H['flagship_nll'], P6a_prequential_gain=B['prequential_last_quarter'] - H['prequential_last_quarter'],
                        P6a=H['flagship_nll'] < B['flagship_nll'] and H['prequential_last_quarter'] < B['prequential_last_quarter'],
                        P6b_word_initial_gain=wi(B) - wi(H), P6b_history64_word_initial=dict(base=B['history_beyond_64_by_role']['word_initial'], hierarchy=H['history_beyond_64_by_role']['word_initial']),
                        P6b=(wi(B) - wi(H)) > (B['flagship_nll'] - H['flagship_nll']) * 0.5 and H['history_beyond_64_by_role']['word_initial'] > max(0.023, B['history_beyond_64_by_role']['word_initial']),
                        P6c=all(v['word_initial'] > 0 for v in H['level_unique_value_by_role']) if 'level_unique_value_by_role' in H else None)
        if len(a0.runs) > 2 and rows[a0.runs[2]].get('status') == 'completed':
            N = rows[a0.runs[2]]; decision['P6d_history64_word_initial'] = dict(with_future=H['history_beyond_64_by_role']['word_initial'], without=N['history_beyond_64_by_role']['word_initial'])
            decision['P6d'] = H['history_beyond_64_by_role']['word_initial'] > N['history_beyond_64_by_role']['word_initial']
    out = dict(status='completed', battle='R1', tag=a0.tag, runs=a0.runs, rows=rows, decision=decision,
               protocol=dict(targets=TARGETS, target_start=START + 1, window=WINDOW, history_cut=64, level_ablation='u^k mean-replaced in the readout input only'),
               source_sha256={n: hashlib.sha256((ROOT / n).read_bytes()).hexdigest() for n in ('analysis/part_whole_decision.py', 'experiments/r1_part_whole_lm.py', 'analysis/part_whole_structure.py')},
               process_wall_s=time.monotonic() - began, scope='Single seed, one pass over the same 4M fresh tokens; development protocol.')
    with (ROOT / f'experiments/results/diagnostics/{a0.tag}.json').open('x') as f: json.dump(out, f, indent=2); f.write('\n')
    print(json.dumps(decision))


if __name__ == '__main__':
    main()
