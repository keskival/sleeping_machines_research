"""R1: where the episodic store's gain comes from (frozen arm, recorded DEV protocol).

Per-target losses of the completed episodic arm (mixed), the same weights through the
native readout only, and the 5.978388 parent, on the 8192 DEV targets in 1024-token
windows. Buckets by training-frequency rank and by whether the target already
occurred earlier in its window. Also lists the largest single-target gains with their
decoded context for inspection.
"""
import argparse, json, sys, time, hashlib
from pathlib import Path
import numpy as np
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'experiments'), str(ROOT / 'experiments/credit')]
import episodic_memory_continuation as run
from public_token_scoring import windows, reference_forward
from sleeping_machines.episodic_pointer_memory import EpisodicPointerMemory

RANKS = ((0, 2000), (2000, 10000), (10000, 30000), (30000, 50257))


@torch.no_grad()
def losses(model, memory, dev, args):
    model.eval(); forward = reference_forward(model, run.driver); gen = torch.Generator().manual_seed(args.seed + 100000)
    d = model.core.total_payload * model.core.depth; mixed, native, ins, labs = [], [], [], []
    for inputs, labels, _ in windows(dev, 25165824, 8192, 1024, 8):
        state = None; mem = memory.empty(inputs.shape[0], d) if memory is not None else None; m_rows, n_rows = [], []
        for pos in range(0, 1024, args.chunk):
            x, state, _ = forward(inputs[:, pos:pos + args.chunk], state, gen, args, deterministic=args.deterministic_eval)
            nat = model.readout.nll(x, labels[:, pos:pos + args.chunk]).double().reshape(inputs.shape[0], -1)
            n_rows.append(nat)
            if memory is not None:
                mix, mem, _ = memory.mixed_nll(nat.float().reshape(-1), x, inputs[:, pos:pos + args.chunk], labels[:, pos:pos + args.chunk], mem)
                m_rows.append(mix.double())
        native.append(torch.cat(n_rows, 1)); mixed.append(torch.cat(m_rows, 1) if m_rows else None); ins.append(inputs); labs.append(labels)
    cat = lambda xs: torch.cat(xs, 0).numpy()
    return (cat(mixed) if memory is not None else None), cat(native), cat(ins), cat(labs)


def load(tag, compiled):
    model, args, train = run.build(compiled)
    rec = json.loads((ROOT / f'experiments/results/token_language/{tag}.json').read_text()) if tag else None
    path = ROOT / rec['final_weights'] if rec else run.PARENT.with_suffix('.pt')
    ck = torch.load(path, weights_only=False); model.load_state_dict(ck['model'])
    for n, r in ck['owned_rng'].items(): getattr(model, n).set_state(r)
    memory = None
    if rec and rec['memory'] == 'episodic':
        b = rec['binding']; memory = EpisodicPointerMemory(model.core.total_payload * model.core.depth, b['key_dim'], b['capacity'], b['topk']); memory.load_state_dict(ck['memory'])
    return model, args, train, memory, rec


def main():
    p = argparse.ArgumentParser(); p.add_argument('--tag', required=True); p.add_argument('--run', required=True); a = p.parse_args()
    torch.set_num_threads(1); began = time.monotonic()
    model, args, train, memory, rec = load(a.run, rec_compiled := json.loads((ROOT / f'experiments/results/token_language/{a.run}.json').read_text())['binding'].get('compiled', False))
    dev = run.driver.load_tokens(args.dev_file)
    mixed, native, inputs, labels = losses(model, memory, dev, args)
    assert abs(mixed.mean() - rec['scored']['nll']) < 1e-5, (mixed.mean(), rec['scored']['nll'])
    pmodel, pargs, _, _, _ = load(None, False)
    _, parent, _, _ = losses(pmodel, None, dev, pargs)
    assert abs(parent.mean() - 5.978387557528713) < 1e-6
    counts = run.driver.LaneTokens(train, 0, args.train_tokens, args.lanes).counts()
    rank = torch.empty_like(counts); rank[torch.argsort(counts, descending=True, stable=True)] = torch.arange(len(counts)); rank = rank.numpy()[labels]
    seen = np.zeros_like(labels, dtype=bool)
    for i in range(labels.shape[0]):
        observed = set()
        for t in range(labels.shape[1]): observed.add(int(inputs[i, t])); seen[i, t] = int(labels[i, t]) in observed
    def row(mask): return dict(target_fraction=float(mask.mean()), episodic=float(mixed[mask].mean()), native_readout=float(native[mask].mean()), parent=float(parent[mask].mean()),
                               gain_vs_parent=float(parent[mask].mean() - mixed[mask].mean()), contribution_to_total_gain=float((parent - mixed)[mask].sum() / (parent - mixed).sum()))
    buckets = [dict(rank_low=lo, rank_high=hi, **row((rank >= lo) & (rank < hi))) for lo, hi in RANKS]
    split = dict(seen_in_window=row(seen), unseen_in_window=row(~seen))
    try:
        import tiktoken; enc = tiktoken.get_encoding('gpt2'); decode = enc.decode
    except Exception:
        decode = None
    gain = parent - mixed; order = np.argsort(gain, axis=None)[::-1][:12]; examples = []
    for flat in order:
        i, t = np.unravel_index(flat, gain.shape)
        ctx = inputs[i, max(0, t - 15):t + 1].tolist(); y = int(labels[i, t])
        first = next((j for j in range(t) if int(inputs[i, j + 1]) == y and j + 1 <= t), None)
        examples.append(dict(window=int(i), position=int(t), target=decode([y]) if decode else y, context=decode(ctx) if decode else ctx,
                             earlier_occurrence_position=first, parent_nll=float(parent[i, t]), episodic_nll=float(mixed[i, t])))
    out = dict(status='completed', battle='R1', tag=a.tag, run=a.run, totals=dict(episodic=float(mixed.mean()), native_readout=float(native.mean()), parent=float(parent.mean())),
               buckets=buckets, history_split=split, largest_gains=examples, decoder='tiktoken gpt2' if decode else None,
               source_sha256={n: hashlib.sha256((ROOT / n).read_bytes()).hexdigest() for n in ('analysis/episodic_bucket_gains.py', 'experiments/episodic_memory_continuation.py', 'sleeping_machines/episodic_pointer_memory.py')},
               process_wall_s=time.monotonic() - began, scope='Frozen weights, recorded 8192 DEV targets; gains vs the parent include the 2M added fresh targets (the identical-stream control separates them).')
    with (ROOT / f'experiments/results/diagnostics/{a.tag}.json').open('x') as f: json.dump(out, f, indent=2); f.write('\n')
    print(json.dumps(dict(totals=out['totals'], buckets=[(b['rank_low'], round(b['gain_vs_parent'], 3), round(b['contribution_to_total_gain'], 3)) for b in buckets],
                          seen=round(split['seen_in_window']['gain_vs_parent'], 3), unseen=round(split['unseen_in_window']['gain_vs_parent'], 3))))


if __name__ == '__main__':
    main()
