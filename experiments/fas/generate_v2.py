"""FAS v2 generator (FAS_V2_CONFIRMATORY_PROTOCOL.md, Stage 0): K production lines merged into one anonymous log.

Each sample merges K independent production runs of the unmodified simulator (generate.generate_run's setup, with
item identities recorded by oracle_bound.run_with_identity). In `*_faulty` splits line 0 carries the fault; all other
lines are clean.

- Seeds: seed_base + 10_000_000 * split_index + K * run + line (disjoint per split and line; the same formula for
  every K, so settings with different K use different runs).
- Speed offset: line l's timestamps are scaled by (1 + delta * (+1 for even l, -1 for odd l)) after simulation, then
  rounded to ms; identical in every split.
- Event dropout: each process event is dropped independently with probability p, from an RNG seeded only by that
  line's run seed (never by fault or identity).
- One plant clock: the K simulated TICK streams are discarded and a single TICK every 10 s is emitted from 0 to the
  last process event.
- Merge by timestamp, ties broken by a seeded random permutation (the sample's line-0 seed), not by a stable sort.
- Event ids are shared across lines; there is no line tag.

Output, experiments/data/fas/<name>/: <split>.npz in generate.py's format (ids, times_ms, offsets, fault, target,
seed = line-0 seed) plus line_seeds, so native.py, dense.py and the classical baselines read it unchanged.
identity.npz holds <split>_line and <split>_item per event (-1 for TICK). Only oracle and diagnostic code may open it.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import generate as G  # noqa: E402
import oracle_bound as O  # noqa: E402

ROOT = HERE.parents[1]
TICK_MS = 10_000
SPLIT_INDEX = {n: i for i, n in enumerate(G.SPLITS)}


def line_seed(seed_base, split, K, run, line):
    return seed_base + 10_000_000 * SPLIT_INDEX[split] + K * run + line


def merge(lines, drop, delta, tie_seed):
    """lines: [(seed, ids, times, item)] per line. Returns merged ids, times, line, item."""
    ids, times, line_of, item_of = [], [], [], []
    for l, (seed, i, t, it) in enumerate(lines):
        proc = i != 0
        i, t, it = i[proc], t[proc], it[proc]
        keep = np.random.default_rng([seed, 0xD509]).random(len(i)) >= drop
        scale = 1 + delta * (1 if l % 2 == 0 else -1)
        ids.append(i[keep]); times.append(np.rint(t[keep] * scale).astype(np.int64))
        line_of.append(np.full(int(keep.sum()), l, np.int8)); item_of.append(it[keep])
    t_end = max((int(t.max()) for t in times if len(t)), default=0)
    ticks = np.arange(0, t_end + 1, TICK_MS, dtype=np.int64)
    ids.append(np.zeros(len(ticks), np.uint8)); times.append(ticks)
    line_of.append(np.full(len(ticks), -1, np.int8)); item_of.append(np.full(len(ticks), -1, np.int64))
    ids, times = np.concatenate(ids).astype(np.uint8), np.concatenate(times)
    line_of, item_of = np.concatenate(line_of), np.concatenate(item_of)
    order = np.lexsort((np.random.default_rng([tie_seed, 0x71E]).permutation(len(ids)), times))
    return ids[order], times[order], line_of[order], item_of[order]


def sample(seed_base, split, run, K, drop, delta, faulty, items=30):
    lines, kind = [], 0
    for l in range(K):
        s = line_seed(seed_base, split, K, run, l)
        i, t, it, k = O.run_with_identity(s, faulty and l == 0, items)
        if l == 0:
            kind = k
        lines.append((s, i, t, it))
    ids, times, line, item = merge(lines, drop, delta, lines[0][0])
    return ids, times, line, item, kind, [s for s, *_ in lines]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--name', required=True); p.add_argument('--seed-base', type=int, default=20261005)
    p.add_argument('--K', type=int, required=True); p.add_argument('--drop', type=float, required=True)
    p.add_argument('--delta', type=float, default=0.)
    p.add_argument('--train', type=int, default=10000); p.add_argument('--val', type=int, default=1000)
    p.add_argument('--test', type=int, default=2000); p.add_argument('--items', type=int, default=30)
    a = p.parse_args()
    out = ROOT / 'experiments/data/fas' / a.name
    if out.exists():
        raise ValueError('Unique unused dataset name required')
    out.mkdir(parents=True)
    counts = dict(train_clean=a.train, val_clean=a.val, val_faulty=a.val, test_clean=a.test, test_faulty=a.test)
    manifest = dict(name=a.name, args=vars(a), events=G.EVENTS, generator='experiments/fas/generate_v2.py',
                    generator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    simulator='keskival/FAS-Simulator@3839b10 (vendored, unmodified)',
                    seeds='seed_base + 10_000_000 * split_index + K * run + line', splits={}, started=time.time())
    identity = {}
    for split, (_, faulty) in G.SPLITS.items():
        if not counts[split]:
            continue
        ids, times, lines, items, offsets, kinds, seeds = [], [], [], [], [0], [], []
        t0 = time.perf_counter()
        for run in range(counts[split]):
            i, t, l, it, k, s = sample(a.seed_base, split, run, a.K, a.drop, a.delta, faulty, a.items)
            ids.append(i); times.append(t); lines.append(l); items.append(it.astype(np.int16))
            offsets.append(offsets[-1] + len(i)); kinds.append(k); seeds.append(s)
        target = np.array([''] * len(kinds))            # module names are not recorded by run_with_identity
        path = out / f'{split}.npz'
        np.savez_compressed(path, ids=np.concatenate(ids), times_ms=np.concatenate(times), offsets=np.array(offsets),
                            fault=np.array(kinds, dtype=np.int8), target=target, seed=np.array([s[0] for s in seeds]),
                            line_seeds=np.array(seeds))
        identity[f'{split}_line'] = np.concatenate(lines); identity[f'{split}_item'] = np.concatenate(items)
        lengths = np.diff(offsets); process = int((np.concatenate(ids) != 0).sum())
        manifest['splits'][split] = dict(runs=counts[split], events=int(offsets[-1]), process_events=process,
                                         mean_events=float(lengths.mean()), wall_s=time.perf_counter() - t0,
                                         sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        print(json.dumps({split: manifest['splits'][split]}), flush=True)
    np.savez_compressed(out / 'identity.npz', **identity)
    manifest['identity_sha256'] = hashlib.sha256((out / 'identity.npz').read_bytes()).hexdigest()
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=1) + '\n')


if __name__ == '__main__':
    main()
