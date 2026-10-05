"""FAS v2 probe (FAS_BENCHMARK.md 5 Oct 11:00): does a merged two-line log with event dropout break structure-aware binding?

v2 construction (no simulator changes): two independent production runs (seeds s and s + 5,000,000 of the same split; both
clean, or the first faulty) are merged by timestamp into one log, and each process event is dropped independently with
probability --drop (deterministic per seed). Both lines emit identical event types, so FIFO matching can confuse items
across lines, and drops break strict route matching. The oracle keeps true identity (line, item), drops included.

Reports item-own pair accuracy of the FIFO de-interleaver, and AUROC at prefixes for the oracle and the de-interleaver
(the same scoring as oracle_bound.py). Probe scale only; a v2 dataset would be generated and frozen separately.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import generate as G  # noqa: E402
import oracle_bound as O  # noqa: E402
import deinterleave_baseline as D  # noqa: E402
from baselines import auroc  # noqa: E402

PREFIXES = [128, 256, 512, 1024]


def v2_run(seed, faulty, drop):
    a = O.run_with_identity(seed, faulty); b = O.run_with_identity(seed + 5_000_000, False)
    ids = np.concatenate([a[0], b[0]]); t = np.concatenate([a[1], b[1]])
    item = np.concatenate([a[2], np.where(b[2] >= 0, b[2] + 100, -1)])
    order = np.argsort(t, kind='stable'); ids, t, item = ids[order], t[order], item[order]
    keep = (ids == 0) | (np.random.default_rng(seed).random(len(ids)) >= drop)
    return ids[keep], t[keep], item[keep], a[3]


def deinterleave_timed(ids, t, route, gstats):
    """like D.deinterleave, but among the items waiting for this type choose the one whose elapsed time since its last
    event has the smallest robust |z| under clean (prev type, type) gaps; ties fall back to FIFO order."""
    waiting = defaultdict(list); pos = {}; last_t = {}; last_e = {}; out = np.full(len(ids), -1); n_items = 0
    for k, e in enumerate(ids):
        e = int(e)
        if e == 0:
            continue
        cands = waiting[e]
        if cands:
            def z(i):
                st = gstats.get((last_e[i], e))
                return abs((t[k] - last_t[i] - st[0]) / st[1]) if st else 1e9
            i = min(cands, key=z); cands.remove(i)
        elif e == route[0]:
            i = n_items; n_items += 1; pos[i] = 0
        else:
            continue
        out[k] = i; last_t[i] = t[k]; last_e[i] = e
        pos[i] += 1
        if pos[i] < len(route):
            waiting[route[pos[i]]].append(i)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs', type=int, default=300); p.add_argument('--drop', type=float, default=.02)
    p.add_argument('--seed-base', type=int, default=20261004); p.add_argument('--out', default='')
    a = p.parse_args()
    si = {n: i for i, n in enumerate(G.SPLITS)}
    gen = lambda split, n, f: [v2_run(a.seed_base + 10_000_000 * si[split] + r, f, a.drop) for r in range(n)]
    train = [O.run_with_identity(a.seed_base + 10_000_000 * si['train_clean'] + r, False) for r in range(a.runs)]
    route, _ = D.learn_route(train)
    tbox = {}
    trainv2 = gen('train_clean', a.runs, False)

    def fit(assign):
        acc = defaultdict(list)
        for ids, t, item, _ in trainv2:
            tbox['t'] = t
            for _, k, d in O.own_durations(ids, t, assign(ids, item)):
                acc[k].append(d)
        return {k: (np.median(v), max(1.4826 * np.median(np.abs(np.asarray(v) - np.median(v))), 1.0)) for k, v in acc.items()}

    oracle = lambda ids, item: item
    deint = lambda ids, item: D.deinterleave(ids, route)
    stats = {'oracle': fit(oracle), 'deint': fit(deint)}
    gstats = stats['oracle']                     # clean item-own gap statistics (training split only)
    def timed(ids, item):
        return deinterleave_timed(ids, tbox['t'], route, gstats)
    clean = gen('test_clean', a.runs, False); faulty = gen('test_faulty', a.runs, True)

    def pair_acc(runs):
        ok = n = 0
        for ids, t, item, _ in runs[:100]:
            inf = D.deinterleave(ids, route); last = {}
            for k in range(len(ids)):
                if inf[k] < 0:
                    continue
                j = last.get(inf[k])
                if j is not None:
                    ok += item[j] == item[k]; n += 1
                last[inf[k]] = k
        return ok / max(n, 1)

    def score(run, n, name, assign):
        ids, t, item, _ = run
        pos = np.flatnonzero(ids != 0)
        if len(pos) < n:
            return np.nan
        end = pos[n - 1] + 1; per = defaultdict(list); tbox['t'] = t[:end]
        for _, k, d in O.own_durations(ids[:end], t[:end], assign(ids[:end], item[:end])):
            if k in stats[name]:
                m, s = stats[name][k]; per[k].append(float(np.clip((d - m) / s, -50, 50)))
        return max(np.mean(v) for v in per.values()) if per else 0.

    stats['timed'] = fit(timed)

    def pair_acc_timed(runs):
        ok = n = 0
        for ids, t, item, _ in runs[:100]:
            inf = deinterleave_timed(ids, t, route, gstats); last = {}
            for k in range(len(ids)):
                if inf[k] < 0:
                    continue
                j = last.get(inf[k])
                if j is not None:
                    ok += item[j] == item[k]; n += 1
                last[inf[k]] = k
        return ok / max(n, 1)

    res = dict(drop=a.drop, runs=a.runs, pair_accuracy=dict(clean=pair_acc(clean), faulty=pair_acc(faulty)),
               pair_accuracy_timed=dict(clean=pair_acc_timed(clean), faulty=pair_acc_timed(faulty)), auroc={})
    print(json.dumps(res), flush=True)
    for name, assign in (('oracle', oracle), ('deint', deint), ('timed', timed)):
        for n in PREFIXES:
            v = auroc(np.array([score(r, n, name, assign) for r in clean]), np.array([score(r, n, name, assign) for r in faulty]))
            res['auroc'].setdefault(name, {})[n] = v; print(name, n, round(v, 3), flush=True)
    if a.out:
        out = HERE.parents[1] / 'experiments/results/fas' / f'{a.out}.json'
        out.write_text(json.dumps(res, indent=1, default=float) + '\n')


if __name__ == '__main__':
    main()
