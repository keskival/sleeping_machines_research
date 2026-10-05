"""Route-aware de-interleaving control for FAS (FAS_BENCHMARK.md, after R8 "binding, not horizon").

Identity-free classical control: each item follows a fixed route of event types (learned from clean training runs with
the oracle's recovered identities). Process events are assigned online to the waiting item whose next expected type
matches, first in first out among candidates. A route-start event opens a new item. Unmatched events are left
unassigned. The inferred identities then feed the same scoring as oracle_bound.py (robust z of item-own step durations
against clean training durations). This reports the assignment accuracy on test runs (against the recovered truth) and AUROC.

If this rule recovers most of the oracle's signal, it is a strong control for any FAS home-field claim and specifies the
binding computation a race model must learn. It uses no labels beyond clean-run structure.
"""
import argparse
from collections import Counter, defaultdict, deque
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import generate as G  # noqa: E402
import oracle_bound as O  # noqa: E402
from baselines import PREFIXES, auroc  # noqa: E402

ROOT = HERE.parents[1]


def learn_route(train):
    seqs = Counter()
    for ids, t, item, _ in train:
        for i in range(item.max() + 1):
            seqs[tuple(int(x) for x in ids[item == i])] += 1
    route = max(seqs, key=seqs.get)
    return route, seqs[route] / sum(seqs.values())


def deinterleave(ids, route):
    """online FIFO assignment; returns inferred item index per event (-1 for TICK/unassigned)."""
    nxt = defaultdict(deque)          # expected next type -> queue of item ids waiting for it
    pos = {}; out = np.full(len(ids), -1); n_items = 0
    for k, e in enumerate(ids):
        e = int(e)
        if e == 0:
            continue
        if nxt[e]:
            i = nxt[e].popleft()
        elif e == route[0]:
            i = n_items; n_items += 1; pos[i] = 0
        else:
            continue
        out[k] = i
        if e == route[0] and pos.get(i, 0) == 0 and out[k] == i:
            pos[i] = 0
        p = pos[i] + 1
        pos[i] = p
        if p < len(route):
            nxt[route[p]].append(i)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', required=True); p.add_argument('--seed-base', type=int, default=20261004)
    p.add_argument('--fit-runs', type=int, default=2000); p.add_argument('--test-runs', type=int, default=2000)
    p.add_argument('--out', default='')
    a = p.parse_args()
    split_index = {name: i for i, name in enumerate(G.SPLITS)}

    def gen(split, n, faulty):
        return [O.run_with_identity(a.seed_base + 10_000_000 * split_index[split] + r, faulty) for r in range(n)]

    train = gen('train_clean', a.fit_runs, False)
    route, share = learn_route(train)
    acc = defaultdict(list)
    for ids, t, _, _ in train:
        for _, key, d in O.own_durations(ids, t, deinterleave(ids, route)):
            acc[key].append(d)
    stats = {k: (np.median(v), max(1.4826 * np.median(np.abs(np.asarray(v) - np.median(v))), 1.0)) for k, v in acc.items()}
    clean = gen('test_clean', a.test_runs, False); faulty = gen('test_faulty', a.test_runs, True)
    kinds = np.array([k for *_, k in faulty])

    def pair_accuracy(run):
        """fraction of item-own consecutive pairs under the inferred assignment that are true same-item pairs."""
        ids, t, item, _ = run; inf = deinterleave(ids, route)
        pairs = O.own_durations(ids, t, inf)
        last = {}; ok = 0
        for k in range(len(ids)):
            if inf[k] < 0:
                continue
            j = last.get(inf[k])
            if j is not None:
                ok += item[j] == item[k]
            last[inf[k]] = k
        return ok / max(len(pairs), 1), float(np.mean(inf[ids != 0] >= 0))

    acc_clean = np.array([pair_accuracy(r) for r in clean[:200]]); acc_faulty = np.array([pair_accuracy(r) for r in faulty[:200]])

    def scores(run, n):
        ids, t, _, _ = run
        pos = np.flatnonzero(ids != 0)
        if len(pos) < n:
            return (np.nan,) * 2
        end = pos[n - 1] + 1
        z = []; per = defaultdict(list)
        for k, key, d in O.own_durations(ids[:end], t[:end], deinterleave(ids[:end], route)):
            if key in stats:
                m, s = stats[key]; v = float(np.clip((d - m) / s, -50, 50)); z.append(v); per[key].append(v)
        if not z:
            return 0., 0.
        z = np.asarray(z)
        return float((z ** 2).mean()), float(max(np.mean(v) for v in per.values()))

    result = dict(data=a.data, fit_runs=a.fit_runs, test_runs=a.test_runs, route_length=len(route), route_share=share,
                  pair_accuracy=dict(clean=acc_clean[:, 0].mean(), faulty=acc_faulty[:, 0].mean()),
                  assigned_fraction=dict(clean=acc_clean[:, 1].mean(), faulty=acc_faulty[:, 1].mean()),
                  prefixes=PREFIXES, auroc={})
    print(json.dumps({k: v for k, v in result.items() if k != 'auroc'}), flush=True)
    for n in PREFIXES:
        sn = np.array([scores(r, n) for r in clean]); sp = np.array([scores(r, n) for r in faulty])
        for j, name in enumerate(('deint_sq', 'deint_max_step')):
            row = dict(all=auroc(sn[:, j], sp[:, j]))
            for k, kname in ((1, 'wear_and_tear'), (2, 'retry_delay')):
                row[kname] = auroc(sn[:, j], sp[kinds == k, j])
            result['auroc'].setdefault(name, {})[n] = row
            print(name, n, {k: round(v, 3) for k, v in row.items()}, flush=True)
    if a.out:
        out = ROOT / 'experiments/results/fas' / f'{a.out}.json'
        if out.exists():
            raise ValueError('Unique unused tag required')
        out.write_text(json.dumps(result, indent=1, default=float) + '\n')


if __name__ == '__main__':
    main()
