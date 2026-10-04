"""Classical baselines for the FAS interlaced-event anomaly benchmark (experiments/FAS_BENCHMARK.md).

One-class protocol: fit on clean training runs only; score clean and faulty test runs; AUROC (faulty = positive) on the
prefix up to the N-th process (non-TICK) event, for several N, overall and per fault type.  Baselines:
  elapsed    - time of the N-th process event (timestamped track; the order-only track sees the same through TICK counts)
  tick_count - number of TICK events in the prefix (order-only track)
  gap_z      - per (previous process event, event) bigram, Gaussian model of the gap in ms; mean squared z-score over
               the prefix (unseen bigram: z^2 = 100)
  ngram3     - order-only trigram over all ids including TICK (add-0.1 smoothing); mean NLL over the prefix
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
PREFIXES = [32, 64, 128, 256, 512, 1024]


def runs(path):
    z = np.load(path)
    o, ids, times = z['offsets'], z['ids'], z['times_ms']          # decompress each array once
    return [(ids[o[r]:o[r + 1]].astype(np.int64), times[o[r]:o[r + 1]]) for r in range(len(o) - 1)], z['fault']


def prefix_end(ids, n):
    """index after the n-th non-TICK event (None if the run has fewer)."""
    pos = np.flatnonzero(ids != 0)
    return int(pos[n - 1]) + 1 if len(pos) >= n else None


def auroc(neg, pos):
    s = np.concatenate([neg, pos]); r = s.argsort().argsort().astype(float) + 1
    # midranks for ties
    order = np.argsort(s, kind='mergesort'); ranks = np.empty(len(s)); i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[order[j + 1]] == s[order[i]]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1; i = j + 1
    return float((ranks[len(neg):].sum() - len(pos) * (len(pos) + 1) / 2) / (len(neg) * len(pos)))


class GapZ:
    def fit(self, data):
        acc = defaultdict(list)
        for ids, t in data:
            m = ids != 0; i, tt = ids[m], t[m]
            for a, b, g in zip(i[:-1], i[1:], np.diff(tt)):
                acc[(a, b)].append(g)
        self.stats = {k: (np.mean(v), max(np.std(v), 1.0)) for k, v in acc.items()}
        return self

    def score(self, ids, t):
        m = ids != 0; i, tt = ids[m], t[m]
        z = [((g - self.stats[(a, b)][0]) / self.stats[(a, b)][1]) ** 2 if (a, b) in self.stats else 100.
             for a, b, g in zip(i[:-1], i[1:], np.diff(tt))]
        return float(np.mean(z)) if z else 0.


class Ngram:
    def fit(self, data, V=46, k=.1):
        self.V, self.k = V, k
        self.c3 = defaultdict(lambda: np.zeros(V))
        for ids, _ in data:
            for a, b, c in zip(ids[:-2], ids[1:-1], ids[2:]):
                self.c3[(a, b)][c] += 1
        return self

    def score(self, ids, t):
        nll = []
        for a, b, c in zip(ids[:-2], ids[1:-1], ids[2:]):
            row = self.c3.get((a, b))
            p = (row[c] + self.k) / (row.sum() + self.k * self.V) if row is not None else 1 / self.V
            nll.append(-np.log(p))
        return float(np.mean(nll)) if nll else 0.


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', required=True); p.add_argument('--split', default='test')
    p.add_argument('--fit-runs', type=int, default=2000); p.add_argument('--out', default='')
    a = p.parse_args()
    d = ROOT / 'experiments/data/fas' / a.data
    train, _ = runs(d / 'train_clean.npz'); train = train[:a.fit_runs]
    clean, _ = runs(d / f'{a.split}_clean.npz'); faulty, kinds = runs(d / f'{a.split}_faulty.npz')
    gap = GapZ().fit(train); ng = Ngram().fit(train)
    methods = dict(elapsed=lambda i, t: float(t[-1]), tick_count=lambda i, t: float((i == 0).sum()),
                   gap_z=gap.score, ngram3=ng.score)
    result = dict(data=a.data, split=a.split, fit_runs=len(train), prefixes=PREFIXES, auroc={})
    for name, fn in methods.items():
        result['auroc'][name] = {}
        for n in PREFIXES:
            def sc(rs):
                out = []
                for ids, t in rs:
                    e = prefix_end(ids, n)
                    out.append(fn(ids[:e], t[:e]) if e else np.nan)
                return np.array(out)
            sn, sp = sc(clean), sc(faulty)
            row = dict(all=auroc(sn, sp))
            for k, kname in ((1, 'wear_and_tear'), (2, 'retry_delay')):
                row[kname] = auroc(sn, sp[kinds == k])
            result['auroc'][name][n] = row
            print(name, n, {k: round(v, 3) for k, v in row.items()}, flush=True)
    if a.out:
        out = ROOT / 'experiments/results/fas' / f'{a.out}.json'
        out.parent.mkdir(parents=True, exist_ok=True)
        if out.exists():
            raise ValueError('Unique unused tag required')
        out.write_text(json.dumps(result, indent=1) + '\n')


if __name__ == '__main__':
    main()
