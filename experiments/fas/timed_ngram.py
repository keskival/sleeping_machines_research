"""Order-and-timing control for the FAS benchmark (experiments/FAS_BENCHMARK.md), same one-class protocol as baselines.py.

Motivated by fas_v1_strong_classical_test_20261005T010000Z: faults mainly change which events become adjacent
(interleaving), not the gap of a fixed adjacency.  This control scores the process-event stream jointly:
  timed_ngram - mean over the prefix of  -log P(e_k | e_{k-2}, e_{k-1})  -  log p(gap_k | e_{k-1}, e_k)
                trigram over process events (add-0.1 smoothing over 45 process ids); gap density from a per-bigram
                histogram of log(1 + gap_ms) with 24 bins over the clean range, add-0.5 smoothing plus a floor bin for
                out-of-range gaps; unseen bigrams use a pooled histogram.
  order3      - the trigram term alone (process events only, no TICK), to separate order from timing.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from baselines import PREFIXES, auroc, prefix_end, runs  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
BINS = 24


class TimedNgram:
    def fit(self, data, V=46, k=.1):
        self.V, self.k = V, k
        self.c3 = defaultdict(lambda: np.zeros(V)); gaps = defaultdict(list); pooled = []
        for ids, t in data:
            m = ids != 0; i, tt = ids[m], t[m]
            for a, b, c in zip(i[:-2], i[1:-1], i[2:]):
                self.c3[(a, b)][c] += 1
            lg = np.log1p(np.maximum(np.diff(tt), 0))
            for a, b, g in zip(i[:-1], i[1:], lg):
                gaps[(a, b)].append(g); pooled.append(g)
        pooled = np.asarray(pooled)
        self.edges = np.linspace(pooled.min(), pooled.max() + 1e-9, BINS + 1)
        def hist(v):
            h = np.histogram(v, self.edges)[0] + .5
            return np.log(np.append(h, .5) / (h.sum() + .5))      # last entry: out-of-range floor
        self.pooled = hist(pooled)
        self.hist = {key: hist(np.asarray(v)) for key, v in gaps.items()}
        return self

    def terms(self, ids, t):
        m = ids != 0; i, tt = ids[m], t[m]
        order = []
        for a, b, c in zip(i[:-2], i[1:-1], i[2:]):
            row = self.c3.get((a, b))
            order.append(-np.log((row[c] + self.k) / (row.sum() + self.k * self.V)) if row is not None else np.log(self.V))
        timing = []
        for a, b, g in zip(i[:-1], i[1:], np.log1p(np.maximum(np.diff(tt), 0))):
            h = self.hist.get((a, b), self.pooled)
            j = np.searchsorted(self.edges, g, 'right') - 1
            timing.append(-h[j] if 0 <= j < BINS else -h[BINS])
        return np.asarray(order), np.asarray(timing)

    def joint(self, ids, t):
        o, g = self.terms(ids, t)
        return float(o.mean() + g[1:].mean()) if len(o) else 0.

    def order3(self, ids, t):
        o, _ = self.terms(ids, t); return float(o.mean()) if len(o) else 0.


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', required=True); p.add_argument('--split', default='test')
    p.add_argument('--fit-runs', type=int, default=2000); p.add_argument('--out', default='')
    a = p.parse_args()
    d = ROOT / 'experiments/data/fas' / a.data
    train, _ = runs(d / 'train_clean.npz'); train = train[:a.fit_runs]
    clean, _ = runs(d / f'{a.split}_clean.npz'); faulty, kinds = runs(d / f'{a.split}_faulty.npz')
    tn = TimedNgram().fit(train)
    result = dict(data=a.data, split=a.split, fit_runs=len(train), prefixes=PREFIXES, auroc={})
    for name, fn in dict(timed_ngram=tn.joint, order3=tn.order3).items():
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
        if out.exists():
            raise ValueError('Unique unused tag required')
        out.write_text(json.dumps(result, indent=1) + '\n')


if __name__ == '__main__':
    main()
