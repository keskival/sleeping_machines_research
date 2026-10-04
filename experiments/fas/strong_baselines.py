"""Stronger classical controls for the FAS benchmark (experiments/FAS_BENCHMARK.md), same one-class protocol as
baselines.py (fit on clean training runs only; AUROC on prefixes of N process events; faulty = positive).

Both fault types only ever ADD delay, so a directional score has more power than the squared z of gap_z, and an unseen
bigram should not dominate the mean.  For every (previous process event, event) bigram, the clean gaps give an empirical
distribution. Scores over the prefix:
  gap_quantile - mean of (u - 1/2), u = the gap's mid-rank quantile within that bigram's clean gaps (distribution-free,
                 one-sided; unseen bigrams skipped)
  gap_cusum    - one-sided CUSUM of the same centred quantiles, max_k S_k with S_k = max(0, S_{k-1} + (u_k - 1/2) - 0.05);
                 faults grow with use, so late evidence accumulates
  gap_robust_z - mean signed robust z ((g - median) / (1.4826 MAD), clipped to +-10; unseen skipped)
The original gap_z is recomputed as a reproduction check against fas_v1_classical_test_20261004T221500Z.json.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from baselines import GapZ, PREFIXES, auroc, prefix_end, runs  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]


class GapQuantile:
    def fit(self, data):
        acc = defaultdict(list)
        for ids, t in data:
            m = ids != 0; i, tt = ids[m], t[m]
            for a, b, g in zip(i[:-1], i[1:], np.diff(tt)):
                acc[(int(a), int(b))].append(g)
        self.sorted = {k: np.sort(np.asarray(v, float)) for k, v in acc.items()}
        self.robust = {k: (np.median(v), max(1.4826 * np.median(np.abs(v - np.median(v))), 1.0)) for k, v in self.sorted.items()}
        return self

    def centred(self, ids, t):
        m = ids != 0; i, tt = ids[m], t[m]
        u, z = [], []
        for a, b, g in zip(i[:-1], i[1:], np.diff(tt)):
            s = self.sorted.get((int(a), int(b)))
            if s is None:
                continue
            lo, hi = np.searchsorted(s, g, 'left'), np.searchsorted(s, g, 'right')
            u.append((lo + hi) / 2 / len(s) - .5)
            med, scale = self.robust[(int(a), int(b))]
            z.append(np.clip((g - med) / scale, -10, 10))
        return np.asarray(u), np.asarray(z)

    def quantile(self, ids, t):
        u, _ = self.centred(ids, t); return float(u.mean()) if len(u) else 0.

    def cusum(self, ids, t, k=.05):
        u, _ = self.centred(ids, t); s = best = 0.
        for x in u:
            s = max(0., s + x - k); best = max(best, s)
        return best

    def robust_z(self, ids, t):
        _, z = self.centred(ids, t); return float(z.mean()) if len(z) else 0.


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', required=True); p.add_argument('--split', default='test')
    p.add_argument('--fit-runs', type=int, default=2000); p.add_argument('--out', default='')
    a = p.parse_args()
    d = ROOT / 'experiments/data/fas' / a.data
    train, _ = runs(d / 'train_clean.npz'); train = train[:a.fit_runs]
    clean, _ = runs(d / f'{a.split}_clean.npz'); faulty, kinds = runs(d / f'{a.split}_faulty.npz')
    gq = GapQuantile().fit(train); gz = GapZ().fit(train)
    methods = dict(gap_z_repro=gz.score, gap_quantile=gq.quantile, gap_cusum=gq.cusum, gap_robust_z=gq.robust_z)
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
        if out.exists():
            raise ValueError('Unique unused tag required')
        out.write_text(json.dumps(result, indent=1) + '\n')


if __name__ == '__main__':
    main()
