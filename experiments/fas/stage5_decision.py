#!/usr/bin/env python3
"""B3 Stage 5: the pre-registered FAS v2 decision (FAS_V2_CONFIRMATORY_PROTOCOL.md, Stage 5), applied mechanically.

Inputs (TEST only, each scored once and recorded in the ledger):
- native: the three seeds' `*_TEST_scores.npz` from score_sealed_test.py (rule `total`, prefix N* = merged 1,024);
- baselines: any number of `name=path` pairs; each path is an npz with `clean_total`/`faulty_total` (native format) or the
  classical format (`<detector>_clean`/`<detector>_faulty`, with `--classical-keys`); neural reference families are given
  as one entry per family whose AUROC is the mean over its seeds (pass the seeds as `family=path1,path2,path3`).
Rule: strongest baseline = maximum TEST AUROC at N* over all baselines. **Win** iff (1) native seed mean − strongest
≥ 0.02; (2) paired bootstrap (10,000 resamples, stratified clean/faulty, native scores averaged over seeds by rank) gives a
95% interval with lower bound > 0; (3) every native seed's AUROC exceeds the strongest baseline's point estimate.
Otherwise **tie** (difference within ±0.02 or interval covering 0) or **loss**.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
NSTAR_INDEX = 5                                         # PREFIXES = [32, 64, 128, 256, 512, 1024] (merged); N* = 1,024


def load_pair(path, key):
    z = np.load(ROOT / path)
    if f'clean_{key}' in z:
        return z[f'clean_{key}'][:, NSTAR_INDEX], z[f'faulty_{key}'][:, NSTAR_INDEX]
    return z[f'{key}_clean'][:, NSTAR_INDEX], z[f'{key}_faulty'][:, NSTAR_INDEX]


def auroc(c, f):
    y = np.r_[np.zeros(len(c)), np.ones(len(f))]; s = np.r_[c, f]; ok = ~np.isnan(s)
    return roc_auc_score(y[ok], s[ok])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--native', nargs=3, required=True, help='three native *_TEST_scores.npz (seeds 6, 7, 8)')
    ap.add_argument('--baseline', action='append', default=[], help='name=path[,path...] (native-format npz, rule total)')
    ap.add_argument('--classical', default=None, help='classical TEST scores npz (detector_clean / detector_faulty)')
    ap.add_argument('--classical-keys', default='elapsed,tick_count,gap_z,ngram3,gap_quantile,gap_cusum,gap_robust_z,order3,timed_ngram')
    ap.add_argument('--out', required=True)
    ap.add_argument('--resamples', type=int, default=10000)
    a = ap.parse_args()
    nat = [load_pair(p, 'total') for p in a.native]
    nat_auc = [auroc(c, f) for c, f in nat]
    base = {}
    if a.classical:
        for k in a.classical_keys.split(','):
            base[k] = dict(auroc=auroc(*load_pair(a.classical, k)), pair=load_pair(a.classical, k))
    for spec in a.baseline:
        name, paths = spec.split('=', 1); pairs = [load_pair(p, 'total') for p in paths.split(',')]
        base[name] = dict(auroc=float(np.mean([auroc(c, f) for c, f in pairs])), pair=pairs[0], seeds=len(pairs),
                          seed_aucs=[auroc(c, f) for c, f in pairs])
    strongest = max(base, key=lambda k: base[k]['auroc']); sb = base[strongest]['auroc']
    # seed-averaged native scores (mean rank per run across seeds), and the strongest baseline's per-run scores
    pooled = [rankdata(np.nan_to_num(np.r_[c, f], nan=-np.inf)) for c, f in nat]     # rank over clean+faulty per seed
    avg = np.mean(pooled, 0); nc, nf = avg[:len(nat[0][0])], avg[len(nat[0][0]):]
    bc, bf = base[strongest]['pair']
    rng = np.random.default_rng(0); diffs = []
    for _ in range(a.resamples):
        ic = rng.integers(0, len(nc), len(nc)); jf = rng.integers(0, len(nf), len(nf))
        diffs.append(auroc(nc[ic], nf[jf]) - auroc(bc[ic], bf[jf]))
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    diff = float(np.mean(nat_auc) - sb)
    c1, c2, c3 = diff >= 0.02, lo > 0, all(x > sb for x in nat_auc)
    verdict = 'win' if (c1 and c2 and c3) else ('loss' if (diff <= -0.02 and hi < 0) else 'tie')
    out = dict(native_seed_aucs=nat_auc, native_mean=float(np.mean(nat_auc)), strongest_baseline=strongest,
               strongest_auroc=sb, difference=diff, bootstrap_95=[float(lo), float(hi)],
               criteria=dict(margin_ge_0_02=bool(c1), bootstrap_lower_gt_0=bool(c2), every_seed_above=bool(c3)), verdict=verdict,
               baselines={k: {kk: v for kk, v in d.items() if kk != 'pair'} for k, d in base.items()})
    (ROOT / a.out).write_text(json.dumps(out, indent=1, default=float) + '\n'); print(json.dumps(out, default=float))


if __name__ == '__main__':
    main()
