#!/usr/bin/env python3
"""B2 diagnostic (classical, no neural training): how much of the P12/P19 signal do per-channel summary statistics carry?

The user asked (7 Oct) whether the previous leader would also gain from our statistic-valued channel slots. Training
MTM is outside the 6 October own-model rule, so this measures the part of the question the rules allow: gradient-boosted
trees on the same per-record summaries our slots hold. Features per channel (Raindrop convention: observed = non-zero):
count, mean, min, max, first, last, last − first, time of the last observation (0 / NaN-safe when unobserved), plus
the static covariates. Official five Raindrop splits: fit on train, model selection (boosting rounds) on the official
validation split, TEST scored once per split. Reported beside ours (race_irts_v3 protocol) and MTM.
The P12 run of 7 Oct (B2_IRREGULAR_TS.md: val 0.867 / 0.581, split 0) used the same feature list ad hoc; this script
is its committed form.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[2]


def features(times, vals, mask, lens, static):
    N, T, C = vals.shape
    cnt = mask.sum(1).astype(np.float32)
    nan_vals = np.where(mask, vals, np.nan)
    with np.errstate(all='ignore'):
        mean = np.nanmean(nan_vals, 1); mn = np.nanmin(nan_vals, 1); mx = np.nanmax(nan_vals, 1)
    idx = np.arange(T)[None, :, None]
    first_i = np.where(mask, idx, T).min(1); last_i = np.where(mask, idx, -1).max(1)
    rows = np.arange(N)[:, None]; cols = np.arange(C)[None, :]
    first = np.where(cnt > 0, vals[rows, np.clip(first_i, 0, T - 1), cols], np.nan)
    last = np.where(cnt > 0, vals[rows, np.clip(last_i, 0, T - 1), cols], np.nan)
    last_t = np.where(cnt > 0, times[rows, np.clip(last_i, 0, T - 1)], np.nan)
    return np.concatenate([cnt, mean, mn, mx, first, last, last - first, last_t, static], 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default='P19')
    ap.add_argument('--tag', required=True)
    ap.add_argument('--max-iter', type=int, default=1000)
    ap.add_argument('--lr', type=float, default=0.05)
    a = ap.parse_args()
    t0 = time.time()
    z = np.load(ROOT / f'data/raindrop/cache/{a.dataset}.npz')
    X = features(z['times'], z['vals'], z['mask'], z['lens'], z['static']); y = z['y']
    rows = []
    for k in range(5):
        tr, va, te = (z[f'split{k}_{p}'] for p in ('train', 'val', 'test'))
        best = None
        for it in (100, 200, 400, 700, a.max_iter):                    # rounds chosen on the official validation split
            clf = HistGradientBoostingClassifier(max_iter=it, learning_rate=a.lr, early_stopping=False, random_state=k)
            clf.fit(X[tr], y[tr])
            v = roc_auc_score(y[va], clf.predict_proba(X[va])[:, 1])
            if best is None or v > best[0]:
                best = (v, it, clf)
        p = best[2].predict_proba(X[te])[:, 1]
        rows.append(dict(split=k, rounds=best[1], val_auroc=best[0], test_auroc=roc_auc_score(y[te], p),
                         test_auprc=average_precision_score(y[te], p)))
        print(json.dumps(rows[-1]), flush=True)
    au = [r['test_auroc'] for r in rows]; ap_ = [r['test_auprc'] for r in rows]
    refs = dict(P19=dict(ours=(0.916, 0.022, 0.639, 0.039), MTM=(0.903, 0.020, 0.583, 0.053)),
                P12=dict(ours=None, MTM=(0.880, 0.010, 0.586, 0.041)))
    res = dict(status='completed', battle='B2', kind='classical summary-statistic diagnostic (no neural training)',
               dataset=a.dataset, tag=a.tag, args=vars(a), n_features=int(X.shape[1]), splits=rows,
               test_auroc_mean=float(np.mean(au)), test_auroc_sd=float(np.std(au, ddof=1)),
               test_auprc_mean=float(np.mean(ap_)), test_auprc_sd=float(np.std(ap_, ddof=1)),
               references=refs.get(a.dataset), wall_s=time.time() - t0,
               source_sha256={'experiments/irts/summary_tree_diagnostic.py':
                              hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    out = ROOT / f'experiments/results/irts/{a.tag}.json'; out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1, default=float) + '\n')
    print('RESULT', json.dumps({k: res[k] for k in ('test_auroc_mean', 'test_auroc_sd', 'test_auprc_mean', 'test_auprc_sd')}))


if __name__ == '__main__':
    main()
