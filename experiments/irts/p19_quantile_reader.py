#!/usr/bin/env python3
"""B2 development (curie, 8 Oct): a tree-strength statistics reader inside the event-native P19 model.

Finding (B2_IRREGULAR_TS.md, P19 complementarity): our network reading only its statistic slots scores 0.900 AUROC,
below gradient-boosted trees on the same statistics (0.914), while the temporal memory adds +0.017 / +0.072 that trees
lack. Trees read statistics through threshold comparisons, so they are invariant to monotone transforms and to heavy
tails; the network reads z-scored raw statistics. Change: each per-record statistic (summary_tree_diagnostic.features:
count, mean, min, max, first, last, last − first, last time per channel, plus statics) is encoded by comparison with
the TRAIN quantiles of that statistic (empirical CDF value, missing → its own flag), i.e. typed comparisons against
data-set thresholds, and the encoded vector enters the head beside the temporal features. Everything else is
race_irts_v3 (temporal memory, addressed channel slots, typed channel comparisons, silence features).
Validation-only by default; --score-test scores TEST once at the selected checkpoint.
"""
import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'experiments/irts'))
from race_irts_v3 import RaceIRTS  # noqa: E402
from summary_tree_diagnostic import features  # noqa: E402


def quantile_encode(X, tr, n_q=64):
    """Empirical-CDF encoding fitted on TRAIN rows; NaN -> 0 with a separate missing flag."""
    out = np.zeros_like(X, dtype=np.float32); miss = np.isnan(X)
    for j in range(X.shape[1]):
        col = X[tr, j]; col = col[~np.isnan(col)]
        if len(col) == 0:
            continue
        qs = np.unique(np.quantile(col, np.linspace(0, 1, n_q + 1)))
        v = X[:, j]; ok = ~np.isnan(v)
        out[ok, j] = np.interp(v[ok], qs, np.linspace(0, 1, len(qs))) if len(qs) > 1 else 0.5
    keep = miss[tr].any(0)                                     # missing flags only where TRAIN has missing values
    return np.concatenate([out, miss[:, keep].astype(np.float32)], 1)


class QuantileReaderIRTS(RaceIRTS):
    def __init__(self, C, S, d, modes, layers, J, dv, dropout, n_classes, n_q_feats):
        super().__init__(C, S, d, modes, layers, J, dv, dropout, n_classes)
        self.q_reader = nn.Sequential(nn.Linear(n_q_feats, 2 * d), nn.GELU(), nn.Dropout(dropout), nn.Linear(2 * d, 2 * d))
        old = self.head[0]
        self.head[0] = nn.Linear(old.in_features + 2 * d, old.out_features)
        self._q = None

    def forward(self, t, z, mask, lens, static, q):
        self._q = q
        return super().forward(t, z, mask, lens, static)

    def _head_in(self, feats):
        return torch.cat([feats, F.gelu(self.q_reader(self._q))], -1)


class _HeadWrap(nn.Module):
    def __init__(self, owner, head):
        super().__init__()
        self.inner = head
        object.__setattr__(self, 'owner', owner)

    def forward(self, feats):
        return self.inner(self.owner._head_in(feats))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--split', type=int, required=True)
    ap.add_argument('--tag', required=True)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--score-test', action='store_true')
    a = ap.parse_args()
    t0 = time.time(); torch.set_num_threads(1); random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    D = np.load(ROOT / 'data/raindrop/cache/P19.npz')
    tr, va, te = (D[f'split{a.split}_{p}'] for p in ('train', 'val', 'test'))
    vals, mask = D['vals'], D['mask']; C = vals.shape[2]
    obs = mask[tr]; v = vals[tr]
    nonneg = np.array([(v[..., c][obs[..., c]] >= 0).all() for c in range(C)])
    tv = np.where(nonneg, np.log1p(np.clip(vals, 0, None)), vals)
    mu = np.array([tv[tr][..., c][obs[..., c]].mean() if obs[..., c].any() else 0 for c in range(C)])
    sd = np.array([tv[tr][..., c][obs[..., c]].std() + 1e-6 if obs[..., c].any() else 1 for c in range(C)])
    z = np.clip((tv - mu) / sd, -6, 6) * mask
    st = D['static']; stz = (st - st[tr].mean(0)) / (st[tr].std(0) + 1e-6)
    Q = quantile_encode(features(D['times'], vals, mask, D['lens'], st), tr)
    t = torch.from_numpy(D['times']); Z = torch.from_numpy(z.astype(np.float32)); M = torch.from_numpy(mask)
    Lens = torch.from_numpy(D['lens']); S = torch.from_numpy(stz.astype(np.float32)); Qt = torch.from_numpy(Q); y = D['y']
    model = QuantileReaderIRTS(C, S.shape[1], 32, 16, 2, 4, 4, 0.2, 1, Q.shape[1])
    model.head = nn.Sequential(_HeadWrap(model, model.head))
    params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    pw = torch.tensor((y[tr] == 0).sum() / max(1, (y[tr] == 1).sum()), dtype=torch.float32)

    def probs(idx):
        model.eval(); out = []
        with torch.no_grad():
            for i in range(0, len(idx), 256):
                b = torch.from_numpy(idx[i:i + 256]); T = int(Lens[b].max())
                out.append(torch.sigmoid(model(t[b, :T], Z[b, :T], M[b, :T], Lens[b], S[b], Qt[b])[:, 0]))
        return torch.cat(out).numpy()

    best, best_ep, state, hist = -1.0, -1, None, []
    for ep in range(40):
        model.train(); perm = np.random.permutation(tr)
        for i in range(0, len(perm), 128):
            b = torch.from_numpy(perm[i:i + 128]); T = int(Lens[b].max())
            logit = model(t[b, :T], Z[b, :T], M[b, :T], Lens[b], S[b], Qt[b])[:, 0]
            loss = F.binary_cross_entropy_with_logits(logit, torch.from_numpy(y[b.numpy()]).float(), pos_weight=pw)
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        pv = probs(va); va_auc = roc_auc_score(y[va], pv); hist.append(dict(epoch=ep, val_auroc=va_auc,
                                                                             val_auprc=average_precision_score(y[va], pv)))
        print(json.dumps(hist[-1]), flush=True)
        if va_auc > best:
            best, best_ep, state = va_auc, ep, {k: x.clone() for k, x in model.state_dict().items()}
        if ep - best_ep >= 8:
            break
    model.load_state_dict(state)
    pv = probs(va)
    res = dict(status='completed', battle='B2', dataset='P19', split=a.split, tag=a.tag, seed=a.seed, parameters=params,
               best_epoch=best_ep, history=hist, val=dict(auroc=roc_auc_score(y[va], pv), auprc=average_precision_score(y[va], pv)),
               wall_s=time.time() - t0,
               source_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in
                              (Path(__file__).resolve(), ROOT / 'experiments/irts/race_irts_v3.py',
                               ROOT / 'experiments/irts/summary_tree_diagnostic.py')})
    if a.score_test:
        pt = probs(te); res['test'] = dict(auroc=roc_auc_score(y[te], pt), auprc=average_precision_score(y[te], pt))
    (ROOT / f'experiments/results/irts/{a.tag}.json').write_text(json.dumps(res, indent=1, default=float) + '\n')
    print('RESULT', json.dumps({k: res.get(k) for k in ('val', 'test', 'best_epoch', 'parameters')}, default=float), flush=True)


if __name__ == '__main__':
    main()
