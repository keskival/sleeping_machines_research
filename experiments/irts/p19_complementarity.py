#!/usr/bin/env python3
"""B2: is the P19 win only the statistics? Paired per-split tests (user direction, 7 Oct 23:40 UTC).

For one official split, with the race_irts_v3 final configuration (seed 0, the AWS b2_final_p19_v3 arguments):
1. FULL: race_irts_v3 retrained on curie (an independent rerun of the protocol model), per-record val/test
   probabilities saved;
2. STATS-ONLY: the same network with its temporal features (final state, mean state, learned channel slots) zeroed
   at the head, so it reads only the statistic-valued slots, staleness and statics; same training;
3. TREE: gradient-boosted trees on the same per-channel statistics (summary_tree_diagnostic.py features), rounds
   chosen on the official validation split;
4. BLEND: rank-average of FULL and TREE with the weight chosen on validation from {0, .25, .5, .75, 1}.
TEST is scored once per model. Reading: FULL > STATS-ONLY isolates what the temporal memory adds inside our model;
BLEND > max(FULL, TREE) shows that FULL carries information the summaries do not.
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
from scipy.stats import rankdata
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'experiments/irts'))
from race_irts_v3 import RaceIRTS  # noqa: E402
from summary_tree_diagnostic import features  # noqa: E402


class _MaskedHead(nn.Module):
    def __init__(self, head, keep):
        super().__init__()
        self.inner = head
        self.register_buffer('keep', keep)

    def forward(self, feats):
        return self.inner(feats * self.keep)


class StatsOnlyIRTS(RaceIRTS):
    """Zero the temporal parts of the head input: last_h, mean_h (d each) and the learned channel slots."""

    def __init__(self, C, S, d, modes, layers, J, dv, dropout, n_classes):
        super().__init__(C, S, d, modes, layers, J, dv, dropout, n_classes)
        n_in = self.head[0].in_features
        keep = torch.ones(n_in)
        keep[:2 * d] = 0                                         # last_h, mean_h
        keep[3 * d:3 * d + C * dv] = 0                           # learned channel slots (after static)
        self.head = _MaskedHead(self.head, keep)


def metrics(y, p):
    return dict(auroc=float(roc_auc_score(y, p)), auprc=float(average_precision_score(y, p)))


def train_model(cls, D, split, seed, epochs=40, patience=8, lr=2e-3, wd=1e-4, batch=128):
    torch.set_num_threads(1); random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    tr, va, te = (D[f'split{split}_{p}'] for p in ('train', 'val', 'test'))
    vals, mask = D['vals'], D['mask']; C = vals.shape[2]
    obs = mask[tr]; v = vals[tr]
    nonneg = np.array([(v[..., c][obs[..., c]] >= 0).all() for c in range(C)])
    tv = np.where(nonneg, np.log1p(np.clip(vals, 0, None)), vals)
    mu = np.array([tv[tr][..., c][obs[..., c]].mean() if obs[..., c].any() else 0 for c in range(C)])
    sd = np.array([tv[tr][..., c][obs[..., c]].std() + 1e-6 if obs[..., c].any() else 1 for c in range(C)])
    z = np.clip((tv - mu) / sd, -6, 6) * mask
    st = D['static']; stz = (st - st[tr].mean(0)) / (st[tr].std(0) + 1e-6)
    t = torch.from_numpy(D['times']); Z = torch.from_numpy(z.astype(np.float32)); M = torch.from_numpy(mask)
    Lens = torch.from_numpy(D['lens']); S = torch.from_numpy(stz.astype(np.float32)); y = D['y']
    model = cls(C, S.shape[1], 32, 16, 2, 4, 4, 0.2, 1)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    pw = torch.tensor((y[tr] == 0).sum() / max(1, (y[tr] == 1).sum()), dtype=torch.float32)

    def probs(idx):
        model.eval(); out = []
        with torch.no_grad():
            for i in range(0, len(idx), 256):
                b = torch.from_numpy(idx[i:i + 256]); T = int(Lens[b].max())
                out.append(torch.sigmoid(model(t[b, :T], Z[b, :T], M[b, :T], Lens[b], S[b])[:, 0]))
        return torch.cat(out).numpy()

    best, best_ep, state = -1.0, -1, None
    for ep in range(epochs):
        model.train(); perm = np.random.permutation(tr)
        for i in range(0, len(perm), batch):
            b = torch.from_numpy(perm[i:i + batch]); T = int(Lens[b].max())
            logit = model(t[b, :T], Z[b, :T], M[b, :T], Lens[b], S[b])[:, 0]
            loss = F.binary_cross_entropy_with_logits(logit, torch.from_numpy(y[b.numpy()]).float(), pos_weight=pw)
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        v_auc = roc_auc_score(y[va], probs(va))
        print(json.dumps(dict(model=cls.__name__, epoch=ep, val_auroc=v_auc)), flush=True)
        if v_auc > best:
            best, best_ep, state = v_auc, ep, {k: x.clone() for k, x in model.state_dict().items()}
        if ep - best_ep >= patience:
            break
    model.load_state_dict(state)
    return probs(va), probs(te), best_ep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--split', type=int, required=True)
    ap.add_argument('--tag', required=True)
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    t0 = time.time()
    D = np.load(ROOT / 'data/raindrop/cache/P19.npz')
    y = D['y']; va, te = D[f'split{a.split}_val'], D[f'split{a.split}_test']; tr = D[f'split{a.split}_train']
    pv_full, pt_full, ep_full = train_model(RaceIRTS, D, a.split, a.seed)
    pv_stat, pt_stat, ep_stat = train_model(StatsOnlyIRTS, D, a.split, a.seed)
    X = features(D['times'], D['vals'], D['mask'], D['lens'], D['static'])
    best = None
    for it in (100, 200, 400, 700, 1000):
        clf = HistGradientBoostingClassifier(max_iter=it, learning_rate=0.05, early_stopping=False, random_state=a.split)
        clf.fit(X[tr], y[tr]); v = roc_auc_score(y[va], clf.predict_proba(X[va])[:, 1])
        if best is None or v > best[0]:
            best = (v, it, clf)
    pv_tree, pt_tree = best[2].predict_proba(X[va])[:, 1], best[2].predict_proba(X[te])[:, 1]
    r = lambda p: rankdata(p) / len(p)
    w = max((0, .25, .5, .75, 1), key=lambda w: roc_auc_score(y[va], w * r(pv_full) + (1 - w) * r(pv_tree)))
    pt_blend = w * r(pt_full) + (1 - w) * r(pt_tree)
    res = dict(status='completed', battle='B2', dataset='P19', split=a.split, tag=a.tag, seed=a.seed,
               full=dict(best_epoch=ep_full, test=metrics(y[te], pt_full)),
               stats_only=dict(best_epoch=ep_stat, test=metrics(y[te], pt_stat)),
               tree=dict(rounds=best[1], test=metrics(y[te], pt_tree)),
               blend=dict(weight_full=w, test=metrics(y[te], pt_blend)),
               aws_protocol_model=json.loads((ROOT / f'experiments/results/irts/b2_final_p19_v3_split{a.split}.json').read_text())['test'],
               wall_s=time.time() - t0,
               source_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in
                              (Path(__file__).resolve(), ROOT / 'experiments/irts/race_irts_v3.py',
                               ROOT / 'experiments/irts/summary_tree_diagnostic.py')})
    out = ROOT / f'experiments/results/irts/{a.tag}'
    np.savez_compressed(str(out) + '_preds.npz', y_val=y[va], y_test=y[te], full_val=pv_full, full_test=pt_full,
                        stat_val=pv_stat, stat_test=pt_stat, tree_val=pv_tree, tree_test=pt_tree)
    Path(str(out) + '.json').write_text(json.dumps(res, indent=1) + '\n')
    print('RESULT', json.dumps({k: res[k] for k in ('full', 'stats_only', 'tree', 'blend', 'aws_protocol_model')}), flush=True)


if __name__ == '__main__':
    main()
