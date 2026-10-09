#!/usr/bin/env python3
"""B5 node affinity (TGB tgbn-*, first tgbn-trade): predict a node's next-period affinity vector over destinations;
metric NDCG@10, official protocol (py-tgb NodePropPredDataset + Evaluator).

Protocol, reproduced exactly from TGB's examples (examples/nodeproppred/*/persistant_forecast.py): edges are streamed in
batches of 200 through the train, validation and test splits in order with ONE label pointer; whenever a batch's last
timestamp exceeds the pointer's label time ts, the labels at ts are queried and scored with Evaluator('ndcg') (sklearn
ndcg_score, k=10, averaged over the nodes of that label time); a split's score is the mean over its fired label times.
The label at ts summarises the edges of the period starting at ts (verified on tgbn-trade: label(1992) = normalised
1992 flows). Causality here: the prediction for ts uses only edges with t < ts.

Model (race over destinations): per (node, destination) features from addressed temporal state, all strictly before ts:
the last four period vectors (persistent forecast and moving average are special cases), exponentially decayed
affinities at four time constants (in periods), the reverse flow (destination -> node), the destination's global share
and growth, the pair's growth, and node context (volume, growth, periods since active). A small MLP scores each
destination; P(next unit of affinity goes to c) = softmax over all destinations (the race of destination clocks);
loss = cross-entropy against the realised affinity distribution. Unlike convex combinations of past vectors (NAVIS,
moving averages), the nonlinear readout can extrapolate trends. Selection on VALIDATION NDCG; test scored once with
--score-test. Run under run_safe.
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / '.cache/pylib_tgb'))                                  # py-tgb 2.3.0 (python -m pip --target)
TAUS = (1.0, 2.0, 4.0, 8.0)
EPS = 1e-4


def rownorm(a):
    s = a.sum(-1, keepdims=True)
    return np.where(s > 0, a / np.maximum(s, 1e-30), 0.0)


def fired_label_times(ds, split_masks, bs=200):
    """Replay TGB's evaluation loop: which label times fire in which split (one pointer, batches of 200)."""
    t_all = ds.full_data['timestamps']; ds.reset_label_time(); fired = {}
    for name, mask in split_masks:
        ts = t_all[mask]; label_t = ds.return_label_ts(); out = []
        for b in range(0, len(ts), bs):
            q = ts[b:b + bs][-1]
            if q > label_t:
                r = ds.find_next_labels_batch(q)
                if r is None:
                    break
                out.append(int(r[0][0])); label_t = ds.return_label_ts()
        fired[name] = out
    return fired


class Periods:
    """Per-period aggregates A[k] = summed edge weights (src, class) of edges in [L_k, L_{k+1}); A[-1] = before L_0."""

    def __init__(self, ds, cls_of_dst):
        d = ds.full_data; self.L = np.sort(np.asarray(ds.label_ts)).astype(float)
        t = d['timestamps'].astype(float); w = np.asarray(d['edge_feat'], float).reshape(len(t), -1)[:, 0]
        k = np.searchsorted(self.L, t, side='right') - 1                               # -1 = before the first label time
        src, dst = d['sources'].astype(np.int64), d['destinations'].astype(np.int64)  # stored as floats in tgbn-trade
        self.N = int(max(src.max(), dst.max())) + 1; self.C = ds.num_classes
        self.A = np.zeros((len(self.L) + 1, self.N, self.C), np.float32)               # index k+1
        c = cls_of_dst[dst]; ok = c >= 0
        np.add.at(self.A, (k[ok] + 1, src[ok], c[ok]), w[ok])
        self.Arev = None
        if self.N == self.C:                                                           # node-to-node affinity: reverse flow
            self.Arev = np.transpose(self.A, (0, 2, 1))

    def features(self, ts, nodes):
        """phi[n, c, f] for label time ts using periods strictly before ts."""
        k = int(np.searchsorted(self.L, ts)) + 1                                       # A index of the period starting at ts
        past = self.A[:k]                                                              # all periods before ts
        def lag(j, arr=past):
            return arr[k - j] if k - j >= 0 else np.zeros_like(arr[0])
        f = []
        P = [rownorm(lag(j)[nodes]) for j in (1, 2, 3, 4)]; f += P
        ages = np.arange(k)[::-1].astype(float)                                        # 0 = most recent period
        E = []
        for tau in TAUS:
            wts = np.exp(-ages / tau).astype(np.float32)
            E.append(np.tensordot(wts, past[:, nodes], axes=(0, 0)))
        f += [rownorm(e) for e in E]
        if self.Arev is not None:
            R1 = rownorm(self.Arev[k - 1][nodes]) if k >= 1 else np.zeros_like(P[0])
            wts = np.exp(-ages / 4.0).astype(np.float32); R4 = rownorm(np.tensordot(wts, self.Arev[:k][:, nodes], axes=(0, 0)))
        else:
            R1 = R4 = np.zeros_like(P[0])
        f += [R1, R4]
        g1 = lag(1).sum(0); g2 = lag(2).sum(0); gE = np.tensordot(np.exp(-ages / 4.0).astype(np.float32), past.sum(1), axes=(0, 0))
        G = [np.broadcast_to(x / max(x.sum(), 1e-30), P[0].shape) for x in (g1, gE)]; f += G
        props = np.stack(f, -1)                                                        # [n, C, 12]
        growth_g = np.broadcast_to(np.log((g1 + 1) / (g2 + 1)), P[0].shape)
        growth_p = np.log((P[0] + EPS) / (P[1] + EPS))
        vol1 = lag(1)[nodes].sum(-1); vol2 = lag(2)[nodes].sum(-1)
        active = (past[:, nodes].sum(-1) > 0)                                          # [k, n]
        since = np.array([k - 1 - np.max(np.flatnonzero(active[:, i])) if active[:, i].any() else k + 1 for i in range(len(nodes))], float)
        ctx = np.stack([np.log1p(vol1), np.log((vol1 + 1) / (vol2 + 1)), np.log1p(since)], -1)
        ctx = np.broadcast_to(ctx[:, None], P[0].shape + (3,))
        x = np.concatenate([props, np.log(props + EPS), growth_g[..., None], growth_p[..., None], ctx], -1)
        return x.astype(np.float32)


class Readout(nn.Module):
    def __init__(self, nf, h):
        super().__init__()
        self.norm = nn.LayerNorm(nf); self.mlp = nn.Sequential(nn.Linear(nf, h), nn.GELU(), nn.Linear(h, h), nn.GELU(), nn.Linear(h, 1))

    def forward(self, x):
        return self.mlp(self.norm(x)).squeeze(-1)


def lambda_loss(score, y, top=20):
    """LambdaRank-style pairwise loss for NDCG with sklearn's linear gains: pairs (i in the label's top-k, j with y_j <
    y_i), weighted by |y_i - y_j| * |1/log2(1+r_i) - 1/log2(1+r_j)| at the current (detached) predicted ranks."""
    k = min(top, y.shape[-1]); yi, ii = y.topk(k, -1); si = score.gather(-1, ii)            # [n, k]
    with torch.no_grad():
        rank = score.argsort(-1, descending=True).argsort(-1).float() + 1                  # [n, C]
        disc = 1.0 / torch.log2(1 + rank); di = disc.gather(-1, ii)
        w = (yi[..., None] - y[:, None, :]).clamp_min(0) * (di[..., None] - disc[:, None, :]).abs()
    return (w * nn.functional.softplus(-(si[..., None] - score[:, None, :]))).sum((-1, -2)).mean()


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--tag', required=True); ap.add_argument('--dataset', default='tgbn-trade')
    ap.add_argument('--seed', type=int, default=0); ap.add_argument('--epochs', type=int, default=200)
    ap.add_argument('--hidden', type=int, default=32); ap.add_argument('--lr', type=float, default=3e-3)
    ap.add_argument('--wd', type=float, default=1e-3); ap.add_argument('--loss', choices=('ce', 'lambda', 'ce+lambda'), default='ce')
    ap.add_argument('--score-test', action='store_true')
    ap.add_argument('--root', default=str(ROOT / 'data/tgb'))
    a = ap.parse_args(); torch.manual_seed(a.seed); np.random.seed(a.seed); torch.set_num_threads(1); t0 = time.time()
    from tgb.nodeproppred.dataset import NodePropPredDataset
    from tgb.nodeproppred.evaluate import Evaluator
    ds = NodePropPredDataset(name=a.dataset, root=a.root, preprocess=True); ev = Evaluator(name=a.dataset)
    d = ds.full_data
    if a.dataset != 'tgbn-trade':
        raise SystemExit('class mapping verified for tgbn-trade only; verify label index <-> destination id first')
    cls_of_dst = np.arange(int(d['destinations'].max()) + 1, dtype=np.int64)                          # verified: label index = node id
    fired = fired_label_times(ds, [('train', ds.train_mask), ('val', ds.val_mask), ('test', ds.test_mask)])
    print('fired label times', fired, flush=True)
    per = Periods(ds, cls_of_dst); L = ds.label_dict

    def batch(ts):
        nodes = np.array(list(L[ts].keys()), dtype=np.int64); y = np.stack([np.asarray(L[ts][n], np.float32) for n in nodes])
        return torch.from_numpy(per.features(ts, nodes)), torch.from_numpy(y)

    data = {s: [(ts,) + batch(ts) for ts in fired[s]] for s in ('train', 'val')}
    if a.score_test:
        data['test'] = [(ts,) + batch(ts) for ts in fired['test']]
    nf = data['train'][0][1].shape[-1]
    model = Readout(nf, a.hidden); opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)

    @torch.no_grad()
    def score(split, fn=None):
        model.eval(); res = []
        for ts, x, y in data[split]:
            pred = model(x).numpy() if fn is None else fn(x)
            res.append(float(ev.eval({'y_true': y.numpy(), 'y_pred': pred, 'eval_metric': ['ndcg']})['ndcg']))
        return float(np.mean(res)), res

    pf_val = score('val', fn=lambda x: x[..., 0].numpy())[0]                          # persistent forecast via feature 0
    print(f'features {nf} params {sum(p.numel() for p in model.parameters())} persistent-forecast val {pf_val:.4f}', flush=True)
    hist, best, state = [], -1.0, None
    for ep in range(a.epochs):
        model.train(); tot = 0.0
        for i in np.random.permutation(len(data['train'])):
            _, x, y = data['train'][i]; sc = model(x); loss = 0.0
            if 'ce' in a.loss:
                loss = loss - (y * torch.log_softmax(sc, -1)).sum(-1).mean()
            if 'lambda' in a.loss:
                loss = loss + lambda_loss(sc, y)
            opt.zero_grad(); loss.backward(); opt.step(); tot += float(loss)
        val, _ = score('val'); hist.append(dict(epoch=ep, train_ce=tot / len(data['train']), val_ndcg=val))
        if ep % 10 == 0:
            print(json.dumps(hist[-1]), flush=True)
        if val > best:
            best, state = val, {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(state)
    out = dict(status='completed', battle='B5', tag=a.tag, dataset=a.dataset, model='race_affinity v1', args=vars(a),
               fired_label_times=fired, features=nf, parameters=sum(p.numel() for p in model.parameters()),
               persistent_forecast_val_ndcg=pf_val, best_val_ndcg=best, best_epoch=int(np.argmax([h['val_ndcg'] for h in hist])),
               val_per_label_time=score('val')[1], history=hist[::10], test_ndcg='not scored (development run)',
               wall_s=time.time() - t0, source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    if a.score_test:
        out['test_ndcg'], out['test_per_label_time'] = score('test')
        out['persistent_forecast_test_ndcg'] = score('test', fn=lambda x: x[..., 0].numpy())[0]
    (ROOT / f'experiments/results/tgb/{a.tag}.json').write_text(json.dumps(out, indent=1) + '\n')
    print('RESULT', json.dumps({k: out[k] for k in ('persistent_forecast_val_ndcg', 'best_val_ndcg', 'best_epoch', 'test_ndcg', 'wall_s')}))


if __name__ == '__main__':
    main()
