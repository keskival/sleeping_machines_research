#!/usr/bin/env python3
"""B5-N node affinity on the large TGB datasets (tgbn-genre, tgbn-reddit, tgbn-token): streaming version of race_affinity
(AWS, 9 Oct 2026). Same scoring and readout idea as v1 (curie); the dense [periods x nodes x classes] tensor is replaced
by streamed addressed state so the 18M-72M-edge datasets fit.

Protocol (TGB examples, e.g. examples/nodeproppred/tgbn-genre/persistant_forecast.py): edges stream in batches of 200
through train, validation and test with one label pointer; when a batch's last timestamp exceeds the pointer's label
time ts, the labels at ts are queried, predicted and scored with Evaluator('ndcg') (k = 10, mean over nodes); afterwards
the labels at ts are REVEALED (the official Persistent Forecast and Moving Average baselines predict from them). Then the
batch's edges enter the state. Our features at ts therefore use only edges of earlier batches and labels revealed at
earlier label times. Checked facts: destination id = class index (0..C-1) and users are numbered from C (py-tgb
pre_process.load_edgelist_datetime / _sr / _token); on tgbn-genre labels are daily, sum to 1 and summarise the following
days (L1 to the next 7 days of edges 1.18 vs 1.40 to the previous 7), so strictly-earlier edges are causal.

State (lazy decay; only touched rows change): per user decayed edge affinity over classes at TAU_DAYS; per user the last
three revealed label vectors and decayed label averages (in label steps); global class shares from edges and labels.
Readout: MLP over per-(user, class) features, softmax race over classes trained against the label distribution (CE,
optionally + LambdaRank as in v1). Training updates online at each training label time; one streaming pass per epoch;
selection on VALIDATION NDCG@10; TEST only with --score-test. Run under run_safe.
"""
import argparse, hashlib, json, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / '.cache/pylib_tgb')); sys.path.insert(0, str(Path(__file__).resolve().parent))
import safe_tgb  # noqa: E402

DAY = 86400.0
TAU_DAYS = np.array([1.0, 3.0, 7.0, 30.0])
TAU_LAB = (2.0, 7.0)                                       # decayed label averages, in label steps
EPS = 1e-4


def rownorm(a):
    s = a.sum(-1, keepdims=True)
    return np.where(s > 0, a / np.maximum(s, 1e-30), 0.0)


class State:
    def __init__(self, U, C):
        self.U, self.C = U, C
        self.E = np.zeros((len(TAU_DAYS), U, C), np.float32); self.elast = np.full(U, np.nan)
        self.G = np.zeros((len(TAU_DAYS), C)); self.glast = np.nan
        self.lab = np.zeros((3, U, C), np.float32); self.has = np.zeros((3, U), bool)
        self.LA = np.zeros((len(TAU_LAB), U, C), np.float32); self.nlab = np.zeros(U)
        self.GL = np.zeros(C)

    def _decay_rows(self, u, t):
        dt = np.where(np.isnan(self.elast[u]), 0.0, t - np.nan_to_num(self.elast[u])) / DAY
        f = np.exp(-dt[None, :] / TAU_DAYS[:, None]).astype(np.float32)               # [K, n]
        self.E[:, u] *= f[..., None]; self.elast[u] = t

    def add_edges(self, u, c, w, t):
        """u: user rows, c: classes, w: weights; decays touched rows to the batch's last time t (taus are days, a batch
        spans minutes, so in-batch decay is negligible)."""
        uu = np.unique(u); self._decay_rows(uu, t)
        for k in range(len(TAU_DAYS)):
            np.add.at(self.E[k], (u, c), w)
        gd = 0.0 if np.isnan(self.glast) else (t - self.glast) / DAY
        self.G *= np.exp(-gd / TAU_DAYS)[:, None]; np.add.at(self.G, (slice(None), c), np.broadcast_to(w, (len(TAU_DAYS), len(w)))); self.glast = t

    def reveal(self, u, y):
        self.lab[2, u] = self.lab[1, u]; self.lab[1, u] = self.lab[0, u]; self.lab[0, u] = y
        self.has[2, u] = self.has[1, u]; self.has[1, u] = self.has[0, u]; self.has[0, u] = True
        for i, tau in enumerate(TAU_LAB):
            a = np.exp(-1.0 / tau)
            self.LA[i, u] = a * self.LA[i, u] + (1 - a) * y
        self.nlab[u] += 1; self.GL = 0.9 * self.GL + 0.1 * y.mean(0)

    def features(self, u, t):
        dt = np.where(np.isnan(self.elast[u]), 1e9, t - np.nan_to_num(self.elast[u])) / DAY
        f = np.exp(-dt[None, :] / TAU_DAYS[:, None]).astype(np.float32)
        E = [rownorm(self.E[k, u] * f[k][:, None]) for k in range(len(TAU_DAYS))]
        vol = np.log1p(self.E[2, u].sum(-1) * f[2])
        gd = 0.0 if np.isnan(self.glast) else (t - self.glast) / DAY
        G = self.G[2] * np.exp(-gd / TAU_DAYS[2]); G = G / max(G.sum(), 1e-30)
        L = [self.lab[j, u] for j in range(3)] + [rownorm(self.LA[i, u]) for i in range(len(TAU_LAB))]
        n, C = len(u), self.C
        props = np.stack(L + E + [np.broadcast_to(G, (n, C)), np.broadcast_to(self.GL / max(self.GL.sum(), 1e-30), (n, C))], -1)
        growth = np.log((self.lab[0, u] + EPS) / (self.lab[1, u] + EPS))
        ctx = np.stack([vol, np.log1p(np.minimum(dt, 1e4)), np.log1p(self.nlab[u]), self.has[0, u].astype(float)], -1)
        x = np.concatenate([props, np.log(props + EPS), growth[..., None], np.broadcast_to(ctx[:, None], (n, C, 4))], -1)
        return x.astype(np.float32)


class Readout(nn.Module):
    def __init__(self, nf, h):
        super().__init__()
        self.norm = nn.LayerNorm(nf); self.mlp = nn.Sequential(nn.Linear(nf, h), nn.GELU(), nn.Linear(h, h), nn.GELU(), nn.Linear(h, 1))

    def forward(self, x):
        return self.mlp(self.norm(x)).squeeze(-1)


def run_stream(ds, ev, model, opt, splits, C, U, train_ok, bs=200, max_label_times=0):
    """One streaming pass over the given splits; returns per-split mean NDCG over fired label times."""
    d = ds.full_data; src_all = d['sources'].astype(np.int64); dst_all = d['destinations'].astype(np.int64)
    t_all = d['timestamps'].astype(float); w_all = np.asarray(d['edge_feat'], float).reshape(len(t_all), -1)[:, 0]
    st = State(U, C); ds.reset_label_time(); out = {}; fired = 0
    for name, mask in splits:
        idx = np.where(mask)[0]; res = []; label_t = ds.return_label_ts()
        for b in range(0, len(idx), bs):
            ix = idx[b:b + bs]; tq = t_all[ix][-1]
            if tq > label_t:
                r = ds.find_next_labels_batch(tq)
                if r is None:
                    break
                lt, nodes, y = r[0][0], np.asarray(r[1], np.int64), np.asarray(r[2], np.float32)
                u = nodes - C; assert u.min() >= 0
                x = torch.from_numpy(st.features(u, float(lt))); yt = torch.from_numpy(y)
                if name == 'train' and train_ok and opt is not None:
                    model.train(); sc = model(x); loss = -(yt * torch.log_softmax(sc, -1)).sum(-1).mean()
                    opt.zero_grad(); loss.backward(); opt.step()
                else:
                    model.eval()
                    with torch.no_grad():
                        pred = model(x).numpy()
                    res.append(float(ev.eval({'y_true': y, 'y_pred': pred, 'eval_metric': ['ndcg']})['ndcg']))
                st.reveal(u, y); label_t = ds.return_label_ts(); fired += 1
                if max_label_times and fired >= max_label_times:
                    break
            s = src_all[ix] - C; keep = s >= 0
            st.add_edges(s[keep], dst_all[ix][keep], w_all[ix][keep], tq)
        out[name] = (float(np.mean(res)) if res else None, len(res))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', required=True); ap.add_argument('--dataset', default='tgbn-genre')
    ap.add_argument('--root', default='../../../data/tgb_aws/root'); ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--epochs', type=int, default=4); ap.add_argument('--hidden', type=int, default=32)
    ap.add_argument('--lr', type=float, default=3e-3); ap.add_argument('--max-label-times', type=int, default=0)
    ap.add_argument('--score-test', action='store_true')
    a = ap.parse_args(); t0 = time.time(); torch.manual_seed(a.seed); np.random.seed(a.seed); torch.set_num_threads(1)
    safe_tgb.patch_tgb()
    from tgb.nodeproppred.dataset import NodePropPredDataset
    from tgb.nodeproppred.evaluate import Evaluator
    ds = NodePropPredDataset(name=a.dataset, root=a.root, preprocess=True); ev = Evaluator(name=a.dataset)
    d = ds.full_data; C = ds.num_classes
    assert int(d['destinations'].max()) < C and int(d['sources'].min()) >= C, 'expected destination id = class index, users >= C'
    U = int(d['sources'].max()) + 1 - C
    nf = 2 * (3 + len(TAU_LAB) + len(TAU_DAYS) + 2) + 1 + 4
    model = Readout(nf, a.hidden); opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-3)
    od = ROOT / 'experiments/results/tgb'; od.mkdir(parents=True, exist_ok=True); ckpt = od / f'{a.tag}.pt'
    # training-free reference through the same stream: persistent forecast (last revealed label) on validation
    pf = run_stream(ds, ev, _PF(), None, [('train', ds.train_mask), ('val', ds.val_mask)], C, U, False, max_label_times=a.max_label_times)
    print('persistent forecast (our replay)', json.dumps(pf), flush=True)
    hist, best = [], -1.0
    for ep in range(a.epochs):
        e0 = time.time()
        r = run_stream(ds, ev, model, opt, [('train', ds.train_mask), ('val', ds.val_mask)], C, U, True, max_label_times=a.max_label_times)
        hist.append(dict(epoch=ep, val_ndcg=r['val'][0], val_label_times=r['val'][1], epoch_s=time.time() - e0)); print(json.dumps(hist[-1]), flush=True)
        if r['val'][0] is not None and r['val'][0] > best:
            best = r['val'][0]; torch.save(model.state_dict(), ckpt)
    res = dict(status='completed', battle='B5', tag=a.tag, dataset=a.dataset, model='race_affinity v2 (streaming)', args=vars(a),
               classes=C, users=U, features=nf, parameters=sum(p.numel() for p in model.parameters()),
               persistent_forecast_val_ndcg=pf['val'][0], best_val_ndcg=best, history=hist, checkpoint=str(ckpt.relative_to(ROOT)),
               wall_s=time.time() - t0)
    if a.score_test:
        model.load_state_dict(torch.load(ckpt))
        r = run_stream(ds, ev, model, None, [('train', ds.train_mask), ('val', ds.val_mask), ('test', ds.test_mask)], C, U, False)
        res['test_ndcg'] = r['test'][0]; res['test_label_times'] = r['test'][1]
    res['source_sha256'] = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                            for p in (Path(__file__).resolve(), Path(__file__).resolve().parent / 'safe_tgb.py')}
    (od / f'{a.tag}.json').write_text(json.dumps(res, indent=1) + '\n')
    print('RESULT', json.dumps({k: res.get(k) for k in ('persistent_forecast_val_ndcg', 'best_val_ndcg', 'test_ndcg', 'wall_s')}), flush=True)


class _PF(nn.Module):
    """Persistent forecast through the same stream: score = last revealed label (feature 0)."""
    def forward(self, x):
        return x[..., 0]


if __name__ == '__main__':
    main()
