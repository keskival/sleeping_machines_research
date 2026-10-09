#!/usr/bin/env python3
"""B5 native model v2 for TGB dynamic link prediction (tgbl-wiki-v2). v2 = v1 plus (round-2 diagnosis of the v1 smoke:
repeat pairs 0.874 MRR at 86% share, new pairs 0.125, new sources 0.047) ordinal recency of the candidate among the
source's pages, the pair's share of the source's activity, a last-destination flag, and decayed page-to-page
transitions T[last(s), c] (which page sources move to next), a sequential key read that also scores new pairs.

v1 description: a race over every candidate destination, driven by
addressed temporal memory.

State (sparse, addressed, exact; updated event by event in time order):
  pair slots   z[s, c] = sum over past (s, c) events of exp((-r_f + i w_f)(t - t_k)) for a bank of F clocks (decays from
               a minute to a month; daily and weekly rotations), plus count and last time. Only touched pairs exist.
  destination  decayed in-degree of c at several time constants, last time of c.
  co-visit     W[u, p] decayed user-page weights and C[p, q] = sum_u W[u, p] W[u, q] (page-page co-visitation, kept
               incrementally); the source's three-step evidence for c is (W[s] C)[c] / |W[s]|.
  source       last time and count of s (context broadcast to all candidates).
Readout: score(s, c, t) = MLP(phi(s, c, t)); P(next destination = c) = softmax over ALL candidate destinations, i.e. the
race of destination clocks with hazards exp(score); training maximises the log-probability of the realised winner (the
losers all receive credit). The query keys the source's slot row by candidate (keys = (s, c) addresses, values = clock
sums). Training replays the train stream each epoch (features use only events before the current batch); selection on
VALIDATION MRR with the official negatives; TEST is scored once with --score-test. Run under run_safe.
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / '.cache/pylib_tgb'))                                  # py-tgb 2.3.0 (python -m pip --target)
DAY, WEEK = 86400.0, 7 * 86400.0
# pair clock bank: (time constant s, angular frequency rad/s)
CLOCKS = [(60.0, 0.0), (600.0, 0.0), (3600.0, 0.0), (6 * 3600.0, 0.0), (DAY, 0.0), (3 * DAY, 0.0), (WEEK, 0.0), (30 * DAY, 0.0),
          (DAY, 2 * math.pi / DAY), (3 * DAY, 2 * math.pi / DAY), (WEEK, 2 * math.pi / DAY), (30 * DAY, 2 * math.pi / WEEK)]
POP_TAUS = (3600.0, DAY, WEEK, 30 * DAY)
COV_TAUS = (DAY, 30 * DAY)


def slog(x):
    return np.sign(x) * np.log1p(np.abs(x))


class Memory:
    def __init__(self, n_src, n_dst):
        self.rate = np.array([1 / tau for tau, _ in CLOCKS]); self.freq = np.array([w for _, w in CLOCKS])
        self.n_src, self.n_dst = n_src, n_dst
        self.pair = {}; self.rows = [dict() for _ in range(n_src)]                    # (s,c) -> idx ; s -> {c: idx}
        cap = 1 << 15
        self.z = np.zeros((cap, len(CLOCKS)), np.complex128); self.cnt = np.zeros(cap); self.plast = np.zeros(cap)
        self.pop = np.zeros((len(POP_TAUS), n_dst)); self.poplast = np.zeros(n_dst); self.dlast = np.full(n_dst, -np.inf)
        self.W = np.zeros((len(COV_TAUS), n_src, n_dst), np.float32); self.wlast = np.zeros(n_src)
        self.C = np.zeros((len(COV_TAUS), n_dst, n_dst), np.float32); self.clast = 0.0
        self.slast = np.full(n_src, -np.inf); self.scnt = np.zeros(n_src)
        self.last_dst = np.full(n_src, -1, np.int64)
        self.T = np.zeros((len(COV_TAUS), n_dst, n_dst), np.float32)                 # decayed transitions prev -> next

    def _w(self, u, t):
        return self.W[:, u] * np.exp(-(t - self.wlast[u]) / np.array(COV_TAUS))[:, None].astype(np.float32)

    def add(self, s, c, t):
        dec = np.exp(-(t[-1] - self.clast) / np.array(COV_TAUS)).astype(np.float32) if len(t) else None
        for k in range(len(s)):
            u, p, tk = int(s[k]), int(c[k]), float(t[k])
            idx = self.pair.get((u, p))
            if idx is None:
                idx = len(self.pair)
                if idx >= len(self.cnt):
                    self.z = np.concatenate([self.z, np.zeros_like(self.z)]); self.cnt = np.concatenate([self.cnt, np.zeros_like(self.cnt)])
                    self.plast = np.concatenate([self.plast, np.zeros_like(self.plast)])
                self.pair[(u, p)] = idx; self.rows[u][p] = idx
            dt = tk - self.plast[idx]
            self.z[idx] = self.z[idx] * np.exp((-self.rate + 1j * self.freq) * dt) + 1.0
            self.cnt[idx] += 1; self.plast[idx] = tk
            self.pop[:, p] = self.pop[:, p] * np.exp(-(tk - self.poplast[p]) / np.array(POP_TAUS)) + 1.0
            self.poplast[p] = tk; self.dlast[p] = tk
            wu = self._w(u, tk)                                                       # co-visitation before adding (u,p)
            self.C[:, p, :] += wu; self.C[:, :, p] += wu
            wu[:, p] += 1.0; self.W[:, u] = wu; self.wlast[u] = tk
            self.slast[u] = tk; self.scnt[u] += 1
            if self.last_dst[u] >= 0:
                self.T[:, self.last_dst[u], p] += 1.0
            self.last_dst[u] = p
        if dec is not None:                                                           # batch-level lazy decay of C and T
            self.C *= dec[:, None, None]; self.T *= dec[:, None, None]; self.clast = float(t[-1])

    def features(self, s, t):
        """phi for queries (s[b], t[b]) against every destination: float32 [B, n_dst, n_feat]."""
        B, D = len(s), self.n_dst; nf = 2 * len(CLOCKS) + 6
        pairf = np.zeros((B, D, nf), np.float32)
        pairf[:, :, -4] = slog(1e7)                                                   # time since last pair event: "never"
        pairf[:, :, -3] = np.log1p(1e4)                                               # recency rank: "never"
        for b in range(B):
            row = self.rows[int(s[b])]
            if not row:
                continue
            cs = np.fromiter(row.keys(), np.int64); ix = np.fromiter(row.values(), np.int64)
            dt = t[b] - self.plast[ix]
            z = self.z[ix] * np.exp((-self.rate + 1j * self.freq)[None] * dt[:, None])
            pairf[b, cs, :len(CLOCKS)] = slog(z.real); pairf[b, cs, len(CLOCKS):2 * len(CLOCKS)] = slog(z.imag)
            pairf[b, cs, -6] = np.log1p(self.cnt[ix]); pairf[b, cs, -5] = 1.0; pairf[b, cs, -4] = slog(dt)
            rank = np.argsort(np.argsort(dt))                                          # 0 = most recently touched page
            pairf[b, cs, -3] = np.log1p(rank); pairf[b, cs, -2] = self.cnt[ix] / max(self.scnt[int(s[b])], 1.0)
            ld = self.last_dst[int(s[b])]
            if ld >= 0:
                pairf[b, ld, -1] = 1.0
        popd = t[:, None, None] - self.poplast[None, :, None]
        pop = np.log1p(self.pop.T[None] * np.exp(-popd / np.array(POP_TAUS)))         # [B, D, P]
        dl = slog(np.minimum(t[:, None] - self.dlast[None], 1e7))[..., None]
        cdec = np.exp(-(t[:, None] - self.clast) / np.array(COV_TAUS))                 # [B, Tc]
        cov = []
        for j in range(len(COV_TAUS)):
            wsj = self.W[j, s] * np.exp(-(t - self.wlast[s]) / COV_TAUS[j])[:, None]  # [B, D]
            v = (wsj @ self.C[j]) * cdec[:, j:j + 1] / np.maximum(wsj.sum(1, keepdims=True), 1e-6)
            cov.append(np.log1p(np.maximum(v, 0)))
        ld = self.last_dst[s]; trans = []
        for j in range(len(COV_TAUS)):
            row = np.where((ld >= 0)[:, None], self.T[j][np.maximum(ld, 0)], 0.0) * cdec[:, j:j + 1]
            trans.append(np.log1p(row / np.maximum(row.sum(1, keepdims=True), 1e-6) * 100.0))
        cov = np.stack(cov + trans, -1)
        src = np.stack([slog(np.minimum(t - self.slast[s], 1e7)), np.log1p(self.scnt[s])], -1)  # [B, 2]
        src = np.broadcast_to(src[:, None], (B, D, 2))
        return np.concatenate([pairf, pop, dl, cov, src], -1).astype(np.float32)


class Readout(nn.Module):
    def __init__(self, nf, h):
        super().__init__()
        self.norm = nn.LayerNorm(nf); self.mlp = nn.Sequential(nn.Linear(nf, h), nn.GELU(), nn.Linear(h, h), nn.GELU(), nn.Linear(h, 1))

    def forward(self, x):
        return self.mlp(self.norm(x)).squeeze(-1)


def rr(v):
    return 1.0 / (0.5 * ((v[1:] > v[0]).sum() + (v[1:] >= v[0]).sum()) + 1)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--tag', required=True); ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--epochs', type=int, default=8); ap.add_argument('--hidden', type=int, default=64)
    ap.add_argument('--lr', type=float, default=3e-3); ap.add_argument('--bs', type=int, default=200)
    ap.add_argument('--max-train-events', type=int, default=0); ap.add_argument('--max-eval-events', type=int, default=0)
    ap.add_argument('--score-test', action='store_true'); ap.add_argument('--max-wall-s', type=float, default=1e9)
    ap.add_argument('--root', default=str(ROOT / 'data/tgb'))
    a = ap.parse_args(); torch.manual_seed(a.seed); np.random.seed(a.seed); torch.set_num_threads(1); t0 = time.time()
    from tgb.linkproppred.dataset import LinkPropPredDataset
    from tgb.linkproppred.evaluate import Evaluator
    ds = LinkPropPredDataset(name='tgbl-wiki', root=a.root, preprocess=True); ev = Evaluator(name='tgbl-wiki')
    d = ds.full_data; ds.load_val_ns()
    if a.score_test:
        ds.load_test_ns()
    negset = set()
    for v in ds.negative_sampler.eval_set['val'].values():
        negset.update(int(x) for x in v)
    src_all, dst_all, ts_all = d['sources'], d['destinations'], d['timestamps']
    nodes = np.unique(np.concatenate([src_all, dst_all, np.fromiter(negset, np.int64)]))
    dests = np.unique(np.concatenate([dst_all, np.fromiter(negset, np.int64)]))
    sidx = np.full(nodes.max() + 1, -1, np.int64); sidx[nodes] = np.arange(len(nodes))
    didx = np.full(nodes.max() + 1, -1, np.int64); didx[dests] = np.arange(len(dests))
    S, Dn, T = sidx[src_all], didx[dst_all], ts_all.astype(float)
    tr = np.flatnonzero(ds.train_mask); va = np.flatnonzero(ds.val_mask); te = np.flatnonzero(ds.test_mask)
    if a.max_train_events:
        tr = tr[-a.max_train_events:]
    if a.max_eval_events:
        va = va[:a.max_eval_events]
    nf = Memory(1, 1).features(np.zeros(1, np.int64), np.zeros(1)).shape[-1]
    model = Readout(nf, a.hidden); opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    print(f'nodes {len(nodes)} dests {len(dests)} features {nf} params {sum(p.numel() for p in model.parameters())} '
          f'train {len(tr)} val {len(va)}', flush=True)

    def replay_train(train):
        mem = Memory(len(nodes), len(dests)); model.train(train); tot, n = 0.0, 0
        pre = np.flatnonzero(ds.train_mask)[:np.flatnonzero(ds.train_mask).searchsorted(tr[0])]
        if len(pre):
            mem.add(S[pre], Dn[pre], T[pre])
        for b in range(0, len(tr), a.bs):
            ix = tr[b:b + a.bs]
            if train:
                x = torch.from_numpy(mem.features(S[ix], T[ix])); logit = model(x)
                loss = F.cross_entropy(logit, torch.from_numpy(Dn[ix]))
                opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
                tot += float(loss) * len(ix); n += len(ix)
            mem.add(S[ix], Dn[ix], T[ix])
        return mem, tot / max(n, 1)

    @torch.no_grad()
    def evaluate(mem, idx, split, check=50):
        model.eval(); out = []; kinds = []
        for b in range(0, len(idx), a.bs):
            ix = idx[b:b + a.bs]
            logit = model(torch.from_numpy(mem.features(S[ix], T[ix]))).numpy()
            negs = ds.negative_sampler.query_batch(src_all[ix], dst_all[ix], ts_all[ix], split_mode=split)
            for q in range(len(ix)):
                v = logit[q, didx[np.concatenate([[dst_all[ix[q]]], np.asarray(negs[q])])]]; r = rr(v); out.append(r)
                su, cu = int(S[ix[q]]), int(Dn[ix[q]])                                 # error analysis by query type
                kinds.append('repeat_pair' if (su, cu) in mem.pair else 'new_pair_known_source' if mem.rows[su] else 'new_source')
                if len(out) <= check:
                    o = ev.eval({'y_pred_pos': v[:1], 'y_pred_neg': v[None, 1:], 'eval_metric': ['mrr']})['mrr']
                    assert abs(float(o) - r) < 1e-6
            mem.add(S[ix], Dn[ix], T[ix])
        out, kinds = np.array(out), np.array(kinds)
        evaluate.by_type = {k: dict(share=float((kinds == k).mean()), mrr=float(out[kinds == k].mean()))
                            for k in ('repeat_pair', 'new_pair_known_source', 'new_source') if (kinds == k).any()}
        return float(out.mean())

    hist, best, state = [], -1.0, None
    for ep in range(a.epochs):
        te0 = time.time(); mem, loss = replay_train(True); val = evaluate(mem, va, 'val')
        hist.append(dict(epoch=ep, train_ce=loss, val_mrr=val, val_by_type=evaluate.by_type, epoch_s=time.time() - te0)); print(json.dumps(hist[-1]), flush=True)
        if val > best:
            best, state = val, {k: v.clone() for k, v in model.state_dict().items()}
        if time.time() - t0 > a.max_wall_s:
            break
    model.load_state_dict(state)
    out = dict(status='completed', battle='B5', tag=a.tag, model='race_link v2 (v1 + ordinal recency, pair share, last-destination flag, '
               'decayed page transitions)', args=vars(a), features=nf,
               parameters=sum(p.numel() for p in model.parameters()), history=hist, best_val_mrr=best,
               test_mrr='not scored (development run)', wall_s=time.time() - t0, source_sha256=__import__('hashlib').sha256(
                   Path(__file__).read_bytes()).hexdigest())
    if a.score_test:
        mem, _ = replay_train(False); evaluate(mem, va, 'val'); out['test_mrr'] = evaluate(mem, te, 'test')
    torch.save(state, ROOT / f'experiments/results/tgb/{a.tag}.pt'); out['checkpoint'] = f'experiments/results/tgb/{a.tag}.pt'
    (ROOT / f'experiments/results/tgb/{a.tag}.json').write_text(json.dumps(out, indent=1) + '\n')
    print('RESULT', json.dumps({k: out[k] for k in ('best_val_mrr', 'test_mrr', 'parameters', 'wall_s')}))


if __name__ == '__main__':
    main()
