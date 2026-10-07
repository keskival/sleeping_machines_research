#!/usr/bin/env python3
"""B3 round 3: the B1 race-of-delayed-clocks model (race_tpp_v16, all five EasyTPP datasets won) on FAS v2, optionally with
the R1 keyed predecessor-message read (experiments/R1_RECALL.md).

Why (B3_DEVELOPMENT_LOG.md, 7 Oct 22:10): the native AddressedEventHeads + race readout binds poorly (purity 0.30); its
faulty runs with high binding score .735 at N*, low binding .511. The B1 model is the family's strongest trained event
model; v16 holds every clock's hazard at its one-cell value below the recording cell and dequantizes the scored target
gap inside its cell, the exact remedy for FAS's 36% zero-millisecond gaps (THEORY §439; RACE_OF_DELAYED_CLOCKS §3).
`--keyed 1` adds the keyed mark memory whose written keys carry the predecessor's message (R1 gate 1: binding 21.6% →
97–99%). Retained mechanisms: race of delayed clocks with exact survival, logistic-window clocks, temporal memory,
addressed mark slots; added: separate keys with sparse writes and query-scored race marks.

Protocol (FAS_V2_CONFIRMATORY_PROTOCOL.md): clean-only training, at most 3 passes over the same 5,000 clean training
samples; selection by validation-clean NLL; --no-test is the only mode here (development). Scoring rules and prefixes
are native.py's: per position, type NLL = −(joint − time) and gap NLL = −(time log density + log(gap + 0.01 s)), i.e. the
time-only likelihood in log-gap units, so per-event NLL is comparable to the native drivers.
"""
import argparse
import hashlib
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
sys.path.insert(0, str(ROOT / 'experiments/fas')); sys.path.insert(0, str(ROOT / 'experiments/tpp'))
from native import LOG_EPS, PREFIXES, aurocs, load, prefix_scores  # noqa: E402
from race_tpp_v16 import RaceTPP, cluster_windows  # noqa: E402

OUT = ROOT / 'experiments/results/fas'


class KeyedRaceFAS(RaceTPP):
    """v16 race model plus the keyed predecessor-message mark memory (recall_tpp_v4, --prev-msg)."""

    def __init__(self, *args, dk=16, local=False, **kw):
        super().__init__(*args, **kw)
        d = self.embed.embedding_dim
        self.dk, self.local = dk, local
        self.key_w = nn.Linear(2 * d, dk)
        self.query_w = nn.Linear(d, dk)
        self.key_log_rate = nn.Parameter(torch.logspace(-3, 0, dk).log())

    def encode(self, t, m, mask):
        x, slots = super().encode(t, m, mask)
        xk = x.detach() if self.local else x
        dt = torch.diff(t, dim=1, prepend=t[:, :1]).clamp_min(0) / self.scale
        prev = torch.cat([torch.zeros_like(xk[:, :1]), self.embed(m[:, :-1])], 1)
        k = self.key_w(torch.cat([xk, prev.detach() if self.local else prev], -1))
        decay = torch.exp(-self.key_log_rate.exp() * dt.unsqueeze(-1))
        onehot = F.one_hot(m, self.K).to(x.dtype)
        B, L, _ = x.shape
        s = x.new_zeros(B, self.K, self.dk); keyed = []
        for i in range(L):
            s = s * decay[:, i].unsqueeze(1) + onehot[:, i].unsqueeze(-1) * k[:, i].unsqueeze(1)
            keyed.append(s)
        self._keyed = torch.stack(keyed, 1)
        self._query = self.query_w(xk)
        return x, slots

    def clocks(self, h, slots):
        out = list(super().clocks(h, slots))
        L = h.shape[1]
        bonus = torch.einsum('bld,blkd->blk', self._query[:, :L], self._keyed[:, :L]) / math.sqrt(self.dk)
        out[4] = F.log_softmax(out[4] + bonus.unsqueeze(2), -1)
        return tuple(out)


def to_tensors(runs):
    L = max(len(r[0]) for r in runs)
    t = np.zeros((len(runs), L)); m = np.zeros((len(runs), L), np.int64); mask = np.zeros((len(runs), L), bool)
    for i, (ids, ts) in enumerate(runs):
        n = len(ids); t[i, :n] = ts; m[i, :n] = ids; mask[i, :n] = True; t[i, n:] = ts[-1]; m[i, n:] = ids[-1]
    return torch.from_numpy(t), torch.from_numpy(m), torch.from_numpy(mask)


def per_position(model, runs, batch):
    """Per-run (type, gap) NLL arrays in native.py's convention (position k predicts event k + 1), and the mean NLL."""
    model.eval(); pts, pgs, tot, cnt = [], [], 0.0, 0
    with torch.no_grad():
        for i in range(0, len(runs), batch):
            chunk = runs[i:i + batch]
            t, m, mask = to_tensors(chunk)
            time_lp, joint_lp, cache = model.event_terms(t, m, mask)
            tau = cache[3]; valid = mask[:, 1:]
            pt = (-(joint_lp - time_lp)).numpy(); pg = (-(time_lp + torch.log(tau + LOG_EPS))).numpy()
            v = valid.numpy(); pt[~v] = 0.0; pg[~v] = 0.0
            tot += float((pt + pg)[v].sum()); cnt += int(v.sum())
            for r in range(len(chunk)):
                pts.append(pt[r]); pgs.append(pg[r])
    L = max(len(x) for x in pts)
    PT = np.zeros((len(pts), L)); PG = np.zeros((len(pts), L))
    for r in range(len(pts)):
        PT[r, :len(pts[r])] = pts[r]; PG[r, :len(pgs[r])] = pgs[r]
    return PT, PG, tot / max(cnt, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', required=True)
    ap.add_argument('--data', default='fas_v2_K2_drop0.02_delta0_20261005')
    ap.add_argument('--keyed', type=int, default=0, help='0: v16 race model; 1: + keyed predecessor read; 2: local credit')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--fit-runs', type=int, default=5000)
    ap.add_argument('--train-max-events', type=int, default=1024)
    ap.add_argument('--max-events', type=int, default=1100)
    ap.add_argument('--eval-runs', type=int, default=1000)
    ap.add_argument('--d', type=int, default=32)
    ap.add_argument('--modes', type=int, default=16)
    ap.add_argument('--layers', type=int, default=2)
    ap.add_argument('--n-exp', type=int, default=2)
    ap.add_argument('--n-lognormal', type=int, default=8)
    ap.add_argument('--n-window', type=int, default=4)
    ap.add_argument('--dv', type=int, default=4)
    ap.add_argument('--dk', type=int, default=16)
    ap.add_argument('--dropout', type=float, default=0.1)
    ap.add_argument('--lr', type=float, default=3e-3)
    ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--epochs', type=int, default=3, help='protocol cap: at most 3 passes')
    ap.add_argument('--cell-s', type=float, default=0.001, help='recording cell (1 ms)')
    ap.add_argument('--max-wall-s', type=float, default=10800)
    a = ap.parse_args()
    if a.epochs > 3 or a.fit_runs > 5000:
        raise ValueError('protocol cap: at most 3 passes over 5,000 clean training samples')
    out = OUT / f'{a.tag}.json'
    if out.exists():
        raise ValueError('unique unused tag required')
    torch.set_num_threads(1); torch.set_default_dtype(torch.float64); torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed); start = time.time()
    d = ROOT / 'experiments/data/fas' / a.data
    train, _ = load(d / 'train_clean.npz', a.train_max_events); train = train[:a.fit_runs]
    val_c, _ = load(d / 'val_clean.npz', a.max_events); val_f, val_k = load(d / 'val_faulty.npz', a.max_events)
    val_c, val_f, val_k = val_c[:a.eval_runs], val_f[:a.eval_runs], val_k[:a.eval_runs]
    K = 46
    gaps = np.concatenate([np.diff(ts) for _, ts in train[:500]]); pos = gaps[gaps > 0]
    scale = float(np.median(pos))
    qs = np.log(np.quantile(pos, np.linspace(0.1, 0.9, a.n_lognormal))).tolist()
    edges = cluster_windows(pos, a.n_window, a.seed) if a.n_window else None
    margs = (K, a.d, a.modes, a.layers, a.n_exp, a.n_lognormal, a.dv, a.dropout, scale, qs, a.cell_s, a.n_window, edges, 0)
    model = KeyedRaceFAS(*margs, dk=a.dk, local=a.keyed == 2) if a.keyed else RaceTPP(*margs)
    params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.0)
    steps = a.epochs * math.ceil(len(train) / a.batch)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    ckpt = OUT / f'{a.tag}.pt'
    best, best_epoch, history = math.inf, -1, []
    for epoch in range(a.epochs):
        model.train(); e0 = time.time(); tl = tn = 0.0
        order = rng.permutation(len(train))
        for bi, i in enumerate(range(0, len(train), a.batch)):
            t, m, mask = to_tensors([train[j] for j in order[i:i + a.batch]])
            ll_t, ll_m, n, _ = model.loglik(t, m, mask, a.cell_s)
            loss = -(ll_t + ll_m) / n
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step()
            tl += float(loss) * int(n); tn += int(n)
            if bi % 50 == 0:
                print(json.dumps(dict(epoch=epoch, step=bi, train_nll=tl / tn, elapsed_s=round(time.time() - start))), flush=True)
            if time.time() - start > a.max_wall_s:
                break
        _, _, vnll = per_position(model, val_c, 32)
        history.append(dict(epoch=epoch, train_nll=tl / tn, val_clean_nll=vnll, epoch_s=round(time.time() - e0)))
        print(json.dumps(history[-1]), flush=True)
        if vnll < best:
            best, best_epoch = vnll, epoch; torch.save(model.state_dict(), ckpt)
        if time.time() - start > a.max_wall_s:
            break
    model.load_state_dict(torch.load(ckpt))
    PTc, PGc, nll_c = per_position(model, val_c, 32)
    PTf, PGf, _ = per_position(model, val_f, 32)
    sc_c = prefix_scores(PTc, PGc, [len(r[0]) for r in val_c])
    sc_f = prefix_scores(PTf, PGf, [len(r[0]) for r in val_f])
    res = aurocs(sc_c, sc_f, np.asarray(val_k))
    np.savez_compressed(OUT / f'{a.tag}_scores.npz', **{f'clean_{r}': v for r, v in sc_c.items()},
                        **{f'faulty_{r}': v for r, v in sc_f.items()}, kinds=np.asarray(val_k))
    srcs = [Path(__file__).resolve(), ROOT / 'experiments/tpp/race_tpp_v16.py', ROOT / 'experiments/fas/native.py']
    result = dict(status='completed', battle='B3', round=3, tag=a.tag, args=vars(a), data=a.data, parameters=params,
                  scale_s=scale, best_epoch=best_epoch, epochs_run=len(history), wall_s=time.time() - start,
                  history=history, val_clean_nll=nll_c, val_auroc=res, prefixes=PREFIXES,
                  test_auroc='not scored (development run)', checkpoint=str(ckpt.relative_to(ROOT)),
                  decision='does the B1 race model (and the keyed predecessor read) bind FAS v2 lines better than the '
                           'native C1/C6 (best .664 total) and pass order3 (.685) on validation',
                  source_sha256={str(s.relative_to(ROOT)): hashlib.sha256(s.read_bytes()).hexdigest() for s in srcs})
    out.write_text(json.dumps(result, indent=1, default=float) + '\n')
    n_star = 1024
    print('RESULT', json.dumps(dict(val_clean_nll=nll_c, total=res.get(n_star), type=res['rules']['type'].get(n_star)),
                               default=float), flush=True)


if __name__ == '__main__':
    main()
