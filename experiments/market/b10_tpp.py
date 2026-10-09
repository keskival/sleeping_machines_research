#!/usr/bin/env python3
"""B10: Poisson, multivariate Hawkes and race-of-clocks models of the BTCUSDT aggTrades stream under the frozen protocol
(experiments/market/B10_MARKET.md).

Windows: each day's events cut into consecutive windows of 1,024; times in seconds from the window's first event (keeps
the 1 µs recording cell exact in float64). Scored events: positions 128..1023 of every window (context 0..127).
Training: every 20th window of the training days (all positions); validation: every 10th window of the validation days
(scored positions); test (--score-test): every window of the test days (scored positions), once.
Per event: log p = log[S(g) − S(g + c)] + log π_k(g + c/2), c = 1 µs: the interval likelihood of the recording cell
holding the gap, and the mark law at the cell's midpoint, identical for every model.
"""
import argparse
import datetime as dt
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'experiments/tpp'))
from race_tpp_v5 import RaceTPP  # noqa: E402

NPZ = ROOT / 'data/binance/npz'
CELL = 1e-6; W = 1024; CTX = 128; K = 6
TRAIN = [dt.date(2026, 8, 1) + dt.timedelta(d) for d in range(24)]
VAL = [dt.date(2026, 8, 25) + dt.timedelta(d) for d in range(7)]
TEST = [dt.date(2026, 9, 1) + dt.timedelta(d) for d in range(7)]


def windows(day, every):
    z = np.load(NPZ / f'{day}.npz'); t, m = z['t'], z['mark'].astype(np.int64)
    n = len(t) // W; out = []
    for w in range(0, n, every):
        tt = t[w * W:(w + 1) * W]; out.append(((tt - tt[0]) * 1e-6, m[w * W:(w + 1) * W]))
    return out


def log_interval(log_s0, log_s1):
    """log(S0 − S1) for log S0 ≥ log S1."""
    d = (log_s1 - log_s0).clamp_max(-1e-12)
    return log_s0 + torch.log(-torch.expm1(d))


class Poisson(nn.Module):
    def __init__(self, rates):
        super().__init__(); self.log_lam = nn.Parameter(torch.log(torch.tensor(rates)))

    def terms(self, t, m):
        g = t[:, 1:] - t[:, :-1]; lam = self.log_lam.exp(); tot = lam.sum()
        log_s0, log_s1 = -tot * g, -tot * (g + CELL)
        log_pi = (self.log_lam - torch.logsumexp(self.log_lam, 0))[m[:, 1:]]
        return log_interval(log_s0, log_s1) + log_pi


class Hawkes(nn.Module):
    """λ_k(t) = μ_k + Σ_r Σ_j α_{r,k,m_j} β_r e^{−β_r (t − t_j)}; exact compensator."""

    def __init__(self, rates, betas=(100.0, 1.0)):
        super().__init__()
        self.log_mu = nn.Parameter(torch.log(torch.tensor(rates) * 0.5))
        self.log_alpha = nn.Parameter(torch.full((len(betas), K, K), math.log(0.05)))
        self.log_beta = nn.Parameter(torch.log(torch.tensor(betas)))

    def terms(self, t, m):
        B, L = t.shape; mu = self.log_mu.exp(); al = self.log_alpha.exp(); be = self.log_beta.exp()   # [R,K,K], [R]
        A = torch.zeros(B, len(be), K, dtype=t.dtype)                         # Σ_j e^{−β_r (t − t_j)} per source mark
        oh = torch.nn.functional.one_hot(m, K).to(t.dtype)
        out = []
        for i in range(1, L):
            if i == 1:
                A = oh[:, 0].unsqueeze(1).expand(B, len(be), K).clone()           # state right after event 0
            else:
                A = A * torch.exp(-be.view(1, -1, 1) * (t[:, i - 1] - t[:, i - 2]).view(B, 1, 1)) + oh[:, i - 1].unsqueeze(1)
            g = (t[:, i] - t[:, i - 1]).unsqueeze(1)                           # [B,1]
            exc = torch.einsum('rkm,brm->brk', al, A)                          # Σ_m α A  [B,R,K]
            def comp(x):
                return mu.sum() * x + (exc.sum(-1) * (1 - torch.exp(-be * x))).sum(-1, keepdim=True)
            log_s0, log_s1 = -comp(g), -comp(g + CELL)
            lam = mu + (exc * be.view(1, -1, 1) * torch.exp(-be.view(1, -1, 1) * (g + CELL / 2).unsqueeze(-1))).sum(1)
            log_pi = torch.log(lam.gather(1, m[:, i:i + 1])) - torch.log(lam.sum(1, keepdim=True))
            out.append((log_interval(log_s0, log_s1) + log_pi).squeeze(1))
        return torch.stack(out, 1)


class Race(nn.Module):
    def __init__(self, pos_gaps, d, n_ln):
        super().__init__()
        scale = float(np.median(pos_gaps)); qs = np.log(np.quantile(pos_gaps, np.linspace(0.05, 0.95, n_ln))).tolist()
        self.m = RaceTPP(K, d, 16, 2, 1, n_ln, 4, 0.1, scale, qs, CELL)

    def terms(self, t, m):
        mask = torch.ones_like(m, dtype=torch.bool)
        h, slots = self.m.encode(t, m, mask); p = self.m.clocks(h[:, :-1], slots[:, :-1])
        g = (t[:, 1:] - t[:, :-1]).clamp_min(0)
        _, ls0 = self.m.clock_terms(g, *p[:4]); _, ls1 = self.m.clock_terms(g + CELL, *p[:4])
        lh, _ = self.m.clock_terms(g + CELL / 2, *p[:4])
        log_pk = p[4].gather(-1, m[:, 1:, None, None].expand(-1, -1, self.m.M, 1)).squeeze(-1)
        log_pi = torch.logsumexp(lh + log_pk, -1) - torch.logsumexp(lh, -1)
        return log_interval(ls0.sum(-1), ls1.sum(-1)) + log_pi


def batches(ws, bs, rng=None):
    idx = np.arange(len(ws)) if rng is None else rng.permutation(len(ws))
    for b in range(0, len(idx), bs):
        sel = [ws[i] for i in idx[b:b + bs]]
        yield torch.tensor(np.stack([s[0] for s in sel])), torch.tensor(np.stack([s[1] for s in sel]))


@torch.no_grad()
def score(model, ws, bs):
    model.eval(); tot, n = 0.0, 0
    for t, m in batches(ws, bs):
        lp = model.terms(t, m)[:, CTX - 1:]                                     # gaps into positions 128..1023
        tot += float(lp.sum()); n += lp.numel()
    return tot / n


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--tag', required=True)
    ap.add_argument('--model', choices=('poisson', 'hawkes', 'race'), required=True)
    ap.add_argument('--seed', type=int, default=0); ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--patience', type=int, default=5); ap.add_argument('--lr', type=float, default=3e-3)
    ap.add_argument('--bs', type=int, default=16); ap.add_argument('--d', type=int, default=32)
    ap.add_argument('--n-ln', type=int, default=8); ap.add_argument('--train-every', type=int, default=20)
    ap.add_argument('--max-wall-s', type=float, default=5 * 3600); ap.add_argument('--score-test', action='store_true')
    a = ap.parse_args(); torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed)
    torch.set_num_threads(1); torch.set_default_dtype(torch.float64); t0 = time.time()
    tr = [w for d in TRAIN for w in windows(d, a.train_every)]; va = [w for d in VAL for w in windows(d, 10)]
    gaps = np.concatenate([np.diff(w[0]) for w in tr]); pos = gaps[gaps > 0]
    counts = np.bincount(np.concatenate([w[1] for w in tr]), minlength=K); rates = counts / sum(w[0][-1] for w in tr)
    model = dict(poisson=lambda: Poisson(rates.tolist()), hawkes=lambda: Hawkes(rates.tolist()),
                 race=lambda: Race(pos, a.d, a.n_ln))[a.model]()
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4 if a.model == 'race' else 0.0)
    print(f'train windows {len(tr)} val windows {len(va)} zero-gap frac {float((gaps == 0).mean()):.3f} '
          f'params {sum(p.numel() for p in model.parameters())}', flush=True)
    hist, best, state, bad = [], -math.inf, None, 0
    for ep in range(a.epochs):
        model.train(); tl, tn = 0.0, 0
        for t, m in batches(tr, a.bs, rng):
            lp = model.terms(t, m); loss = -lp.mean()
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            tl += float(lp.sum()); tn += lp.numel()
        v = score(model, va, 32); hist.append(dict(epoch=ep, train_ll=tl / tn, val_ll=v, elapsed_s=time.time() - t0))
        print(json.dumps(hist[-1]), flush=True)
        if v > best:
            best, state, bad = v, {k: x.clone() for k, x in model.state_dict().items()}, 0
        else:
            bad += 1
        if bad >= a.patience or time.time() - t0 > a.max_wall_s:
            break
    model.load_state_dict(state)
    out = dict(status='completed', battle='B10', tag=a.tag, model=a.model, args=vars(a), parameters=sum(p.numel() for p in model.parameters()),
               train_windows=len(tr), val_windows=len(va), best_val_ll_per_event=best, history=hist,
               test='not scored (development run)', source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    if a.score_test:
        per_day = {}
        for d in TEST:
            per_day[str(d)] = score(model, windows(d, 1), 32)
        out['test'] = dict(per_day=per_day, mean=float(np.mean(list(per_day.values()))))
    out['wall_s'] = time.time() - t0
    rd = ROOT / 'experiments/results/market'; rd.mkdir(parents=True, exist_ok=True)
    torch.save(state, rd / f'{a.tag}.pt'); (rd / f'{a.tag}.json').write_text(json.dumps(out, indent=1) + '\n')
    print('RESULT', json.dumps(dict(best_val=best, test=out['test'], wall_s=out['wall_s'])))


if __name__ == '__main__':
    main()
