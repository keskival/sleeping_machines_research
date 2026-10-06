#!/usr/bin/env python3
"""B1: race of delayed clocks over persistent temporal memory (marked temporal point process).

After each event, M latent clocks start: exponential clocks (positive hazard at zero elapsed time) and log-normal clocks
(a learned delay with a learned dispersion). The first clock to fire produces the next event; each clock carries its own
mark distribution. Hence
    total intensity  λ(τ)   = Σ_m h_m(τ),          survival S(τ) = Π_m S_m(τ)  (exact compensator, no Monte Carlo)
    marked intensity λ_k(τ) = Σ_m h_m(τ) p_m(k)     (marks depend on elapsed time through the race).
Clock parameters come from persistent state: stacked complex-diagonal memories that decay and rotate with real elapsed
time, written by small messages mixing event content with memory, and an addressed mark memory (one slot per mark,
written only when that mark occurs, decaying at learned rates, read by each mark's own score).

Protocol (EasyTPP / S2P2 fork): events 2..N of every sequence are scored with log λ_{k_i}(t_i) − ∫_{t_{i-1}}^{t_i} λ;
per-event LL = total over the split / number of scored events. See experiments/B1_EASYTPP.md.
"""
import argparse
import hashlib
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
LOG2PI = math.log(2 * math.pi)


def load_split(ds, split):
    raw = json.load(open(ROOT / f'data/easytpp/{ds}/{split}.json'))
    return [(np.asarray(x['time_since_start'], np.float64), np.asarray(x['type_event'], np.int64)) for x in raw]


def batches(seqs, batch_size, shuffle, rng):
    order = sorted(range(len(seqs)), key=lambda i: len(seqs[i][1]))
    groups = [order[i:i + batch_size] for i in range(0, len(order), batch_size)]
    if shuffle:
        rng.shuffle(groups)
    for g in groups:
        n = max(len(seqs[i][1]) for i in g)
        t = np.zeros((len(g), n)); m = np.zeros((len(g), n), np.int64); mask = np.zeros((len(g), n), bool)
        for r, i in enumerate(g):
            L = len(seqs[i][1]); t[r, :L] = seqs[i][0]; m[r, :L] = seqs[i][1]; mask[r, :L] = True
            t[r, L:] = seqs[i][0][-1]
        yield torch.from_numpy(t), torch.from_numpy(m), torch.from_numpy(mask)


class TemporalMemoryLayer(nn.Module):
    """Complex-diagonal persistent memory: decays and rotates with elapsed time, then absorbs the event's message."""

    def __init__(self, d, n_modes, dropout):
        super().__init__()
        self.n = n_modes
        rates = torch.logspace(-2, 1, n_modes)                    # decay rates in units of the median gap
        self.log_rate = nn.Parameter(rates.log())
        self.freq = nn.Parameter(torch.randn(n_modes) * 0.5)
        self.write = nn.Linear(d, 2 * n_modes)
        self.gate = nn.Linear(d, n_modes)
        self.read = nn.Linear(2 * n_modes, d)
        self.mlp = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, d))
        self.norm1, self.norm2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.drop = nn.Dropout(dropout)

    def forward(self, u, dt):
        """u: [B, L, d] event messages; dt: [B, L] scaled elapsed time since the previous event (0 for the first)."""
        B, L, _ = u.shape
        rate, freq = self.log_rate.exp(), self.freq
        w = self.write(u); g = torch.sigmoid(self.gate(u))
        wr, wi = w[..., :self.n] * g, w[..., self.n:] * g
        decay = torch.exp(-rate * dt.unsqueeze(-1)); ang = freq * dt.unsqueeze(-1)
        cr, ci = decay * torch.cos(ang), decay * torch.sin(ang)
        zr = u.new_zeros(B, self.n); zi = u.new_zeros(B, self.n); outs = []
        for i in range(L):
            zr, zi = cr[:, i] * zr - ci[:, i] * zi + wr[:, i], cr[:, i] * zi + ci[:, i] * zr + wi[:, i]
            outs.append(torch.cat([zr, zi], -1))
        z = torch.stack(outs, 1)
        h = self.norm1(u + self.drop(self.read(z)))
        return self.norm2(h + self.drop(self.mlp(h)))


class AddressedMarkMemory(nn.Module):
    """One slot per mark; a slot is written only when its mark occurs; all slots decay with elapsed time."""

    def __init__(self, d, K, dv):
        super().__init__()
        self.K, self.dv = K, dv
        self.log_rate = nn.Parameter(torch.logspace(-2, 1, dv).log())
        self.value = nn.Linear(d, dv)

    def forward(self, h, marks, dt):
        """Slot contents after event i (inclusive): [B, L, K, dv]."""
        B, L, _ = h.shape
        v = F.softplus(self.value(h))                              # non-negative excitation written by the event
        decay = torch.exp(-self.log_rate.exp() * dt.unsqueeze(-1))  # [B, L, dv]
        onehot = F.one_hot(marks, self.K).to(h.dtype)               # [B, L, K]
        s = h.new_zeros(B, self.K, self.dv); outs = []
        for i in range(L):
            s = s * decay[:, i].unsqueeze(1) + onehot[:, i].unsqueeze(-1) * v[:, i].unsqueeze(1)
            outs.append(s)
        return torch.stack(outs, 1)


class RaceTPP(nn.Module):
    def __init__(self, K, d, n_modes, layers, n_exp, n_lognormal, dv, dropout, scale, log_gap_quantiles, floor_cell=0.0,
                 n_window=0, window_edges=None):
        super().__init__()
        self.floor_cell = floor_cell
        self.K, self.scale, self.n_exp, self.n_ln, self.n_win = K, scale, n_exp, n_lognormal, n_window
        self.M = n_exp + n_lognormal + n_window
        self.embed = nn.Embedding(K, d)
        self.gap = nn.Linear(3, d)                     # log gap, zero-gap flag, log age (time since sequence start)
        self.layers = nn.ModuleList(TemporalMemoryLayer(d, n_modes, dropout) for _ in range(layers))
        self.marks = AddressedMarkMemory(d, K, dv)
        self.clock = nn.Linear(d + K * dv, n_exp + 3 * n_lognormal + self.M + 4 * n_window)
        # layout: rates | ln mu | ln sigma | ln fire | clock weights (M) | window (log a, log width, edge, fire)
        self.mark_ctx = nn.Linear(d, self.M * K)
        self.mark_slot = nn.Parameter(torch.zeros(self.M, dv))
        with torch.no_grad():
            b = self.clock.bias
            b[:n_exp] = 0.0
            b[n_exp:n_exp + n_lognormal] = torch.tensor(log_gap_quantiles, dtype=b.dtype) - math.log(scale)
            b[n_exp + n_lognormal:n_exp + 2 * n_lognormal] = -0.5
            b[n_exp + 2 * n_lognormal:n_exp + 3 * n_lognormal] = 0.0
            if n_window:
                o = n_exp + 3 * n_lognormal + self.M
                lo, hi = (torch.tensor(e, dtype=b.dtype) for e in window_edges)      # gap quantile windows
                b[o:o + n_window] = lo.log() - math.log(scale)
                b[o + n_window:o + 2 * n_window] = (hi - lo).log() - math.log(scale)
                b[o + 2 * n_window:o + 3 * n_window] = -2.0                           # edges ~ 6% of the width
                b[o + 3 * n_window:o + 4 * n_window] = 0.0

    def encode(self, t, m, mask):
        dt = torch.diff(t, dim=1, prepend=t[:, :1]).clamp_min(0) / self.scale
        age = (t - t[:, :1]) / self.scale
        x = self.embed(m) + self.gap(torch.stack([torch.log1p(dt), (dt > 0).to(t.dtype), torch.log1p(age)], -1))
        for layer in self.layers:
            x = layer(x, dt)
        slots = self.marks(x, m, dt)
        return x, slots

    def clocks(self, h, slots):
        """Clock parameters after event i, for the gap to event i+1. h: [B,L,d], slots: [B,L,K,dv]."""
        B, L, _ = h.shape
        raw = self.clock(torch.cat([h, slots.flatten(-2)], -1))
        ne, nl = self.n_exp, self.n_ln
        log_rate = raw[..., :ne] - math.log(self.scale)
        mu = raw[..., ne:ne + nl] + math.log(self.scale)
        # Resolution floor: a clock may not resolve time below the data's recording cell (its spread at its
        # delay e^mu is at least one cell), so densities cannot spike on recording-grid points.
        floor = (self.floor_cell * torch.exp(-mu)).clamp_min(0.005) if self.floor_cell > 0 else 0.005
        sigma = F.softplus(raw[..., ne + nl:ne + 2 * nl]) + floor
        fire = raw[..., ne + 2 * nl:ne + 3 * nl]                                  # logit of P(delayed clock ever fires)
        log_w = F.log_softmax(raw[..., ne + 3 * nl:ne + 3 * nl + self.M], -1)
        log_w = torch.cat([log_w[..., :ne], F.logsigmoid(fire)], -1)               # exp: hazard weight; ln: fire prob
        nw = self.n_win
        if nw:
            o = ne + 3 * nl + self.M
            log_a = raw[..., o:o + nw] + math.log(self.scale)
            width = torch.exp(raw[..., o + nw:o + 2 * nw] + math.log(self.scale))
            if self.floor_cell > 0:                                                  # windows no finer than a cell
                width = width + self.floor_cell
            edge = width * (0.005 + 0.495 * torch.sigmoid(raw[..., o + 2 * nw:o + 3 * nw]))
            if self.floor_cell > 0:
                edge = edge + self.floor_cell
            mu = torch.cat([mu, log_a, width.log()], -1)
            sigma = torch.cat([sigma, edge], -1)                                     # logistic edge scale s
            log_w = torch.cat([log_w, F.logsigmoid(raw[..., o + 3 * nw:o + 4 * nw])], -1)
        logits = self.mark_ctx(h).view(B, L, self.M, self.K) + torch.einsum('blkv,mv->blmk', slots, self.mark_slot)
        return log_rate, mu, sigma, log_w, F.log_softmax(logits, -1)

    def clock_terms(self, tau, log_rate, mu, sigma, log_w):
        """Per-clock log hazard and log survival at gap tau. tau: [...]; returns [..., M] each.

        Exponential clocks always fire (hazard weight w_m scales the rate). A delayed log-normal clock fires with
        probability π_m and otherwise stays silent: S_m = (1 − π_m) + π_m S0_m, h_m = π_m f0_m / S_m. The race of
        independent clocks keeps an exact survival; the always-firing exponential clocks keep it proper."""
        tau = tau.unsqueeze(-1)
        ne, nl, nw = log_rate.shape[-1], self.n_ln, self.n_win
        if nw:
            win_mu, mu = mu[..., nl:], mu[..., :nl]
            win_ab, sigma = sigma[..., nl:], sigma[..., :nl]
            win_pi, log_w = log_w[..., ne + nl:], log_w[..., :ne + nl]
        w = log_w[..., :ne].exp()
        ls_exp = -log_rate.exp() * tau * w
        lh_exp = (log_rate + log_w[..., :ne]).expand_as(ls_exp)
        log_pi = log_w[..., ne:]
        log_1mpi = torch.log1p(-log_pi.exp().clamp_max(1 - 1e-12))
        pos = tau > 0
        lt = torch.log(tau.clamp_min(1e-30))
        z = (lt - mu) / sigma
        log_s0 = torch.special.log_ndtr(-z)
        log_f0 = -lt - sigma.log() - 0.5 * LOG2PI - 0.5 * z ** 2
        log_s = torch.logaddexp(log_1mpi, log_pi + log_s0)
        lh_ln = torch.where(pos, log_pi + log_f0 - log_s, torch.full_like(z, -math.inf))
        ls_ln = torch.where(pos, log_s, torch.zeros_like(z))
        if not nw:
            return torch.cat([lh_exp, lh_ln], -1), torch.cat([ls_exp, ls_ln], -1)
        # Logistic-window delayed clocks: density ∝ σ((τ−a)/s) − σ((τ−b)/s) on τ > 0 (flat on [a, b] with learned edge
        # scale s), firing with probability π. With u = (τ−a)/s, v = (τ−b)/s and Z = s[softplus(b/s) − softplus(a/s)]:
        #   log f0 = logσ(u) + logσ(−v) + log(1 − e^{−(b−a)/s}) − log Z
        #   log S0 = log s + log(softplus(−v) − softplus(−u)) − log Z          (all differences in stable log form)
        a, width = win_mu[..., :nw].exp(), win_mu[..., nw:].exp()
        s = win_ab
        b = a + width
        u, v = (tau - a) / s, (tau - b) / s
        def log_softplus(x):
            return torch.where(x < -30, x, torch.log(F.softplus(x).clamp_min(1e-300)))
        def log_diff_softplus(x, y):                                                 # log(softplus(x) − softplus(y)), x > y
            lx, ly = log_softplus(x), log_softplus(y)
            return lx + torch.log1p(-torch.exp((ly - lx).clamp_max(-1e-15)))
        log_z = s.log() + log_diff_softplus(b / s, a / s)
        log_f0 = F.logsigmoid(u) + F.logsigmoid(-v) + torch.log1p(-torch.exp(-width / s).clamp_max(1 - 1e-15)) - log_z
        log_s0 = s.log() + log_diff_softplus(-v, -u) - log_z
        log_s0 = log_s0.clamp_max(0.0)
        log_1mpw = torch.log1p(-win_pi.exp().clamp_max(1 - 1e-12))
        log_sw = torch.logaddexp(log_1mpw, win_pi + log_s0)
        pos = tau > 0
        lh_w = torch.where(pos, win_pi + log_f0 - log_sw, torch.full_like(u, -math.inf))
        log_sw = torch.where(pos, log_sw, torch.zeros_like(log_sw))
        return torch.cat([lh_exp, lh_ln, lh_w], -1), torch.cat([ls_exp, ls_ln, log_sw], -1)

    def loglik(self, t, m, mask):
        h, slots = self.encode(t, m, mask)
        params = self.clocks(h[:, :-1], slots[:, :-1])
        tau = (t[:, 1:] - t[:, :-1]).clamp_min(0)
        log_h, log_s = self.clock_terms(tau, *params[:4])
        log_pk = params[4]                                                         # [B, L-1, M, K]
        log_lam_total = torch.logsumexp(log_h, -1)
        log_lam_k = torch.logsumexp(log_h + log_pk.gather(-1, m[:, 1:, None, None].expand(-1, -1, self.M, 1)).squeeze(-1), -1)
        surv = log_s.sum(-1)
        valid = mask[:, 1:]
        time_ll = (log_lam_total + surv) * valid
        mark_ll = (log_lam_k - log_lam_total) * valid
        return time_ll.sum(), mark_ll.sum(), valid.sum(), (h, slots, params, tau, valid)

    @torch.no_grad()
    def predict(self, cache, m, grid=256):
        """Expected next gap (∫ S) and mark argmax given the true gap."""
        h, slots, params, tau, valid = cache
        log_rate, mu, sigma, log_w, log_pk = params
        u = torch.linspace(-18, 8, grid, dtype=tau.dtype)
        taus = self.scale * torch.exp(u)                                           # log-spaced quadrature in τ
        shape = tau.shape
        tq = taus.view(1, 1, grid).expand(*shape, grid)
        lh, ls = self.clock_terms(tq, *(p.unsqueeze(-2) for p in (log_rate, mu, sigma, log_w)))
        S = ls.sum(-1).exp()
        expected = torch.trapz(S * tq, u, dim=-1)
        log_h, _ = self.clock_terms(tau, log_rate, mu, sigma, log_w)
        log_lam_k = torch.logsumexp(log_h.unsqueeze(-1) + log_pk, -2)
        pred_mark = log_lam_k.argmax(-1)
        return expected, pred_mark


def evaluate(model, seqs, batch_size, rng, predict=True):
    model.eval()
    tot = dict(time=0.0, mark=0.0, n=0, se=0.0, correct=0)
    with torch.no_grad():
        for t, m, mask in batches(seqs, batch_size, False, rng):
            tl, ml, n, cache = model.loglik(t, m, mask)
            tot['time'] += tl.item(); tot['mark'] += ml.item(); tot['n'] += int(n)
            if predict:
                expected, pred = model.predict(cache, m)
                v = cache[4]
                tot['se'] += (((expected - cache[3]) ** 2) * v).sum().item()
                tot['correct'] += ((pred == m[:, 1:]) & v).sum().item()
    n = tot['n']
    out = dict(ll=(tot['time'] + tot['mark']) / n, time_ll=tot['time'] / n, mark_ll=tot['mark'] / n, events=n)
    if predict:
        out.update(rmse=math.sqrt(tot['se'] / n), acc=tot['correct'] / n)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True)
    ap.add_argument('--tag', required=True)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--d', type=int, default=32)
    ap.add_argument('--modes', type=int, default=16)
    ap.add_argument('--layers', type=int, default=2)
    ap.add_argument('--n-exp', type=int, default=2)
    ap.add_argument('--n-lognormal', type=int, default=4)
    ap.add_argument('--dv', type=int, default=4)
    ap.add_argument('--dropout', type=float, default=0.1)
    ap.add_argument('--lr', type=float, default=3e-3)
    ap.add_argument('--wd', type=float, default=0.0, help='decoupled weight decay on shared parameters')
    ap.add_argument('--wd-private-mult', type=float, default=1.0,
                    help='decay multiplier for per-mark (private) parameters: embeddings, mark readout (THEORY §427.3)')
    ap.add_argument('--batch', type=int, default=32)
    ap.add_argument('--epochs', type=int, default=200)
    ap.add_argument('--patience', type=int, default=20)
    ap.add_argument('--max-wall-s', type=float, default=6 * 3600)
    ap.add_argument('--score-test', action='store_true', help='score TEST once with the best-DEV checkpoint')
    ap.add_argument('--threads', type=int, default=1)
    ap.add_argument('--n-window', type=int, default=0, help='logistic-window delayed clocks')
    ap.add_argument('--floor-cell', type=float, default=0.0,
                    help='recording resolution of event times; clocks may not resolve below it (0 = none)')
    a = ap.parse_args()
    torch.set_num_threads(a.threads); torch.set_default_dtype(torch.float64)
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    rng = random.Random(a.seed)
    train, dev = load_split(a.dataset, 'train'), load_split(a.dataset, 'dev')
    K = max(int(m.max()) for _, m in train + dev) + 1
    gaps = np.concatenate([np.diff(t) for t, _ in train]); pos = gaps[gaps > 0]
    scale = float(np.median(pos))
    qs = np.log(np.quantile(pos, np.linspace(0.1, 0.9, a.n_lognormal))).tolist()
    edges = None
    if a.n_window:
        qq = np.quantile(pos, np.linspace(0.02, 0.98, a.n_window + 1))
        edges = (qq[:-1].tolist(), qq[1:].tolist())
    model = RaceTPP(K, a.d, a.modes, a.layers, a.n_exp, a.n_lognormal, a.dv, a.dropout, scale, qs, a.floor_cell,
                    a.n_window, edges)
    params = sum(p.numel() for p in model.parameters())
    private_names = ('embed.weight', 'mark_ctx.weight', 'mark_ctx.bias')
    private = [p for n, p in model.named_parameters() if n in private_names]
    shared = [p for n, p in model.named_parameters() if n not in private_names]
    opt = torch.optim.AdamW([dict(params=shared, weight_decay=a.wd),
                             dict(params=private, weight_decay=a.wd * a.wd_private_mult)], lr=a.lr)
    out_dir = ROOT / 'experiments/results/tpp'; out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = out_dir / f'{a.tag}.pt'
    best, best_epoch, history, start = -math.inf, -1, [], time.time()
    for epoch in range(a.epochs):
        model.train(); e0 = time.time(); tr_ll = tr_n = 0
        for t, m, mask in batches(train, a.batch, True, rng):
            tl, ml, n, _ = model.loglik(t, m, mask)
            loss = -(tl + ml) / n
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            tr_ll += (tl + ml).item(); tr_n += int(n)
        dv = evaluate(model, dev, 64, rng, predict=False)
        history.append(dict(epoch=epoch, train_ll=tr_ll / tr_n, dev_ll=dv['ll'], dev_time_ll=dv['time_ll'],
                            dev_mark_ll=dv['mark_ll'], epoch_s=time.time() - e0))
        print(json.dumps(history[-1]), flush=True)
        if dv['ll'] > best:
            best, best_epoch = dv['ll'], epoch
            torch.save(model.state_dict(), ckpt)
        if (a.patience > 0 and epoch - best_epoch >= a.patience) or time.time() - start > a.max_wall_s:
            break
    final_dev = evaluate(model, dev, 64, rng, predict=False)
    model.load_state_dict(torch.load(ckpt))
    result = dict(status='completed', battle='B1', tag=a.tag, dataset=a.dataset, args=vars(a), K=K, scale=scale,
                  parameters=params, best_epoch=best_epoch, epochs_run=len(history), wall_s=time.time() - start,
                  dev=evaluate(model, dev, 64, rng), final_dev=final_dev, history=history,
                  checkpoint=str(ckpt.relative_to(ROOT)))
    if a.score_test:
        result['test'] = evaluate(model, load_split(a.dataset, 'test'), 64, rng)
    src = Path(__file__).resolve()
    result['source_sha256'] = {str(src.relative_to(ROOT)): hashlib.sha256(src.read_bytes()).hexdigest()}
    (out_dir / f'{a.tag}.json').write_text(json.dumps(result, indent=1) + '\n')
    print('DEV', json.dumps(result['dev']), flush=True)
    if a.score_test:
        print('TEST', json.dumps(result['test']), flush=True)


if __name__ == '__main__':
    main()
