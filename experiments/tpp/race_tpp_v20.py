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
v19 (7 Oct 2026): one driver with every mechanism of the five EasyTPP wins: defective delayed clocks, logistic windows
anchored to their gap component with component-aware initialization (v17/v18), the state clock, and the resolution
principle for every clock with target-only dequantization (v16). The recording cell is data metadata, not a tuning knob.

v20 (9 Oct 2026, B4 MOOC diagnosis): optional persistent mark pathways, off by default (then identical to v19).
  --mark-mem N   N temporal-memory layers over the event inputs that decay per EVENT, not per elapsed time, read only by
                 the mark distributions (silence does not erase them; time gradients do not reach them: key/value
                 separation of the two tasks);
  --mark-stats   per-sequence running count and frequency of every mark (statistic-valued, non-decaying).
Both enter every mark distribution (clocks and state clock) through a zero-initialized additive term.
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
                 n_window=0, window_edges=None, state_modes=0, quad_nodes=24, mark_mem=0, mark_stats=False):
        super().__init__()
        self.mark_mem_n, self.mark_stats = mark_mem, mark_stats
        if mark_mem:
            self.mark_layers = nn.ModuleList(TemporalMemoryLayer(d, n_modes, dropout) for _ in range(mark_mem))
        if mark_mem or mark_stats:
            self.mark_extra = nn.Linear((d if mark_mem else 0) + (2 * K if mark_stats else 0), K)
            nn.init.zeros_(self.mark_extra.weight); nn.init.zeros_(self.mark_extra.bias)
        self._extra = None
        self.floor_cell = floor_cell
        self.K, self.scale, self.n_exp, self.n_ln, self.n_win = K, scale, n_exp, n_lognormal, n_window
        self.M = n_exp + n_lognormal + n_window
        self.ns = state_modes
        if state_modes:
            # State clock: a persistent complex memory that keeps decaying and rotating through the silent interval;
            # its evolved state drives a hazard and a mark distribution that change continuously with elapsed time.
            self.s_write = nn.Linear(d, 2 * state_modes)
            self.s_log_rate = nn.Parameter(torch.logspace(-2, 1, state_modes).log())
            self.s_freq = nn.Parameter(torch.randn(state_modes) * 0.5)
            self.s_out = nn.Linear(2 * state_modes, 1, bias=False)
            self.s_bias = nn.Linear(d, 1)
            self.s_mark_z = nn.Linear(2 * state_modes, K, bias=False)
            self.s_mark_h = nn.Linear(d, K)
            with torch.no_grad():
                self.s_bias.bias.fill_(-3.0)
            x, w = np.polynomial.legendre.leggauss(quad_nodes)
            self.register_buffer('gl_x', torch.tensor((x + 1) / 2))                 # nodes on [0, 1]
            self.register_buffer('gl_w', torch.tensor(w / 2))
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
                # windows are anchored to the gap component they start on: location and width adapt within bounds
                self.register_buffer('win_log_a0', lo.log())
                self.register_buffer('win_log_w0', (hi - lo).log())
                b[o:o + n_window] = 0.0
                b[o + n_window:o + 2 * n_window] = 0.0
                b[o + 2 * n_window:o + 3 * n_window] = -2.0                           # edges ~ 6% of the width
                b[o + 3 * n_window:o + 4 * n_window] = 0.0

    def encode(self, t, m, mask):
        dt = torch.diff(t, dim=1, prepend=t[:, :1]).clamp_min(0) / self.scale
        age = (t - t[:, :1]) / self.scale
        x = self.embed(m) + self.gap(torch.stack([torch.log1p(dt), (dt > 0).to(t.dtype), torch.log1p(age)], -1))
        x0 = x
        for layer in self.layers:
            x = layer(x, dt)
        slots = self.marks(x, m, dt)
        self._extra = None
        if self.mark_mem_n or self.mark_stats:
            feats = []
            if self.mark_mem_n:
                y = x0
                one = torch.ones_like(dt)                                  # per-event decay: silence does not erase
                for layer in self.mark_layers:
                    y = layer(y, one)
                feats.append(y)
            if self.mark_stats:
                cnt = torch.cumsum(F.one_hot(m, self.K).to(t.dtype) * mask.unsqueeze(-1).to(t.dtype), 1)
                n = torch.arange(1, m.shape[1] + 1, dtype=t.dtype).view(1, -1, 1)
                feats.append(torch.cat([torch.log1p(cnt), cnt / n], -1))
            self._extra = self.mark_extra(torch.cat(feats, -1))            # [B, L, K]
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
            log_a = self.win_log_a0 + 0.3 * torch.tanh(raw[..., o:o + nw])          # start within ×e^±0.3
            width = torch.exp(self.win_log_w0 + 0.5 * torch.tanh(raw[..., o + nw:o + 2 * nw]))   # width within ×e^±0.5
            if self.floor_cell > 0:                                                  # windows no finer than a cell
                width = width + self.floor_cell
            edge = width * (0.005 + 0.495 * torch.sigmoid(raw[..., o + 2 * nw:o + 3 * nw]))
            if self.floor_cell > 0:
                edge = edge + self.floor_cell
            mu = torch.cat([mu, log_a, width.log()], -1)
            sigma = torch.cat([sigma, edge], -1)                                     # logistic edge scale s
            log_w = torch.cat([log_w, F.logsigmoid(raw[..., o + 3 * nw:o + 4 * nw])], -1)
        logits = self.mark_ctx(h).view(B, L, self.M, self.K) + torch.einsum('blkv,mv->blmk', slots, self.mark_slot)
        if self._extra is not None:
            logits = logits + self._extra[:, :L].unsqueeze(-2)
        return log_rate, mu, sigma, log_w, F.log_softmax(logits, -1)

    def clock_terms(self, tau, log_rate, mu, sigma, log_w):
        """Resolution principle for every clock: with a recording cell c, each clock's hazard is held at its value at
        c on [0, c); log S(τ) = −h(c)·min(τ, c) + [log S0(max(τ, c)) − log S0(c)]. Without a cell: unchanged."""
        c = self.floor_cell
        if c <= 0:
            return self._clock_terms_raw(tau, log_rate, mu, sigma, log_w)
        tc = torch.full_like(tau, c)
        lh_c, ls_c = self._clock_terms_raw(tc, log_rate, mu, sigma, log_w)
        lh_t, ls_t = self._clock_terms_raw(tau.clamp_min(c), log_rate, mu, sigma, log_w)
        below = (tau < c).unsqueeze(-1)
        log_h = torch.where(below, lh_c, lh_t)
        head = -lh_c.exp() * tau.clamp_max(c).unsqueeze(-1)
        log_s = head + torch.where(below, torch.zeros_like(ls_t), ls_t - ls_c)
        return log_h, log_s

    def _clock_terms_raw(self, tau, log_rate, mu, sigma, log_w):
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

    def state_dynamics(self):
        """Decay rates and rotation frequencies (scaled time). With a recording cell, the state clock's hazard must be
        smooth at the cell scale: frequencies are capped at one period per 8 cells and decay rates at one per cell."""
        rate, freq = self.s_log_rate.exp(), self.s_freq
        if self.floor_cell > 0:
            cell = self.floor_cell / self.scale
            w_max = 2 * math.pi / (8 * cell); r_max = 1 / cell
            freq = w_max * torch.tanh(freq / w_max)
            rate = r_max * torch.tanh(rate / r_max)
        return rate, freq

    def state_seq(self, h, dt):
        """Complex state after each event: z_i = exp((−r + iω) dt_i) z_{i−1} + W h_i."""
        B, L, _ = h.shape
        w = self.s_write(h); wr, wi = w[..., :self.ns], w[..., self.ns:]
        rate, freq = self.state_dynamics()
        dec = torch.exp(-rate * dt.unsqueeze(-1)); ang = freq * dt.unsqueeze(-1)
        cr, ci = dec * torch.cos(ang), dec * torch.sin(ang)
        zr = h.new_zeros(B, self.ns); zi = h.new_zeros(B, self.ns); outr, outi = [], []
        for i in range(L):
            zr, zi = cr[:, i] * zr - ci[:, i] * zi + wr[:, i], cr[:, i] * zi + ci[:, i] * zr + wi[:, i]
            outr.append(zr); outi.append(zi)
        return torch.stack(outr, 1), torch.stack(outi, 1)

    def state_eval(self, h, zr, zi, tau, marks=True):
        """Log hazard (and log mark distribution) of the state clock at elapsed time tau [..., Q] after the event."""
        x = tau / self.scale
        rate, freq = self.state_dynamics()
        dec = torch.exp(-rate * x.unsqueeze(-1)); ang = freq * x.unsqueeze(-1)
        cr, ci = dec * torch.cos(ang), dec * torch.sin(ang)
        er = cr * zr.unsqueeze(-2) - ci * zi.unsqueeze(-2); ei = cr * zi.unsqueeze(-2) + ci * zr.unsqueeze(-2)
        e = torch.cat([er, ei], -1)
        eta = self.s_out(e).squeeze(-1) + self.s_bias(h)
        log_lam = torch.log(F.softplus(eta).clamp_min(1e-30)) - math.log(self.scale)
        ml = self.s_mark_z(e) + self.s_mark_h(h).unsqueeze(-2)
        if marks and self._extra is not None:
            ml = ml + self._extra[:, :h.shape[1]].unsqueeze(-2)
        log_pk = F.log_softmax(ml, -1) if marks else None
        return log_lam, log_pk

    def state_comp(self, h, zr, zi, tau):
        """Compensator ∫_0^τ λ_s. With a recording cell c the hazard is held at λ_s(c) on [0, c) (the process is not
        resolved below one cell, including at zero gaps): λ_s(c)·min(τ, c) + ∫_c^τ λ_s (Gauss–Legendre on
        τ' = c + (τ − c) x², nodes concentrated near c). Without a cell: Gauss–Legendre on τ' = τ x²."""
        x = self.gl_x
        c = self.floor_cell
        if c <= 0:
            tq = tau.unsqueeze(-1) * x ** 2
            log_lam, _ = self.state_eval(h, zr, zi, tq, marks=False)
            return (log_lam.exp() * (2 * tau.unsqueeze(-1) * x) * self.gl_w).sum(-1)
        lam_c, _ = self.state_eval(h, zr, zi, torch.full_like(tau, c).unsqueeze(-1), marks=False)
        head = lam_c[..., 0].exp() * tau.clamp_max(c)
        span = (tau - c).clamp_min(0)
        tq = c + span.unsqueeze(-1) * x ** 2
        log_lam, _ = self.state_eval(h, zr, zi, tq, marks=False)
        return head + (log_lam.exp() * (2 * span.unsqueeze(-1) * x) * self.gl_w).sum(-1)

    def state_tau(self, tau):
        """Elapsed time at which the state clock's hazard and marks are read: max(τ, cell)."""
        return tau.clamp_min(self.floor_cell) if self.floor_cell > 0 else tau

    def event_terms(self, t, m, mask, target_jitter=0.0):
        """Per-event (time log density, joint log density) and the pieces needed for prediction."""
        h, slots = self.encode(t, m, mask)
        hp = h[:, :-1]
        params = self.clocks(hp, slots[:, :-1])
        tau = (t[:, 1:] - t[:, :-1]).clamp_min(0)
        if target_jitter > 0:            # dequantize only the scored gap inside its recording cell; history stays raw
            u = torch.rand_like(tau)
            tau = torch.where(tau > 0, tau + (u - 0.5) * target_jitter, u * 0.5 * target_jitter).clamp_min(1e-9)
        log_h, log_s = self.clock_terms(tau, *params[:4])
        log_pk = params[4]
        tgt = m[:, 1:, None, None]
        lk_parts = log_h + log_pk.gather(-1, tgt.expand(-1, -1, self.M, 1)).squeeze(-1)
        log_h_all, lk_all, surv = log_h, lk_parts, log_s.sum(-1)
        zr = zi = None
        if self.ns:
            dt = torch.diff(t, dim=1, prepend=t[:, :1]).clamp_min(0) / self.scale
            zr, zi = self.state_seq(h, dt)
            zr, zi = zr[:, :-1], zi[:, :-1]
            ls_lam, ls_pk = self.state_eval(hp, zr, zi, self.state_tau(tau).unsqueeze(-1))
            ls_lam, ls_pk = ls_lam[..., 0], ls_pk[..., 0, :]
            log_h_all = torch.cat([log_h, ls_lam.unsqueeze(-1)], -1)
            lk_all = torch.cat([lk_parts, (ls_lam + ls_pk.gather(-1, m[:, 1:, None]).squeeze(-1)).unsqueeze(-1)], -1)
            surv = surv - self.state_comp(hp, zr, zi, tau)
        time_lp = torch.logsumexp(log_h_all, -1) + surv
        joint_lp = torch.logsumexp(lk_all, -1) + surv
        return time_lp, joint_lp, (h, slots, params, tau, mask[:, 1:], zr, zi)

    def loglik(self, t, m, mask, target_jitter=0.0):
        if True:
            time_lp, joint_lp, cache = self.event_terms(t, m, mask, target_jitter)
            valid = mask[:, 1:]
            return (time_lp * valid).sum(), ((joint_lp - time_lp) * valid).sum(), valid.sum(), cache

    def _loglik_closed_form(self, t, m, mask):
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
        h, slots, params, tau, valid, zr, zi = cache
        h = h[:, :-1]
        log_rate, mu, sigma, log_w, log_pk = params
        u = torch.linspace(-18, 8, grid, dtype=tau.dtype)
        taus = self.scale * torch.exp(u)                                           # log-spaced quadrature in τ
        shape = tau.shape
        tq = taus.view(1, 1, grid).expand(*shape, grid)
        lh, ls = self.clock_terms(tq, *(p.unsqueeze(-2) for p in (log_rate, mu, sigma, log_w)))
        logS = ls.sum(-1)
        if self.ns:
            sl, _ = self.state_eval(h, zr, zi, self.state_tau(tq), marks=False)
            lam = sl.exp()
            inc = 0.5 * (lam[..., 1:] * tq[..., 1:] + lam[..., :-1] * tq[..., :-1]) * (u[1] - u[0])
            first = lam[..., :1] * tq[..., :1]                                    # ∫_0^{τ_0} (tiny)
            logS = logS - torch.cat([first, first + torch.cumsum(inc, -1)], -1)
        S = logS.exp()
        expected = torch.trapz(S * tq, u, dim=-1)
        log_h, _ = self.clock_terms(tau, log_rate, mu, sigma, log_w)
        log_lam_k = torch.logsumexp(log_h.unsqueeze(-1) + log_pk, -2)
        if self.ns:
            sl, spk = self.state_eval(h, zr, zi, self.state_tau(tau).unsqueeze(-1))
            log_lam_k = torch.logaddexp(log_lam_k, sl[..., 0].unsqueeze(-1) + spk[..., 0, :])
        pred_mark = log_lam_k.argmax(-1)
        return expected, pred_mark


def cluster_windows(gaps, n, seed, iters=50, bins=200, empty=1e-4):
    """Initial windows from TRAIN gaps. If the log-gap histogram splits into components separated by near-empty bins
    (Amazon: two flat boxes), give every component ONE window spanning its 0.5%-99.5% range — a logistic window
    represents a flat box exactly — and let spare windows duplicate the most populous components. Otherwise tile with
    1-D k-means clusters (v11)."""
    lg = np.log(gaps)
    h, edges = np.histogram(lg, bins=bins)
    occupied = h > empty * len(lg)
    comps, cur = [], None
    for i, o in enumerate(occupied):
        if o and cur is None:
            cur = [i, i]
        elif o:
            cur[1] = i
        elif cur is not None:
            comps.append(cur); cur = None
    if cur is not None:
        comps.append(cur)
    members = [(lg >= edges[a]) & (lg <= edges[b + 1]) for a, b in comps]
    keep = [i for i, m in enumerate(members) if m.mean() >= 0.05]          # ignore grid fragments and stray tails
    comps = [comps[i] for i in keep]; members = [members[i] for i in keep]
    if len(comps) >= 2:
        sizes = np.array([m.sum() for m in members]); order = list(np.argsort(-sizes))
        owners = [order[i % len(comps)] for i in range(n)]
        lo, hi = [], []
        for j in owners:
            g = gaps[members[j]]
            a, b = np.quantile(g, [0.005, 0.995])
            lo.append(float(a)); hi.append(float(max(b, a * 1.05)))
        return lo, hi
    c = np.quantile(lg, (np.arange(n) + 0.5) / n)
    for _ in range(iters):
        lab = np.abs(lg[:, None] - c[None]).argmin(1)
        c = np.array([lg[lab == j].mean() if (lab == j).any() else c[j] for j in range(n)])
    lab = np.abs(lg[:, None] - c[None]).argmin(1)
    lo, hi = [], []
    for j in range(n):
        g = gaps[lab == j] if (lab == j).sum() > 10 else gaps
        a, b = np.quantile(g, [0.02, 0.98])
        lo.append(float(a)); hi.append(float(max(b, a * 1.05)))
    return lo, hi


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
    ap.add_argument('--state-modes', type=int, default=0, help='complex modes of the continuous-time state clock (0 = off)')
    ap.add_argument('--dequant-target', type=float, default=0.0,
                    help='recording cell: during training, score each target gap jittered uniformly within its cell')
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
        edges = cluster_windows(pos, a.n_window, a.seed)
    model = RaceTPP(K, a.d, a.modes, a.layers, a.n_exp, a.n_lognormal, a.dv, a.dropout, scale, qs, a.floor_cell,
                    a.n_window, edges, a.state_modes)
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
            tl, ml, n, _ = model.loglik(t, m, mask, a.dequant_target)
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
                  dev=evaluate(model, dev, 8, rng), final_dev=final_dev, history=history,
                  checkpoint=str(ckpt.relative_to(ROOT)))
    if a.score_test:
        result['test'] = evaluate(model, load_split(a.dataset, 'test'), 8, rng)
    src = Path(__file__).resolve()
    result['source_sha256'] = {str(src.relative_to(ROOT)): hashlib.sha256(src.read_bytes()).hexdigest()}
    (out_dir / f'{a.tag}.json').write_text(json.dumps(result, indent=1) + '\n')
    print('DEV', json.dumps(result['dev']), flush=True)
    if a.score_test:
        print('TEST', json.dumps(result['test']), flush=True)


if __name__ == '__main__':
    main()
