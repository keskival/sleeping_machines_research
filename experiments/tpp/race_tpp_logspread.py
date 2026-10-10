"""B4 development: same clock likelihood with log-domain spread intermediates.

Derived from v19; clocks() returns log spreads internally, including log window
edges. Inherited clock_terms/event_terms consume this only through the override.
No parameter names, fitting settings, or represented real-valued functions change.
"""
import math
import torch
import torch.nn.functional as F
from race_tpp_v19 import RaceTPP as Original, LOG2PI

class RaceTPP(Original):
    def clocks(self, h, slots):
        """Clock parameters after event i, for the gap to event i+1. h: [B,L,d], slots: [B,L,K,dv]."""
        B, L, _ = h.shape
        raw = self.clock(torch.cat([h, slots.flatten(-2)], -1))
        ne, nl = self.n_exp, self.n_ln
        log_rate = raw[..., :ne] - math.log(self.scale)
        mu = raw[..., ne:ne + nl] + math.log(self.scale)
        # Resolution floor: a clock may not resolve time below the data's recording cell (its spread at its
        # delay e^mu is at least one cell), so densities cannot spike on recording-grid points.
        raw_sigma = raw[..., ne + nl:ne + 2 * nl]
        log_base = torch.where(raw_sigma < -30, raw_sigma,
                               F.softplus(raw_sigma).clamp_min(1e-300).log())
        log_floor = (math.log(self.floor_cell) - mu).clamp_min(math.log(0.005)) if self.floor_cell > 0 else torch.full_like(mu, math.log(0.005))
        sigma = torch.logaddexp(log_base, log_floor) # internal representation: log spread
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
            sigma = torch.cat([sigma, edge.log()], -1)                                     # logistic edge scale s
            log_w = torch.cat([log_w, F.logsigmoid(raw[..., o + 3 * nw:o + 4 * nw])], -1)
        logits = self.mark_ctx(h).view(B, L, self.M, self.K) + torch.einsum('blkv,mv->blmk', slots, self.mark_slot)
        return log_rate, mu, sigma, log_w, F.log_softmax(logits, -1)

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
        ls_exp = -(log_rate + log_w[..., :ne]).exp() * tau
        lh_exp = (log_rate + log_w[..., :ne]).expand_as(ls_exp)
        log_pi = log_w[..., ne:]
        log_1mpi = torch.log1p(-log_pi.exp().clamp_max(1 - 1e-12))
        pos = tau > 0
        lt = torch.log(tau.clamp_min(1e-30))
        z = (lt - mu) * torch.exp(-sigma)
        log_s0 = torch.special.log_ndtr(-z)
        log_f0 = -lt - sigma - 0.5 * LOG2PI - 0.5 * z ** 2
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
        s = win_ab.exp()
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

