#!/usr/bin/env python3
"""Numerical check of theory note 158, Theorem 1: races of one exponential and M defective log-normal clocks approximate
any continuous inter-event density on a compact gap range, with error shrinking as M grows.

Targets (gaps in [0.05, 50]): (A) two log-normal modes separated by a gap of almost no mass; (B) a truncated Pareto
(heavy tail, α = 1.2); (C) a narrow spike (σ = 0.05 in log-time) on a broad background. For each target and
M ∈ {1, 2, 4, 8, 16} the race's parameters are fitted by maximum likelihood on 20,000 samples (Adam, log-time
initialisation at sample quantiles), and evaluated by KL(target ‖ race) and the mean absolute log-density error on a
log-spaced grid over the target's central 99% range. Writes results/theory/race_universality_check.json and a figure.
"""
import json
import math
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
torch.set_default_dtype(torch.float64); torch.set_num_threads(1)
LO, HI = 0.05, 50.0


def targets(rng, n):
    def mixA(n):
        z = rng.random(n) < 0.6
        return np.where(z, rng.lognormal(math.log(0.3), 0.3, n), rng.lognormal(math.log(8.0), 0.25, n))
    def logpdfA(x):
        f = lambda m, s: np.exp(-(np.log(x) - m) ** 2 / (2 * s * s)) / (x * s * math.sqrt(2 * math.pi))
        return np.log(0.6 * f(math.log(0.3), 0.3) + 0.4 * f(math.log(8.0), 0.25))
    a = 1.2
    def parB(n):
        u = rng.random(n); c = 1 - (LO / HI) ** a
        return LO * (1 - u * c) ** (-1 / a)
    def logpdfB(x):
        c = 1 - (LO / HI) ** a
        return np.log(a * LO ** a / c) - (a + 1) * np.log(x)
    def mixC(n):
        z = rng.random(n) < 0.3
        return np.where(z, rng.lognormal(math.log(2.0), 0.05, n), rng.lognormal(math.log(2.0), 1.0, n))
    def logpdfC(x):
        f = lambda m, s: np.exp(-(np.log(x) - m) ** 2 / (2 * s * s)) / (x * s * math.sqrt(2 * math.pi))
        return np.log(0.3 * f(math.log(2.0), 0.05) + 0.7 * f(math.log(2.0), 1.0))
    out = {}
    for name, sam, lp in (('A_two_modes', mixA, logpdfA), ('B_heavy_tail', parB, logpdfB), ('C_spike', mixC, logpdfC)):
        x = sam(4 * n); x = x[(x > LO) & (x < HI)][:n]
        out[name] = (x, lp)
    return out


class Race(torch.nn.Module):
    def __init__(self, M, x):
        super().__init__()
        q = np.quantile(np.log(x), np.linspace(0.05, 0.95, M))
        self.mu = torch.nn.Parameter(torch.tensor(q)); self.ls = torch.nn.Parameter(torch.full((M,), -0.5))
        self.fire = torch.nn.Parameter(torch.zeros(M)); self.lr = torch.nn.Parameter(torch.tensor(-3.0))

    def logpdf(self, t):
        t = t.unsqueeze(-1); s = self.ls.exp() + 0.01; z = (t.log() - self.mu) / s
        pi = torch.sigmoid(self.fire); rate = self.lr.exp()
        log_s0 = torch.special.log_ndtr(-z); log_f0 = -t.log() - s.log() - 0.5 * math.log(2 * math.pi) - 0.5 * z * z
        log_S = torch.logaddexp(torch.log1p(-pi), pi.log() + log_s0)                  # defective clocks
        log_h = pi.log() + log_f0 - log_S
        log_H_total = log_S.sum(-1) - rate * t.squeeze(-1)                             # survival of the race
        log_hz = torch.logsumexp(torch.cat([log_h, self.lr.expand(t.shape[:-1]).unsqueeze(-1)], -1), -1)
        return log_hz + log_H_total


def fit(M, x, steps=3000):
    m = Race(M, x); opt = torch.optim.Adam(m.parameters(), lr=0.03); xt = torch.tensor(x)
    for _ in range(steps):
        loss = -m.logpdf(xt).mean(); opt.zero_grad(); loss.backward(); opt.step()
    return m


def main():
    rng = np.random.default_rng(0); res = {}
    for name, (x, lp) in targets(rng, 20000).items():
        lo, hi = np.quantile(x, [0.005, 0.995]); grid = np.exp(np.linspace(math.log(lo), math.log(hi), 400))
        rows = []
        for M in (1, 2, 4, 8, 16):
            torch.manual_seed(0); m = fit(M, x)
            with torch.no_grad():
                g = m.logpdf(torch.tensor(grid)).numpy(); kl = float(np.mean(lp(x) - m.logpdf(torch.tensor(x)).numpy()))
            rows.append(dict(M=M, kl_target_to_race=kl, mean_abs_logdensity_error=float(np.mean(np.abs(g - lp(grid))))))
            print(name, rows[-1], flush=True)
        res[name] = rows
    out = ROOT / 'experiments/results/theory'; out.mkdir(parents=True, exist_ok=True)
    (out / 'race_universality_check.json').write_text(json.dumps(res, indent=1) + '\n')
    try:
        import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
        for name, rows in res.items():
            Ms = [r['M'] for r in rows]
            ax[0].plot(Ms, [max(r['kl_target_to_race'], 1e-4) for r in rows], 'o-', label=name)
            ax[1].plot(Ms, [r['mean_abs_logdensity_error'] for r in rows], 'o-', label=name)
        for a_, t_ in zip(ax, ('KL(target ‖ race), nats', 'mean |log density error|')):
            a_.set_xscale('log', base=2); a_.set_yscale('log'); a_.set_xlabel('delayed clocks M'); a_.set_title(t_)
        ax[0].legend(fontsize=8); fig.tight_layout(); fig.savefig(out / 'race_universality_check.png', dpi=130)
    except Exception as e:
        print('figure skipped:', e)


if __name__ == '__main__':
    main()
