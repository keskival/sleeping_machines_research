#!/usr/bin/env python3
"""Contract for race_tpp_v10's state clock: with history fixed, the next-gap density (exact clocks + state clock with
Gauss-Legendre compensator) integrates to 1 over τ, and marked densities sum to the time density."""
import math, sys
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
from race_tpp_v10 import RaceTPP

torch.set_default_dtype(torch.float64); torch.manual_seed(3)
K, scale = 4, 0.5
model = RaceTPP(K, 16, 8, 2, 2, 3, 4, 0.0, scale, [math.log(0.1), math.log(0.4), math.log(1.2)],
                n_window=2, window_edges=([0.2, 0.8], [0.5, 1.6]), state_modes=8, quad_nodes=48).eval()
with torch.no_grad():
    for p in model.parameters():
        p.add_(0.3 * torch.randn_like(p))                       # move away from initialization
    hist_t = torch.cumsum(torch.rand(10) * 0.7, 0); hist_m = torch.randint(0, K, (10,))
    u = torch.linspace(-14, 6, 6001); taus = scale * torch.exp(u)
    G = len(taus)
    t = torch.cat([hist_t.expand(G, -1), (hist_t[-1] + taus).unsqueeze(1)], 1)
    total_density, mark_sum = 0.0, []
    time_lp = None
    for k in range(K):
        m = torch.cat([hist_m.expand(G, -1), torch.full((G, 1), k)], 1)
        tl, jl, _ = model.event_terms(t, m, torch.ones_like(m, dtype=torch.bool))
        time_lp = tl[:, -1]; mark_sum.append(jl[:, -1].exp())
    dens = time_lp.exp()
    integral = float(torch.trapz(dens * taus, u))
    ratio = float((torch.stack(mark_sum).sum(0) / dens).max() - 1)
print('density integral', integral, ' max |sum_k p_k / p - 1|', abs(ratio))
model.train()
tl, ml, n, _ = model.loglik(t[:64], m[:64], torch.ones_like(m[:64], dtype=torch.bool)); (-(tl + ml) / n).backward()
finite = all(torch.isfinite(q.grad).all() for q in model.parameters() if q.grad is not None)
ok = abs(integral - 1) < 2e-3 and abs(ratio) < 1e-9 and finite
print('gradients finite', finite); print('PASS' if ok else 'FAIL')
