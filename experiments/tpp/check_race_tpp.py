#!/usr/bin/env python3
"""Numerical contract for race_tpp: proper density, marked intensities sum to total, exact compensator."""
import math, sys
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
import importlib
RaceTPP = importlib.import_module(sys.argv[1] if len(sys.argv) > 1 else 'race_tpp').RaceTPP

torch.set_default_dtype(torch.float64); torch.manual_seed(1)
K, scale = 5, 0.3
model = RaceTPP(K, 16, 8, 2, 2, 3, 4, 0.0, scale, [math.log(0.1), math.log(0.3), math.log(0.9)]).eval()
t = torch.cumsum(torch.rand(3, 12) * 0.6, 1); m = torch.randint(0, K, (3, 12)); mask = torch.ones(3, 12, dtype=torch.bool)
with torch.no_grad():
    h, slots = model.encode(t, m, mask)
    log_rate, mu, sigma, log_w, log_pk = model.clocks(h, slots)
    u = torch.linspace(-25, 12, 200001); taus = scale * torch.exp(u)
    sel = (0, 4)
    p = [x[sel] for x in (log_rate, mu, sigma, log_w)]
    lh, ls = model.clock_terms(taus, *(x.unsqueeze(0) for x in p))
    dens = torch.logsumexp(lh, -1).exp() * ls.sum(-1).exp()
    total = torch.trapz(dens * taus, u)
    print('density integral', float(total))
    tau0 = torch.tensor(0.37)
    lh0, ls0 = model.clock_terms(tau0, *p)
    grid = torch.linspace(0, 0.37, 400001)
    lhg, _ = model.clock_terms(grid, *(x.unsqueeze(0) for x in p))
    comp_num = torch.trapz(torch.logsumexp(lhg, -1).exp(), grid)
    print('compensator exact', float(-ls0.sum()), 'numeric', float(comp_num))
    lam_k = torch.logsumexp(lh0.unsqueeze(-1) + log_pk[sel], -2).exp()
    print('sum_k lambda_k / lambda', float(lam_k.sum() / torch.logsumexp(lh0, -1).exp()))
    ok = abs(float(total) - 1) < 1e-4 and abs(float(-ls0.sum()) - float(comp_num)) < 1e-4 and abs(float(lam_k.sum() / torch.logsumexp(lh0, -1).exp()) - 1) < 1e-10
model.train()
tl, ml, n, _ = model.loglik(t, m, mask); (-(tl + ml) / n).backward()
finite = all(torch.isfinite(q.grad).all() for q in model.parameters() if q.grad is not None)
print('gradients finite', finite); print('PASS' if ok and finite else 'FAIL')
