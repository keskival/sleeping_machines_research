import math, sys, torch
sys.path.insert(0, '/workspace/experiments/tpp')
from race_tpp_v13 import RaceTPP
torch.set_default_dtype(torch.float64); torch.manual_seed(3)
K, scale = 4, 0.5
for cell in (0.0, 0.05):
    model = RaceTPP(K, 16, 8, 2, 2, 3, 4, 0.0, scale, [math.log(0.1), math.log(0.4), math.log(1.2)], cell,
                    n_window=2, window_edges=([0.2, 0.8], [0.5, 1.6]), state_modes=8, quad_nodes=48).eval()
    with torch.no_grad():
        for p in model.parameters(): p.add_(0.3 * torch.randn_like(p))
        hist_t = torch.cumsum(torch.rand(10) * 0.7, 0); hist_m = torch.randint(0, K, (10,))
        u = torch.linspace(-14, 14, 14001); taus = scale * torch.exp(u); G = len(taus)
        t = torch.cat([hist_t.expand(G, -1), (hist_t[-1] + taus).unsqueeze(1)], 1)
        m = torch.cat([hist_m.expand(G, -1), torch.zeros((G, 1), dtype=torch.long)], 1)
        tl, jl, _ = model.event_terms(t, m, torch.ones_like(m, dtype=torch.bool))
        dens = tl[:, -1].exp()
        h, slots, params, tau, valid, zr, zi = _
        log_h_dummy = None
        surv0 = (tl[0, -1] - torch.logsumexp(torch.cat([model.clock_terms(tau[:1, -1:], *(p[:1, -1:] for p in params[:4]))[0][0, 0], model.state_eval(h[:1, -2:-1], zr[:1, -1:], zi[:1, -1:], model.state_tau(tau[:1, -1:]).unsqueeze(-1))[0][0, 0]]), -1))
        below = 1 - float(surv0.exp())
        print('cell', cell, 'grid integral', float(torch.trapz(dens * taus, u)), '+ mass below grid', below, '=', float(torch.trapz(dens * taus, u)) + below)
