#!/usr/bin/env python3
"""Theory note 158 §455 Prediction 4: exact-compensator gradients vs Monte Carlo compensator gradients.

On a fixed batch of Taxi TRAIN sequences and the reproduced race_tpp_v5 checkpoint, compute the gradient of the
negative log-likelihood (a) with the exact closed-form compensator (the model's own objective) and (b) with the
compensator ∫_0^τ λ(s) ds replaced by the uniform Monte Carlo estimate τ · mean_j λ(u_j τ), u_j ~ U(0,1), with
J ∈ {1, 10, 100} points per interval (EasyTPP's scorer uses 10). Repeat (b) R times. Report the relative gradient noise
‖g_MC − g_exact‖ / ‖g_exact‖ (mean over repeats) and the bias ‖mean(g_MC) − g_exact‖ / ‖g_exact‖.
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'experiments/tpp'))
from race_tpp_v5 import RaceTPP, batches, load_split  # noqa: E402

torch.set_default_dtype(torch.float64); torch.set_num_threads(1)


def flat_grad(model, loss):
    model.zero_grad(); loss.backward()
    return torch.cat([p.grad.flatten() for p in model.parameters() if p.grad is not None]).clone()


def mc_nll(model, t, m, mask, J, gen):
    h, slots = model.encode(t, m, mask)
    params = model.clocks(h[:, :-1], slots[:, :-1])
    tau = (t[:, 1:] - t[:, :-1]).clamp_min(0)
    log_h, _ = model.clock_terms(tau, *params[:4])                                   # intensity at the event
    log_pk = params[4]
    log_lam_k = torch.logsumexp(log_h + log_pk.gather(-1, m[:, 1:, None, None].expand(-1, -1, model.M, 1)).squeeze(-1), -1)
    u = torch.rand(*tau.shape, J, generator=gen)
    s = u * tau.unsqueeze(-1)                                                        # [B, L-1, J]
    lh_s, _ = model.clock_terms(s, *(p.unsqueeze(-2) for p in params[:4]))
    lam_s = torch.logsumexp(lh_s, -1).exp()                                          # total intensity at samples
    comp = tau * lam_s.mean(-1)
    valid = mask[:, 1:]
    return -((log_lam_k - comp) * valid).sum() / valid.sum()


def main():
    res = json.loads((ROOT / 'experiments/results/tpp/curie_repro_taxi_v5_s0_20261007T0510Z.json').read_text()); a = res['args']
    train = load_split('taxi', 'train')
    gaps = np.concatenate([np.diff(t) for t, _ in train]); pos = gaps[gaps > 0]
    qs = np.log(np.quantile(pos, np.linspace(0.1, 0.9, a['n_lognormal']))).tolist()
    model = RaceTPP(res['K'], a['d'], a['modes'], a['layers'], a['n_exp'], a['n_lognormal'], a['dv'], a['dropout'],
                    res['scale'], qs, a.get('floor_cell', 0.0))
    model.load_state_dict(torch.load(ROOT / res['checkpoint'])); model.eval()
    import random
    t, m, mask = next(iter(batches(train[:64], 64, False, random.Random(0))))
    tl, ml, n, _ = model.loglik(t, m, mask)
    g_exact = flat_grad(model, -(tl + ml) / n); norm = float(g_exact.norm())
    out = dict(checkpoint=res['checkpoint'], batch_sequences=64, events=int(n), exact_grad_norm=norm, mc={})
    gen = torch.Generator().manual_seed(0)
    for J in (1, 10, 100):
        gs = torch.stack([flat_grad(model, mc_nll(model, t, m, mask, J, gen)) for _ in range(40)])
        noise = float(((gs - g_exact).norm(dim=1) / norm).mean()); bias = float((gs.mean(0) - g_exact).norm() / norm)
        out['mc'][J] = dict(relative_noise=noise, relative_bias_of_mean=bias, repeats=40)
        print(J, out['mc'][J], flush=True)
    p = ROOT / 'experiments/results/theory/compensator_gradient_variance.json'; p.write_text(json.dumps(out, indent=1) + '\n')
    print('RESULT', json.dumps(out))


if __name__ == '__main__':
    main()
