#!/usr/bin/env python3
"""Witnesses for theory 175 section 8 (the route-tree limit). A decaying addressed memory with write races and read races:
- full route-tree model: every race is complete (softmax over all S slots for writes and reads);
- hard model (what we train): top-m writes and top-k reads with weights renormalised over the selected set.
(A) at m = k = S the hard model's loss and gradients equal the full model's exactly;
(B) for any m, k: |L_hard - L_full| and the gradient gaps are bounded by a constant times the total excluded race mass
    sum_d (1 - P_S(d)), measured at run time; the gap -> 0 as races sharpen (temperature -> 0) at fixed k."""
import json, sys
from pathlib import Path
import torch
torch.set_default_dtype(torch.float64)
ROOT = Path(__file__).resolve().parents[2]


def model(U, R, delta, C, b, m, k, temp=1.0):
    """U [T, S] write-race scores, R [T, S] read-race scores; returns loss, total excluded mass."""
    T, S = U.shape; d = delta.shape[1]; M = torch.zeros(S, d); L = 0.0; excl = 0.0
    for t in range(T):
        pw = torch.softmax(U[t]/temp, 0); pr = torch.softmax(R[t]/temp, 0)
        if m < S:
            top = pw.topk(m).indices; w = torch.zeros(S).index_add(0, top, pw[top]/pw[top].sum()); excl = excl + float(1 - pw[top].sum())
        else: w = pw
        if k < S:
            top = pr.topk(k).indices; a = torch.zeros(S).index_add(0, top, pr[top]/pr[top].sum()); excl = excl + float(1 - pr[top].sum())
        else: a = pr
        M = torch.exp(-b)[:, None]*M + w[:, None]*delta[t][None, :]; L = L + (torch.tanh(a @ M)*C[t]).sum()
    return L, excl


def main():
    g_ = torch.Generator().manual_seed(0); T, S, d = 30, 16, 5
    U0 = torch.randn(T, S, generator=g_)*2; R0 = torch.randn(T, S, generator=g_)*2; delta0 = torch.randn(T, d, generator=g_); C = torch.randn(T, d, generator=g_); b = torch.rand(S, generator=g_)*0.3 + 0.02
    def grads(m, k, temp=1.0):
        U, R, delta = U0.clone().requires_grad_(True), R0.clone().requires_grad_(True), delta0.clone().requires_grad_(True)
        L, ex = model(U, R, delta, C, b, m, k, temp); gu, gr, gd = torch.autograd.grad(L, (U, R, delta)); return float(L), ex, gu, gr, gd
    out = {'sweep_k': [], 'sweep_temperature': []}
    for temp in (1.0,):
        Lf, _, guf, grf, gdf = grads(S, S, temp)
        for mk in (1, 2, 4, 8, 12, 16):
            L, ex, gu, gr, gd = grads(mk, mk, temp)
            out['sweep_k'].append(dict(m=mk, k=mk, excluded_mass=ex, loss_gap=abs(L - Lf), grad_values_gap=float((gd - gdf).norm()), grad_scores_gap=float((torch.cat([gu, gr]) - torch.cat([guf, grf])).norm()),
                                       loss_gap_per_mass=abs(L - Lf)/ex if ex > 0 else None, grad_values_gap_per_mass=float((gd - gdf).norm())/ex if ex > 0 else None))
    for temp in (1.0, 0.5, 0.25, 0.1, 0.05):
        Lf, _, guf, grf, gdf = grads(S, S, temp); L, ex, gu, gr, gd = grads(4, 4, temp)
        out['sweep_temperature'].append(dict(temperature=temp, excluded_mass=ex, loss_gap=abs(L - Lf), grad_values_gap=float((gd - gdf).norm()), loss_gap_per_mass=abs(L - Lf)/ex if ex > 1e-12 else None))
    full = out['sweep_k'][-1]; out['exact_at_full_races'] = dict(loss_gap=full['loss_gap'], grad_values_gap=full['grad_values_gap'], grad_scores_gap=full['grad_scores_gap'])
    ratios = [r['loss_gap_per_mass'] for r in out['sweep_k'] + out['sweep_temperature'] if r['loss_gap_per_mass'] is not None]
    out['max_loss_gap_per_excluded_mass'] = max(ratios)
    out['status'] = 'passed' if full['loss_gap'] < 1e-12 and full['grad_values_gap'] < 1e-12 and full['grad_scores_gap'] < 1e-12 and out['sweep_temperature'][-1]['loss_gap'] < out['sweep_temperature'][0]['loss_gap'] else 'failed'
    print(json.dumps(out, indent=1))
    if len(sys.argv) > 1: (ROOT/f'experiments/results/credit/{sys.argv[1]}.json').open('x').write(json.dumps(out, indent=1) + '\n')


if __name__ == '__main__':
    main()
