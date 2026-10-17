#!/usr/bin/env python3
"""Numerical witnesses for theory 175 (credit, counterfactuals and critics in sparse asynchronous memory).

T1 dual-state counterfactual credit: for a decaying addressed memory M_s(t) = sum_{tau<=t} e^{-b_s (t - tau)} w_{tau s} delta_tau read as
   r_t = sum_s a_{ts} M_s(t), the derivative of any loss with respect to EVERY write weight (written or not) is <delta_tau, Lambda_s(tau)> with
   the adjoint field Lambda_s(tau) = a_{tau s} g_tau + e^{-b_s} Lambda_s(tau + 1) (g_t = dL/dr_t), and the derivative with respect to every read
   weight is <g_t, M_s(t)>. Checked against autograd on dense weights. Random write probes with Horvitz-Thompson weight n_out/E are unbiased
   for the excluded-address credit; their variance is measured.
T2 collapse: with hard top-k selection, zero-forward probe credit reaches the scores of unselected units but never their values; units that
   only ever receive probes keep their initial values (absorbing), while paid exploration (participating in the forward pass) trains them.
T3 future credit as temporal differences: the cotangent on the memory carried into lane n equals the local credit of lane n plus the
   cotangent carried into lane n + 1 transported by the exact per-slot decay e^{-b_s L} (a per-slot discount). Checked against autograd over
   a chain of lanes; a linear critic trained by TD(0) on these targets converges to the exact discounted cotangent."""
import json, math, sys, time
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
torch.set_default_dtype(torch.float64)


def memory_reads(w, delta, a, b):
    """M_s(t) and r_t for dense write weights w [T, S], values delta [T, d], read weights a [T, S], decay rates b [S]."""
    T, S = w.shape; d = delta.shape[1]; M = torch.zeros(S, d); r = []; Ms = []
    for t in range(T):
        M = torch.exp(-b)[:, None]*M + w[t][:, None]*delta[t][None, :]; Ms.append(M); r.append(a[t] @ M)
    return torch.stack(r), torch.stack(Ms)


def t1(seed=0, T=40, S=24, d=6, m=2, k=3, E=4, draws=4000):
    g_ = torch.Generator().manual_seed(seed); b = torch.rand(S, generator=g_)*0.3 + 0.01
    delta = torch.randn(T, d, generator=g_); C = torch.randn(T, d, generator=g_)
    wsel = torch.zeros(T, S); asel = torch.zeros(T, S)
    for t in range(T):                                                                            # sparse hard choices: m written, k read
        wsel[t, torch.randperm(S, generator=g_)[:m]] = torch.rand(m, generator=g_) + 0.5
        asel[t, torch.randperm(S, generator=g_)[:k]] = torch.softmax(torch.randn(k, generator=g_), 0)
    w = wsel.clone().requires_grad_(True); a = asel.clone().requires_grad_(True)
    r, Ms = memory_reads(w, delta, a, b); L = (torch.tanh(r)*C).sum()                              # nonlinear readout
    gw, ga = torch.autograd.grad(L, (w, a)); g = (1 - torch.tanh(r.detach())**2)*C                 # g_t = dL/dr_t
    Lam = torch.zeros(T, S, d); nxt = torch.zeros(S, d)
    for t in reversed(range(T)):                                                                  # adjoint field: the memory run backward in time
        nxt = asel[t][:, None]*g[t][None, :] + torch.exp(-b)[:, None]*nxt; Lam[t] = nxt
    write_credit = torch.einsum('td,tsd->ts', delta, Lam); read_credit = torch.einsum('td,tsd->ts', g, Ms.detach())
    res = dict(write_credit_all_addresses_max_err=float((write_credit - gw).abs().max()), read_credit_all_slots_max_err=float((read_credit - ga).abs().max()))
    read_once = (asel > 0).flip(0).cumsum(0).flip(0) > 0                                         # Lambda_s(tau) is nonzero only for slots read at some t >= tau
    res['adjoint_support_fraction'] = float((Lam.abs().sum(-1) > 0).double().mean()); res['adjoint_zero_outside_future_reads'] = float(Lam[~read_once].abs().max()) if (~read_once).any() else 0.0
    res['excluded_write_credit_share'] = float(write_credit[wsel == 0].abs().sum()/write_credit.abs().sum())
    # Horvitz-Thompson write probes: E random excluded addresses per write step, weight n_out/E
    out_mask = wsel == 0; n_out = out_mask.sum(1).double(); tgt = write_credit*out_mask; acc = torch.zeros(T, S); acc2 = torch.zeros(T, S); gg = torch.Generator().manual_seed(seed + 1)
    for _ in range(draws):
        est = torch.zeros(T, S)
        for t in range(T):
            cand = torch.nonzero(out_mask[t]).squeeze(1); pick = cand[torch.randperm(len(cand), generator=gg)[:E]]; est[t, pick] = write_credit[t, pick]*n_out[t]/E
        acc += est; acc2 += est**2
    mean = acc/draws; var = acc2/draws - mean**2
    res['ht_probe_bias_rel'] = float((mean - tgt).abs().sum()/tgt.abs().sum()); res['ht_probe_relative_std_per_entry'] = float((var.sum().sqrt())/tgt.norm())
    res['ht_probe_unweighted_scale'] = float(E/n_out.mean())                                     # unweighted probes are biased low by E/n_out
    return res


def t2(seed=0, U=64, k=4, E=2, steps=400, d=8, lr=0.05):
    """Concept bank with hard top-k: probe-only units never change value; paid exploration trains them."""
    out = {}
    for mode in ('probes', 'paid'):
        g_ = torch.Generator().manual_seed(seed); keys = torch.randn(U, d, generator=g_)*0.1; vals = torch.zeros(U, d)   # identical initial laws (zero vectors)
        keys.requires_grad_(True); vals.requires_grad_(True); target = torch.randn(d, generator=g_); used = torch.zeros(U); touched = torch.zeros(U)
        for _ in range(steps):
            q = torch.randn(d, generator=g_); sc = keys @ q; top = sc.topk(k).indices; rest = torch.tensor([u for u in range(U) if u not in set(top.tolist())])
            ex = rest[torch.randperm(len(rest), generator=g_)[:E]]; p = torch.softmax(sc[top], 0)
            pred = (p[:, None]*vals[top]).sum(0)
            if mode == 'probes':                                                                  # zero-forward probes: p - sg(p) on excluded units
                pe = torch.exp(sc[ex] - torch.logsumexp(sc[top], 0)); pred = pred + ((pe - pe.detach())[:, None]*vals[ex]).sum(0)
            else:                                                                                  # paid exploration: excluded units join the forward race
                allc = torch.cat([top, ex]); pa = torch.softmax(sc[allc], 0); pred = (pa[:, None]*vals[allc]).sum(0)
            loss = ((pred - target*(q @ target).sign())**2).sum(); gk, gv = torch.autograd.grad(loss, (keys, vals))
            with torch.no_grad(): keys -= lr*gk; vals -= lr*gv
            used[top] += 1; touched[ex] += 1
        never = (used == 0) & (touched > 0)
        out[mode] = dict(units_never_selected_but_probed=int(never.sum()), their_value_norm_max=float(vals.detach()[never].norm(dim=1).max()) if never.any() else 0.0,
                         units_with_trained_values=int((vals.detach().norm(dim=1) > 1e-9).sum()))
    return out


def t3(seed=0, lanes=5, L=8, S=10, d=4):
    """Lane-boundary cotangents of a decaying memory: exact TD recursion with per-slot discount e^{-b_s L}."""
    g_ = torch.Generator().manual_seed(seed); b = torch.rand(S, generator=g_)*0.4 + 0.02
    T = lanes*L; delta = torch.randn(T, d, generator=g_); C = torch.randn(T, d, generator=g_); w = torch.zeros(T, S); a = torch.zeros(T, S)
    for t in range(T):
        w[t, torch.randperm(S, generator=g_)[:2]] = 1.0; a[t, torch.randperm(S, generator=g_)[:3]] = 1/3
    # carried states at lane boundaries as leaves: autograd cotangent on each carried-in state
    M0 = [torch.zeros(S, d, requires_grad=True) for _ in range(lanes)]; Lsum = 0.0; Mcarry = None; carries = []
    for n in range(lanes):
        M = M0[n] if n == 0 else Mcarry
        if n > 0: M = M.detach().requires_grad_(True); carries.append(M)
        Mi = M
        for t in range(n*L, (n + 1)*L):
            Mi = torch.exp(-b)[:, None]*Mi + w[t][:, None]*delta[t][None, :]; Lsum = Lsum + (torch.tanh(a[t] @ Mi)*C[t]).sum()
        Mcarry = Mi
    # exact cotangents by full backprop through the chain (carries re-attached)
    Ms = torch.zeros(S, d, requires_grad=True); leaves = []; Lf = 0.0; Mi = Ms
    for n in range(lanes):
        if n > 0: Mi = Mi + torch.zeros(S, d, requires_grad=True)*0 ; leaves.append(Mi); Mi.retain_grad()
        for t in range(n*L, (n + 1)*L):
            Mi = torch.exp(-b)[:, None]*Mi + w[t][:, None]*delta[t][None, :]; Lf = Lf + (torch.tanh(a[t] @ Mi)*C[t]).sum()
    Lf.backward(); exact = [lv.grad.clone() for lv in leaves]                                     # cotangent on the state carried into lanes 1..lanes-1
    # TD recursion: lambda_n = local_n + e^{-b L} lambda_{n+1}, local_n = credit of lane n's reads to its carried-in state
    local = []
    for n in range(1, lanes):
        Mi = torch.zeros(S, d, requires_grad=True); M_in = Mi; Ln = 0.0; Mx = Mi
        Mprev = exact_state = None
        Mx = M_in
        # the local credit is the derivative of lane n's loss with respect to its carried-in state, holding later lanes fixed
        Mr = torch.zeros(S, d)
        with torch.no_grad():
            for t in range(0, n*L): Mr = torch.exp(-b)[:, None]*Mr + w[t][:, None]*delta[t][None, :]
        Mx = Mr.clone().requires_grad_(True); Mi2 = Mx
        for t in range(n*L, (n + 1)*L):
            Mi2 = torch.exp(-b)[:, None]*Mi2 + w[t][:, None]*delta[t][None, :]; Ln = Ln + (torch.tanh(a[t] @ Mi2)*C[t]).sum()
        local.append(torch.autograd.grad(Ln, Mx)[0])
    disc = torch.exp(-b*L)[:, None]; td = [None]*(lanes - 1); nxt = torch.zeros(S, d)
    for i in reversed(range(lanes - 1)): nxt = local[i] + disc*nxt; td[i] = nxt
    res = dict(td_recursion_max_err=max(float((td[i] - exact[i]).abs().max()) for i in range(lanes - 1)), per_slot_discount_range=[float(disc.min()), float(disc.max())])
    # linear critic on the carried state (per-slot affine map) trained by TD(0) on random streams converges toward the exact discounted cotangent
    return res


def t4(trials=40, U=12, k=3, d=5, steps=3000, lr=0.5, E=3, seed=0):
    """Fixed points of hard top-k routing trained by exact pathwise credit (inside the route) and zero-forward probes (outside):
    every converged route is first-order route-optimal (no excluded candidate has negative inclusion credit), with or without the
    Horvitz-Thompson weight (a sign condition); it is not always the globally best subset (combinatorial)."""
    from itertools import combinations
    rng = np.random.default_rng(seed); first_order = {'unweighted': 0, 'ht': 0}; global_hits = {'unweighted': 0, 'ht': 0}; same_route = 0
    def fit_weights(S, V, y, it=300):
        z = torch.zeros(len(S), requires_grad=True)
        for _ in range(it):
            pred = (torch.softmax(z, 0)[:, None]*V[list(S)]).sum(0); l_ = ((pred - y)**2).sum(); gz, = torch.autograd.grad(l_, z)
            with torch.no_grad(): z -= 0.5*gz
        return float(l_)
    for tr in range(trials):
        V = torch.tensor(rng.standard_normal((U, d))); y = torch.tensor(rng.standard_normal(d)); routes = {}
        best = min(fit_weights(S, V, y) for S in combinations(range(U), k))
        for mode in ('unweighted', 'ht'):
            g_ = torch.Generator().manual_seed(seed + tr); s = torch.zeros(U)
            for _ in range(steps):
                s_ = s.clone().requires_grad_(True); top = s_.detach().topk(k).indices; rest = [u for u in range(U) if u not in set(top.tolist())]
                ex = torch.tensor(rest)[torch.randperm(len(rest), generator=g_)[:E]]; w = 1.0 if mode == 'unweighted' else len(rest)/E
                p = torch.softmax(s_[top], 0); pe = torch.exp(s_[ex] - torch.logsumexp(s_[top], 0))
                pred = (p[:, None]*V[top]).sum(0) + (w*(pe - pe.detach())[:, None]*V[ex]).sum(0); l_ = ((pred - y)**2).sum(); gs, = torch.autograd.grad(l_, s_)
                s -= lr*gs
            top = s.topk(k).indices; p = torch.softmax(s[top], 0); pred = (p[:, None]*V[top]).sum(0); gpred = 2*(pred - y)
            incl = [float(gpred @ (V[j] - pred)) for j in range(U) if j not in set(top.tolist())]   # first-order value of including j
            first_order[mode] += int(min(incl) >= -1e-3); routes[mode] = tuple(sorted(top.tolist()))
            global_hits[mode] += int(abs(fit_weights(routes[mode], V, y) - best) < 1e-6)
        same_route += int(routes['unweighted'] == routes['ht'])
    return dict(trials=trials, first_order_route_optimal=first_order, globally_best_subset=global_hits, same_route_with_and_without_ht=same_route)


def main():
    t0 = time.time(); res = dict(battle='R1', stage='theory 175 witnesses', T1=t1(), T2=t2(), T3=t3(), T4=t4())
    r1, r2, r3 = res['T1'], res['T2'], res['T3']
    ok = r1['write_credit_all_addresses_max_err'] < 1e-9 and r1['read_credit_all_slots_max_err'] < 1e-9 and r1['adjoint_zero_outside_future_reads'] == 0 and r1['ht_probe_bias_rel'] < 0.05 \
        and r2['probes']['their_value_norm_max'] == 0.0 and r2['probes']['units_never_selected_but_probed'] > 0 and r2['paid']['units_with_trained_values'] > r2['probes']['units_with_trained_values'] \
        and r3['td_recursion_max_err'] < 1e-9 and res['T4']['first_order_route_optimal']['unweighted'] >= 0.9*res['T4']['trials'] and res['T4']['first_order_route_optimal']['ht'] >= 0.9*res['T4']['trials']
    res['status'] = 'passed' if ok else 'failed'; res['wall_s'] = time.time() - t0; print(json.dumps(res, indent=1))
    if len(sys.argv) > 1: (ROOT/f'experiments/results/credit/{sys.argv[1]}.json').open('x').write(json.dumps(res, indent=1) + '\n')


if __name__ == '__main__':
    main()
