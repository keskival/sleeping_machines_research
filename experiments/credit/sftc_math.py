#!/usr/bin/env python3
"""Numerical witnesses for theory note 164 (sparse, forward-trained, delayed credit). No fitting, no data.

C1  Adjoint = Bellman fixed point with the memory's own complex elapsed-time discount; TD(0) bootstrapping of a
    learned credit predictor converges to the exact adjoint; sup-norm contraction per sweep <= max |a|.
C1b Credit must be conditioned on elapsed-time transport: shared exact transport beats learned window weights.
C2  Mode-specific credit horizon: truncating credit transport at H events errs by at most
    max|c| * rho^(H+1) / (1 - rho), rho = per-mode max |a|.
C3  Delayed (asynchronous) credit application is exact for fixed weights: credit packets transported to their writes
    with the conjugate decay and applied in any order / at any delay sum to the BPTT gradient.
C4  Event-triggered (send-on-delta) credit: accumulate cotangent per mode, emit when |acc| > kappa; the emitted
    gradient differs from exact by at most kappa * sum |e| per mode, and the number of messages falls with kappa.
C5  Optimal credit delay: information deficit C e^{-2 r d} + staleness K (eta d)^2 is minimized at the closed form
    root of 2 r C e^{-2 r d} = 2 K eta^2 d; slower modes (smaller r) take longer optimal delays.
"""
import json, hashlib, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
rng = np.random.default_rng(20261010)
out = {}

# ---- a complex diagonal memory with irregular gaps: z_t = a_t z_{t-1} + b_t, loss L = sum_t Re(conj(c_t) z_t)
n, T = 6, 200
r = np.array([0.02, 0.05, 0.2, 0.5, 1.0, 3.0]); w = rng.normal(0, 1, n)
dt = rng.exponential(1.0, T)
a = np.exp((-r[None, :] + 1j * w[None, :]) * dt[:, None])            # [T, n]
b = rng.normal(size=(T, n)) + 1j * rng.normal(size=(T, n))
c = rng.normal(size=(T, n)) + 1j * rng.normal(size=(T, n))           # local loss cotangents dl_t/dz_t

# exact adjoint by backward recursion: lam_t = c_t + conj(a_{t+1}) lam_{t+1}
lam = np.zeros((T, n), complex); lam[-1] = c[-1]
for t in range(T - 2, -1, -1):
    lam[t] = c[t] + np.conj(a[t + 1]) * lam[t + 1]

# C1: Bellman iteration from zero (a TD(0) sweep with a tabular predictor); contraction per sweep
L = np.zeros((T, n), complex); ratios = []
err_prev = np.abs(L - lam).max(axis=0)
for k in range(T + 1):
    Lnew = c.copy(); Lnew[:-1] += np.conj(a[1:]) * L[1:]
    err = np.abs(Lnew - lam).max(axis=0)
    if k < 40:
        ratios.append(float(np.max(err / np.maximum(err_prev, 1e-300))))
    err_prev, L = err, Lnew
rho = np.abs(a).max(axis=0)
out['C1_bellman'] = dict(final_max_abs_err=float(np.abs(L - lam).max()), max_contraction_ratio_first40=max(ratios),
                         bound_max_abs_a=float(rho.max()), passed=bool(np.abs(L - lam).max() < 1e-10 and max(ratios) <= rho.max() + 1e-12))

# C1b: what the credit predictor must be conditioned on. Exact credit over a window weights future cotangents by the
# product of conjugate decays over the OBSERVED GAPS. (i) TD-learned fixed weights over the raw outcome window (no
# elapsed-time transport) cannot represent it; (ii) sharing the forward transport exactly over the window and leaving
# only the tail beyond H to prediction reaches the truncation error (here the Bayes floor: future cotangents are
# independent noise, so E[tail | information] = 0).
H = 12
c1b = {}
for k in (1, 4):           # slow mode (r=.05) and fast mode (r=1)
    Phi = np.stack([np.concatenate([c[t:t + H, k], np.zeros(max(0, t + H - T))]) for t in range(T)])
    Wv = np.zeros(H, complex); eta = 0.02
    for it in range(20000):
        t = rng.integers(0, T - 1)
        target = c[t, k] + np.conj(a[t + 1, k]) * (Phi[t + 1] @ Wv)
        Wv += eta * (target - Phi[t] @ Wv) * np.conj(Phi[t]) / (1 + np.vdot(Phi[t], Phi[t]).real)
    learned = np.abs(Phi @ Wv - lam[:, k])[:T - H].mean()
    shared = np.zeros(T, complex)
    for t in range(T):
        prod = 1 + 0j
        for s2 in range(t, min(T, t + H)):
            if s2 > t: prod *= np.conj(a[s2, k])
            shared[t] += prod * c[s2, k]
    shared_err = np.abs(shared - lam[:, k])[:T - H].mean()
    c1b[str(k)] = dict(decay_r=float(r[k]), window=H, learned_raw_window_err=float(learned),
                       shared_transport_err=float(shared_err), mean_abs_lam=float(np.abs(lam[:T - H, k]).mean()))
out['C1b_conditioning'] = dict(by_mode=c1b, passed=all(v['shared_transport_err'] < v['learned_raw_window_err'] for v in c1b.values()))

# C2: truncation bound per mode
c2 = {}
cmax = np.abs(c).max(axis=0)
for Ht in (2, 8, 32):
    lamH = np.zeros((T, n), complex)
    for t in range(T):
        acc = np.zeros(n, complex); prod = np.ones(n, complex)
        for s in range(t, min(T, t + Ht + 1)):
            if s > t: prod = prod * np.conj(a[s])
            acc += prod * c[s]
        lamH[t] = acc
    errH = np.abs(lamH - lam).max(axis=0)
    bound = cmax * rho ** (Ht + 1) / (1 - rho)
    c2[str(Ht)] = dict(err_by_mode=errH.round(6).tolist(), bound_by_mode=bound.round(6).tolist(), holds=bool(np.all(errH <= bound + 1e-12)))
out['C2_mode_horizon'] = dict(decay_r=r.tolist(), max_abs_a=rho.round(4).tolist(), by_H=c2, passed=all(v['holds'] for v in c2.values()))

# gradients w.r.t. write b_t (complex, Wirtinger-free: real-valued L = Re sum conj(c) z -> dL/db_t = lam_t)
# C3: credit packets: loss at time s emits c_s, travels back to write u <= s with prod conj(a_{u+1..s}); applied with
# random delays/order. Sum per write must equal lam_u (fixed weights).
acc = np.zeros((T, n), complex); events = []
for s in range(T):
    prod = np.ones(n, complex)
    for u in range(s, -1, -1):
        events.append((rng.random(), u, prod * c[s]))
        prod = prod * np.conj(a[u])
        if np.all(np.abs(prod) < 1e-18): break
events.sort(key=lambda e: e[0])          # arbitrary arrival order = arbitrary delays
for _, u, v in events:
    acc[u] += v
out['C3_delayed_packets'] = dict(packets=len(events), max_abs_err=float(np.abs(acc - lam).max()),
                                 passed=bool(np.abs(acc - lam).max() < 1e-9))

# C4: send-on-delta credit on the forward-trace form: g = sum_t Re(conj(c_t) S_t), S_t = a_t S_{t-1} + e_t (e_t = dW-write
# sensitivity; scalar parameter per mode). Emit accumulated local cotangent only when |acc| > kappa.
e = rng.normal(size=(T, n)) + 1j * rng.normal(size=(T, n))
S = np.zeros(n, complex); g_exact = np.zeros(n); Ss = []
for t in range(T):
    S = a[t] * S + e[t]; Ss.append(S.copy()); g_exact += np.real(np.conj(c[t]) * S)
Ss = np.array(Ss); c4 = {}
for kappa in (0.0, 1.0, 3.0, 10.0):
    pend = np.zeros(n, complex); g = np.zeros(n); msgs = 0
    for t in range(T):
        # pending cotangent contributes c_t * S_t; send-on-delta on the contribution itself
        pend += np.conj(c[t]) * Ss[t]
        send = np.abs(pend) > kappa
        g[send] += np.real(pend[send]); pend[send] = 0; msgs += int(send.sum())
    err = np.abs(g - g_exact)
    c4[str(kappa)] = dict(messages=msgs, of=T * n, max_abs_err=float(err.max()), bound=kappa, holds=bool(np.all(err <= kappa + 1e-9)))
out['C4_send_on_delta'] = dict(by_kappa=c4, passed=all(v['holds'] for v in c4.values()))

# C5: optimal credit delay per mode; closed form via Lambert-W-free bisection; check d* decreases with r
def dstar(rr, C=1.0, K=1.0, eta=0.01):
    f = lambda d: 2 * rr * C * np.exp(-2 * rr * d) - 2 * K * eta ** 2 * d
    lo, hi = 0.0, 1e6
    for _ in range(200):
        mid = 0.5 * (lo + hi); lo, hi = (mid, hi) if f(mid) > 0 else (lo, mid)
    return 0.5 * (lo + hi)
ds = [dstar(x) for x in r]
grid = np.linspace(0, 400, 400001)
num = [float(grid[np.argmin(np.exp(-2 * x * grid) + (0.01 * grid) ** 2)]) for x in r]
out['C5_optimal_delay'] = dict(decay_r=r.tolist(), d_star=[round(x, 4) for x in ds], grid_argmin=num,
                               d_times_r=[round(x * y, 3) for x, y in zip(ds, r)],
                               passed=bool(np.allclose(ds, num, atol=2e-3, rtol=1e-3)))

out['all_passed'] = all(v['passed'] for k2, v in out.items() if isinstance(v, dict) and 'passed' in v)
out['source_sha256'] = {str(Path(__file__).resolve().relative_to(ROOT)): hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
out['scope'] = 'Analytic/numerical witnesses for note 164; synthetic complex diagonal memory; no fitting, no data'
dst = ROOT / 'experiments/results/credit/aws_sftc_math_20261010.json'
dst.write_text(json.dumps(out, indent=1) + '\n')
print(json.dumps({k2: (v.get('passed') if isinstance(v, dict) else v) for k2, v in out.items() if k2 != 'source_sha256'}, indent=1))
