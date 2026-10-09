#!/usr/bin/env python3
"""Numerical checks for theory note 160 (forward/backward duality). Evaluation only, CPU, seconds.
(1) The adjoint (backward pass) of the decaying-rotating diagonal memory z_t = a_t z_{t-1} + b_t, a_t = exp((-r + i w) dt_t),
    is the SAME memory run backward in time with conjugate rotation: lam_t = c_t + conj(a_{t+1}) lam_{t+1}.
(2) Forward-mode eligibility traces S_t = a_t S_{t-1} + [t = s] give the same gradients (dual direction, no stored history).
(3) Race credit is posterior inference: for clocks with hazards h_i = exp(theta_i) k_i(t), d log-likelihood / d theta_i
    = rho_i(tau) - H_i(tau), rho_i = h_i / sum_j h_j the posterior probability that clock i caused the event at tau."""
import math
import torch
torch.set_default_dtype(torch.float64)


def check_adjoint_and_traces(seed=0, T=40, n=6):
    torch.manual_seed(seed)
    dt = torch.rand(T) * 2; r = torch.rand(n) * 0.5; w = torch.randn(n)
    b = torch.randn(T, n, dtype=torch.complex128, requires_grad=True); c = torch.randn(T, n, dtype=torch.complex128)
    a = torch.exp((-r + 1j * w)[None, :] * dt[:, None]); z = torch.zeros(n, dtype=torch.complex128); L = 0
    for t in range(T):
        z = a[t] * z + b[t]; L = L + (c[t].conj() * z).real.sum()
    L.backward(); auto = b.grad
    lam = torch.zeros(n, dtype=torch.complex128); adj = torch.zeros(T, n, dtype=torch.complex128)
    for t in reversed(range(T)):
        lam = c[t] + (a[t + 1].conj() * lam if t + 1 < T else 0); adj[t] = lam
    worst = 0.0
    for s in range(T):
        S = torch.zeros(n, dtype=torch.complex128); g = torch.zeros(n, dtype=torch.complex128)
        for t in range(T):
            S = a[t] * S + (1.0 if t == s else 0.0); g = g + c[t] * S.conj()
        worst = max(worst, float((g - auto[s]).abs().max()))
    return float((adj - auto).abs().max()), worst


def check_race_credit(seed=1, M=5, tau=0.7):
    torch.manual_seed(seed)
    theta = torch.randn(M, requires_grad=True); mu = torch.randn(M); s = torch.rand(M) + 0.3; tau = torch.tensor(tau)
    def k(t):
        ln = torch.exp(-0.5 * ((torch.log(t) - mu) / s) ** 2) / (t * s * math.sqrt(2 * math.pi))
        return torch.where(torch.arange(M) < 2, torch.ones(M), ln + 1e-3)
    grid = torch.linspace(1e-6, float(tau), 20001); Hb = torch.trapz(torch.stack([k(t) for t in grid]), grid, dim=0)
    h = torch.exp(theta) * k(tau); H = torch.exp(theta) * Hb
    (torch.log(h.sum()) - H.sum()).backward()
    return float((theta.grad - (h / h.sum() - H).detach()).abs().max())


if __name__ == '__main__':
    a, f = check_adjoint_and_traces(); r = check_race_credit()
    print(f'(1) adjoint = time-reversed conjugate memory: max |diff| vs autograd {a:.2e}')
    print(f'(2) forward-mode eligibility traces:          max |diff| vs autograd {f:.2e}')
    print(f'(3) race credit = responsibility - exposure:  max |diff| vs autograd {r:.2e}')
    assert a < 1e-12 and f < 1e-12 and r < 1e-12
    print('PASS')
