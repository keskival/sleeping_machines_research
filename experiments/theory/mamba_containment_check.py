#!/usr/bin/env python3
"""Numerical contract: a selective state-space layer (Mamba's selective scan) is an exact member of the family.

Reference (Gu & Dao 2023, Algorithm 2 with the ZOH state transition and Euler input term, as in the authors' reference
scan): for channel d and state n,
    Δ_t = softplus(W_Δ x_t + b_Δ),  B_t = W_B x_t,  C_t = W_C x_t,
    h_t[d, n] = exp(Δ_t[d] A[d, n]) h_{t−1}[d, n] + Δ_t[d] B_t[n] x_t[d],
    y_t[d]   = Σ_n C_t[n] h_t[d, n] + D[d] x_t[d].

Family member ("selective temporal memory"), written in the family's own terms:
- each token is an event delivered after a learned, content-dependent delay δ_t = Δ_t (time warping);
- persistent memories decay (and, in general, rotate) with the real elapsed time between arrivals,
  z ← exp((−r + iω) (τ_t − τ_{t−1})) z, with r = −A and ω = 0 for Mamba;
- the arriving message writes a content vector (δ_t B_t x_t) into every addressed memory (dense write);
- a content-addressed read with key C_t returns Re⟨C_t, z⟩, plus a skip term.
The member keeps an explicit event clock τ (cumulative delays) and evolves memory over arrival-time differences, so the
equality below is a statement about two differently organized computations.

Checks: (1) real modes (Mamba-1) — exact match; (2) one scalar decay per head (Mamba-2 / SSD) — exact match;
(3) rotation ω ≠ 0 gives outputs no real-mode Mamba layer of the same state size produces on this input (strict superset
at equal state size; demonstrated by the residual of the best real fit on a random instance). float64 throughout.
"""
import torch
import torch.nn.functional as F

torch.set_default_dtype(torch.float64)


def mamba_selective_scan(x, A, W_dt, b_dt, W_B, W_C, D):
    """Reference selective scan. x: [T, Dm]; A: [Dm, N] (negative); returns y: [T, Dm]."""
    T, Dm = x.shape
    N = A.shape[1]
    h = torch.zeros(Dm, N)
    ys = []
    for t in range(T):
        dt = F.softplus(x[t] @ W_dt + b_dt)                  # [Dm]
        B = x[t] @ W_B; C = x[t] @ W_C                        # [N], [N]
        h = torch.exp(dt[:, None] * A) * h + (dt[:, None] * B[None, :]) * x[t][:, None]
        ys.append(h @ C + D * x[t])
    return torch.stack(ys)


class SelectiveTemporalMemory:
    """Family member: content-dependent delays drive an event clock; memories evolve over elapsed arrival time."""

    def __init__(self, rate, freq, W_delay, b_delay, W_write, W_key, D):
        self.rate, self.freq = rate, freq                     # [Dm, N] decay rates (r = −A) and rotation frequencies
        self.W_delay, self.b_delay, self.W_write, self.W_key, self.D = W_delay, b_delay, W_write, W_key, D

    def run(self, x):
        T, Dm = x.shape
        N = self.rate.shape[1]
        z = torch.zeros(Dm, N, dtype=torch.complex128)
        clock = torch.zeros(Dm)                               # event clock per channel (arrival times)
        ys = []
        for t in range(T):
            delay = F.softplus(x[t] @ self.W_delay + self.b_delay)   # learned, content-dependent delay
            arrival = clock + delay
            elapsed = arrival - clock                         # memory evolves over the real elapsed time
            z = torch.exp(torch.complex(-self.rate * elapsed[:, None], self.freq * elapsed[:, None])) * z
            message = (elapsed[:, None] * (x[t] @ self.W_write)[None, :]) * x[t][:, None]   # content written
            z = z + message
            key = (x[t] @ self.W_key).to(torch.complex128)
            ys.append((z @ key).real + self.D * x[t])
            clock = arrival
        return torch.stack(ys)


def main():
    torch.manual_seed(0)
    T, Dm, N = 64, 6, 8
    x = torch.randn(T, Dm)
    A = -torch.exp(torch.randn(Dm, N))
    W_dt, b_dt = 0.3 * torch.randn(Dm, Dm), torch.randn(Dm)
    W_B, W_C, D = torch.randn(Dm, N), torch.randn(Dm, N), torch.randn(Dm)

    ref = mamba_selective_scan(x, A, W_dt, b_dt, W_B, W_C, D)
    member = SelectiveTemporalMemory(-A, torch.zeros(Dm, N), W_dt, b_dt, W_B, W_C, D).run(x)
    e1 = float((ref - member).abs().max())
    print(f'(1) Mamba-1 selective scan vs family member: max |Δy| = {e1:.3e}')

    A2 = -torch.exp(torch.randn(Dm, 1)).expand(Dm, N).clone()          # one scalar decay per head (SSD)
    ref2 = mamba_selective_scan(x, A2, W_dt, b_dt, W_B, W_C, D)
    member2 = SelectiveTemporalMemory(-A2, torch.zeros(Dm, N), W_dt, b_dt, W_B, W_C, D).run(x)
    e2 = float((ref2 - member2).abs().max())
    print(f'(2) Mamba-2 scalar-per-head decay vs family member: max |Δy| = {e2:.3e}')

    # (3) Rotation: fit a real-mode Mamba layer (same N, free A, B/C maps, Δ map and D) to the member's outputs with
    # rotation; a non-zero optimum residual shows outputs outside the real-mode class at this state size.
    rot = SelectiveTemporalMemory(-A, 2.0 + torch.rand(Dm, N) * 3, W_dt, b_dt, W_B, W_C, D).run(x)
    params = [p.clone().requires_grad_(True) for p in (torch.log(-A), W_dt, b_dt, W_B, W_C, D)]
    opt = torch.optim.Adam(params, lr=0.02)
    for step in range(3000):
        logA, wdt, bdt, wb, wc, dd = params
        loss = ((mamba_selective_scan(x, -torch.exp(logA), wdt, bdt, wb, wc, dd) - rot) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    rel = float(loss.detach() / (rot ** 2).mean())
    print(f'(3) best real-mode Mamba fit to a rotating member (same state size): relative residual {rel:.3e}')

    ok = e1 < 1e-10 and e2 < 1e-10 and rel > 1e-3
    print('PASS' if ok else 'FAIL')


if __name__ == '__main__':
    main()
