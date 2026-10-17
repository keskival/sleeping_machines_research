#!/usr/bin/env python3
"""Native level-1 state for the race language members (theory 173 §6.3): persistent addressed slots with sparse event writes,
per-slot learned clocks and a race read. Replaces the dense complex-diagonal linear recurrence (ParallelTemporalMemory) as the
level-1 carrier when CERL(level1='slots').

Each token event t:
- WRITE race: scores its write query against S slot write-keys; the top-m slots receive the event's value v_t with magnitude
  g_t * sigmoid(score) (g_t = event strength). Every other slot is untouched: sparse addressed state updates, and capacity
  beyond activity (S slots, m written per event).
- CLOCKS: slot s decays at its own learned rate r_s per token between events (multi-timescale, exact exponential decay), so
  state_s(t) = sum_{tau <= t} exp(-r_s (t - tau)) w_{tau,s} v_tau.
- READ race: the query races over the S slots (read keys, separate from write keys) and an abstain option. A slot is read as
  its evidence-weighted mean, state_s(t) / (eps + mass_s(t)) with mass_s(t) = sum_{tau <= t} exp(-r_s (t - tau)) w_{tau,s}: bounded at
  any lane length (theory 173 C8; a raw sum grows to 1/r_s writes, about 1,000 for slow slots on 8,192-token lanes vs 255 in
  training). The read value is sum_s a_{t,s} state_s(t) / (eps + mass_s(t)).
- Counterfactual write credit (training only): E random slots receive probe writes of magnitude p - sg(p), zero in the forward
  pass; the gradient is the first-order value of writing the event there (theory 173 C7).

Execution: chunked and exact. Within a chunk the decay factorises, exp(-r (i - j)) = exp(-r i) exp(r j), so the intra-chunk read
is [R, S] @ [S, R]; the slot state [S, d] is carried between chunks and across lanes (persistent memory). Work per token is
O(S dk + S d) for the dense reference execution here; the event-driven execution touches m slots per write.
"""
import math
import torch
import torch.nn as nn


class SlotMemory(nn.Module):
    CHUNK = 64
    MAX_RATE = 1.0                                                                                # per-token decay cap: keeps exp(r j) finite within a chunk of 64

    EPS = 1.0                                                                                     # below one unit of evidence a slot reads as a sum, above it as a mean

    def __init__(self, d, slots=1024, write_m=4, dk=64, explore=0):
        super().__init__(); self.S, self.m, self.dk, self.E = slots, write_m, dk, explore
        self.wq, self.rq = nn.Linear(d, dk), nn.Linear(d, dk)
        self.wkey = nn.Parameter(torch.randn(slots, dk)/math.sqrt(dk)); self.rkey = nn.Parameter(torch.randn(slots, dk)/math.sqrt(dk))
        self.null = nn.Parameter(torch.tensor(0.0)); self.val = nn.Linear(d, d); self.gate = nn.Linear(d, 1)
        rates = torch.logspace(-3, 0, slots)[torch.randperm(slots, generator=torch.Generator().manual_seed(0))]
        self.log_rate = nn.Parameter(rates.log())                                                 # clocks from 1,000-token to 1-token time constants

    def rate(self):
        return self.log_rate.exp().clamp(max=self.MAX_RATE)

    def writes(self, u):
        """Sparse write magnitudes w [B, L, S] (m nonzeros per event, plus zero-valued probes in training)."""
        g = torch.sigmoid(self.gate(u)); sw = (self.wq(u) @ self.wkey.T)/math.sqrt(self.dk)
        top = sw.topk(self.m, -1); w = torch.zeros_like(sw).scatter(-1, top.indices, g*torch.sigmoid(top.values))
        if self.training and self.E:
            ex = torch.randint(0, self.S, (*sw.shape[:-1], self.E), device=u.device); p = g*torch.sigmoid(sw.gather(-1, ex))
            valid = ~(ex[..., :, None] == top.indices[..., None, :]).any(-1)
            w = w.scatter_add(-1, ex, (p - p.detach())*valid.to(p.dtype))
        return w

    def forward(self, u, state0=None):
        if torch.is_autocast_enabled(u.device.type):                                             # fp32 under autocast: chunk factors span e^{+-63} and the mass is a long cumsum
            with torch.autocast(u.device.type, enabled=False):
                return self.forward(u.float(), None if state0 is None else tuple(t_.float() for t_ in state0))
        B, L, d = u.shape; S = self.S; dv = u.device
        w = self.writes(u); v = self.val(u)
        a = torch.softmax(torch.cat([(self.rq(u) @ self.rkey.T)/math.sqrt(self.dk), self.null.expand(B, L, 1)], -1), -1)[..., :S]
        r = self.rate(); st, ms = (u.new_zeros(B, S, d), u.new_zeros(B, S)) if state0 is None else state0; out = []
        for c0 in range(0, L, self.CHUNK):
            c1 = min(c0 + self.CHUNK, L); R = c1 - c0; i = torch.arange(R, device=dv, dtype=u.dtype)
            ac, wc, vc = a[:, c0:c1], w[:, c0:c1], v[:, c0:c1]
            mass = torch.exp(-r[None, :]*i[:, None])*torch.cumsum(wc*torch.exp(r[None, :]*i[:, None]), 1) + torch.exp(-r[None, :]*(i[:, None] + 1))*ms[:, None, :]   # [B, R, S]
            ac = ac/(self.EPS + mass)                                                              # evidence-weighted mean read
            left = ac*torch.exp(-r[None, :]*i[:, None]); right = wc*torch.exp(r[None, :]*i[:, None])
            coef = (left @ right.transpose(1, 2))*(i[:, None] >= i[None, :]).to(u.dtype)          # [B, R, R]: sum_s a_is w_js e^{-r_s (i - j)}, j <= i
            out.append(coef @ vc + (ac*torch.exp(-r[None, :]*(i[:, None] + 1))) @ st)                # + carried state decayed to step i
            st = torch.exp(-r*R)[None, :, None]*st + (wc*torch.exp(-r[None, :]*(R - 1 - i[:, None]))).transpose(1, 2) @ vc; ms = mass[:, -1]
        self.last_state = (st, ms)
        return torch.cat(out, 1)


def contracts():
    """Chunked execution equals the explicit slot sum; causality; carried state equals one long lane; probes change no forward value."""
    torch.manual_seed(0); res = {}; d, S, L = 8, 16, 40
    m = SlotMemory(d, S, 3, 8, explore=2); m.CHUNK = 7
    with torch.no_grad(): m.log_rate.uniform_(-3, 0.5)                                            # some rates hit the cap
    u = torch.randn(2, L, d); m.eval()
    with torch.no_grad():
        y = m(u); w = m.writes(u); v = m.val(u); r = m.rate()
        a = torch.softmax(torch.cat([(m.rq(u) @ m.rkey.T)/math.sqrt(m.dk), m.null.expand(2, L, 1)], -1), -1)[..., :S]
        t = torch.arange(L, dtype=u.dtype); lag = t[:, None] - t[None, :]
        K = torch.exp(-r[None, None, :]*lag.clamp_min(0)[..., None])*(lag >= 0)[..., None].to(u.dtype)   # [t, tau, S]
        state = torch.einsum('tus,bus,bud->btsd', K, w, v); mass = torch.einsum('tus,bus->bts', K, w); ref = torch.einsum('bts,btsd->btd', a/(m.EPS + mass), state)
        res['chunked_equals_explicit'] = float((y - ref).abs().max())
        u2 = u.clone(); u2[:, 25] += 1.0; res['causal_prefix_max_diff'] = float((m(u2)[:, :25] - y[:, :25]).abs().max())
        y1 = m(u[:, :21]); s1 = m.last_state; y2 = m(u[:, 21:], s1)
        res['carry_equals_long_lane'] = float((torch.cat([y1, y2], 1) - y).abs().max())
        res['write_density'] = float((w != 0).float().mean()); res['expected_density'] = 3/S
        ul = torch.randn(1, 2000, d); yl = m(ul); res['long_lane_read_bounded'] = float(yl.abs().max()/(m.val(ul).abs().max() + 1e-9))   # a mean of values: <= max |value|
    m.train(); torch.manual_seed(3); yt = m(u); m.E = 0; torch.manual_seed(3); y0 = m(u); m.E = 2
    res['probes_forward_unchanged'] = float((yt - y0).abs().max())
    yt.sum().backward(); g_e = m.wkey.grad.clone(); m.zero_grad(); y0.sum().backward(); res['probe_credit_to_write_keys'] = float((g_e - m.wkey.grad).abs().max())
    ok = res['chunked_equals_explicit'] < 1e-4 and res['causal_prefix_max_diff'] == 0 and res['carry_equals_long_lane'] < 1e-4 and res['probes_forward_unchanged'] < 1e-6 and res['probe_credit_to_write_keys'] > 0 \
        and abs(res['write_density'] - res['expected_density']) < 1e-6 and res['long_lane_read_bounded'] <= 1.0 + 1e-5
    res['status'] = 'passed' if ok else 'failed'; return res


if __name__ == '__main__':
    import json
    print(json.dumps(contracts(), indent=1))
