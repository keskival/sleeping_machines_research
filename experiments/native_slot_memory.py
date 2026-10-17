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


def pk_scores(q, k1, k2, idx):
    """Product-key scores of slots idx (slot = a * n + b): q[:h].k1[a] + q[h:].k2[b]."""
    h, n = q.shape[-1]//2, k1.shape[0]
    return (q[..., None, :h]*k1[idx//n]).sum(-1) + (q[..., None, h:]*k2[idx % n]).sum(-1)


def pk_topk(q, k1, k2, k):
    """Exact top-k over n*n product-key slots from 2n half-key scores: the k best sums lie in the k x k grid of per-half top-k."""
    h, n = q.shape[-1]//2, k1.shape[0]
    v1, i1 = (q[..., :h] @ k1.T).topk(k, -1); v2, i2 = (q[..., h:] @ k2.T).topk(k, -1)
    vs, g = (v1[..., :, None] + v2[..., None, :]).flatten(-2).topk(k, -1)
    return vs, i1.gather(-1, g//k)*n + i2.gather(-1, g % k)


class SlotMemory(nn.Module):
    CHUNK = 64
    MAX_RATE = 1.0                                                                                # per-token decay cap: keeps exp(r j) finite within a chunk of 64

    EPS = 1.0                                                                                     # below one unit of evidence a slot reads as a sum, above it as a mean

    def __init__(self, d, slots=1024, write_m=4, dk=64, explore=0, read_topk=0, product_keys=False):
        super().__init__(); self.S, self.m, self.dk, self.E, self.read_topk, self.pk = slots, write_m, dk, explore, read_topk, product_keys
        self.wq, self.rq = nn.Linear(d, dk), nn.Linear(d, dk)
        if product_keys:                                                                           # slot addressing at 2 sqrt(S) half-key scores per race (capacity beyond activity, sublinear discovery)
            n = int(round(math.sqrt(slots))); assert n*n == slots and read_topk > 0, 'product-key slots need a square count and a top-k read'; self.n = n
            self.wkey1, self.wkey2, self.rkey1, self.rkey2 = (nn.Parameter(torch.randn(n, dk//2)/math.sqrt(dk)) for _ in range(4))
        else:
            self.wkey = nn.Parameter(torch.randn(slots, dk)/math.sqrt(dk)); self.rkey = nn.Parameter(torch.randn(slots, dk)/math.sqrt(dk))
        self.null = nn.Parameter(torch.tensor(0.0)); self.val = nn.Linear(d, d); self.gate = nn.Linear(d, 1)
        rates = torch.logspace(-3, 0, slots)[torch.randperm(slots, generator=torch.Generator().manual_seed(0))]
        self.log_rate = nn.Parameter(rates.log())                                                 # clocks from 1,000-token to 1-token time constants

    def rate(self):
        return self.log_rate.exp().clamp(max=self.MAX_RATE)

    def writes(self, u):
        """Sparse write magnitudes w [B, L, S] (m nonzeros per event, plus zero-valued probes in training)."""
        g = torch.sigmoid(self.gate(u)); qw = self.wq(u); sc = 1/math.sqrt(self.dk)
        if self.pk: tv, ti = pk_topk(qw, self.wkey1, self.wkey2, self.m); tv = tv*sc; score = lambda ix: pk_scores(qw, self.wkey1, self.wkey2, ix)*sc
        else: sw = (qw @ self.wkey.T)*sc; tv, ti = sw.topk(self.m, -1); score = lambda ix: sw.gather(-1, ix)
        w = torch.zeros(*u.shape[:-1], self.S, device=u.device, dtype=g.dtype).scatter(-1, ti, g*torch.sigmoid(tv))
        if self.training and self.E:
            ex = torch.randint(0, self.S, (*u.shape[:-1], self.E), device=u.device); p = g*torch.sigmoid(score(ex))
            valid = ~(ex[..., :, None] == ti[..., None, :]).any(-1)
            w = w.scatter_add(-1, ex, (p - p.detach())*valid.to(p.dtype))
        return w

    def forward(self, u, state0=None):
        if torch.is_autocast_enabled(u.device.type):                                             # fp32 under autocast: chunk factors span e^{+-63} and the mass is a long cumsum
            with torch.autocast(u.device.type, enabled=False):
                return self.forward(u.float(), None if state0 is None else tuple(t_.float() for t_ in state0))
        B, L, d = u.shape; S = self.S; dv = u.device
        w = self.writes(u); v = self.val(u); a, probe = self.reads(u)
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
        self.last_state = (st, ms); out = torch.cat(out, 1)
        return out if probe is None else out - probe[..., None]*out.detach()                    # read probes: d/dp = value_j - read (first-order value of including slot j)

    def reads(self, u):
        """Read race weights a [B, L, S] (abstain option dropped after normalisation). read_topk > 0: a hard race among the top-k slots
        (k values delivered); training adds E probe slots with zero-forward weight p - sg(p), p = exp(score - logZ_topk), so unread
        slots learn whether reading them would help (theory 173 C7). Returns (a, probe weight sum or None)."""
        B, L, _ = u.shape; qr = self.rq(u); sc = 1/math.sqrt(self.dk); nul = self.null.expand(B, L, 1)
        if not self.pk: sr = (qr @ self.rkey.T)*sc
        if not self.read_topk or self.read_topk >= self.S:
            return torch.softmax(torch.cat([sr, nul], -1), -1)[..., :self.S], None
        if self.pk: tv, ti = pk_topk(qr, self.rkey1, self.rkey2, self.read_topk); tv = tv*sc; score = lambda ix: pk_scores(qr, self.rkey1, self.rkey2, ix)*sc
        else: tv, ti = sr.topk(self.read_topk, -1); score = lambda ix: sr.gather(-1, ix)
        lz = torch.logsumexp(torch.cat([tv, nul], -1), -1, keepdim=True)
        a = torch.zeros(B, L, self.S, device=u.device, dtype=tv.dtype).scatter(-1, ti, torch.exp(tv - lz))
        if not (self.training and self.E): return a, None
        ex = torch.randint(0, self.S, (B, L, self.E), device=u.device); valid = ~(ex[..., :, None] == ti[..., None, :]).any(-1)
        p = torch.exp(score(ex) - lz)*valid.to(tv.dtype); q = p - p.detach()
        return a.scatter_add(-1, ex, q), q.sum(-1)

    @torch.no_grad()
    def event_forward(self, u, state0=None):
        """Event-driven execution (inference): per token, write only the m chosen slots (lazy exact decay from each slot's last
        write time) and deliver only the read slots' values. Returns (reads, traffic): traffic counts state rows touched per token
        (writes + value reads) next to the dense chunked execution's S. Equals forward() in eval mode (contract)."""
        B, L, d = u.shape; S = self.S; r = self.rate(); w = self.writes(u); v = self.val(u); a, _ = self.reads(u)
        st, ms = (u.new_zeros(B, S, d), u.new_zeros(B, S)) if state0 is None else (state0[0].clone(), state0[1].clone())
        last = torch.full((B, S), -1.0, device=u.device, dtype=u.dtype); t0 = 0.0                 # state/mass are exact at time last (t0 = -1 for a carried state)
        out = []; touched = 0
        for t in range(L):
            for b in range(B):
                ws = w[b, t].nonzero().squeeze(-1)                                                  # the m written slots
                f = torch.exp(-r[ws]*(t - last[b, ws])); st[b, ws] = f[:, None]*st[b, ws] + w[b, t, ws, None]*v[b, t]; ms[b, ws] = f*ms[b, ws] + w[b, t, ws]; last[b, ws] = t
                rs = a[b, t].nonzero().squeeze(-1)                                                  # the read slots (all S for a dense read)
                g = torch.exp(-r[rs]*(t - last[b, rs])); out.append((a[b, t, rs, None]*g[:, None]*st[b, rs]/(self.EPS + g*ms[b, rs])[:, None]).sum(0))
                touched += len(ws) + len(rs)
        reads = torch.stack(out).view(L, B, d).transpose(0, 1)
        # Materialize lazy timestamps once per lane for canonical persistent carry.
        decay = torch.exp(-r[None, :]*(L - 1 - last))
        self.last_state = ((st*decay[..., None]).detach(), (ms*decay).detach())
        keys = 4*self.n if self.pk else 2*S                                                        # key scores per token (write race + read race)
        return reads, dict(state_rows_touched_per_token=touched/(B*L), carry_materialization_rows_per_lane=B*S, dense_chunked_rows_per_token=S, key_scores_per_token=keys, slots=S, written=self.m, read=self.read_topk or S)


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
        ye, tr = m.event_forward(u); res['event_driven_equals_chunked'] = float((ye - y).abs().max()); res['event_traffic_dense_read'] = tr
        mk = SlotMemory(d, S, 3, 8, explore=2, read_topk=4); mk.load_state_dict(m.state_dict()); mk.CHUNK = 7; mk.eval()
        yk = mk(u); yke, trk = mk.event_forward(u); res['sparse_read_event_equals_chunked'] = float((yke - yk).abs().max()); res['event_traffic_sparse_read'] = trk
    mp_ = SlotMemory(d, S, 3, 8, explore=2, read_topk=4, product_keys=True); mp_.CHUNK = 7; mp_.eval()
    with torch.no_grad():
        qq = torch.randn(2, 5, 8); full = torch.stack([pk_scores(qq, mp_.wkey1, mp_.wkey2, torch.full((2, 5, 1), s_, dtype=torch.long)).squeeze(-1) for s_ in range(S)], -1)
        vv, ii = pk_topk(qq, mp_.wkey1, mp_.wkey2, 3); res['pk_topk_exact'] = float((vv - full.topk(3, -1).values).abs().max())
        yp = mp_(u); ype, trp = mp_.event_forward(u); res['pk_event_equals_chunked'] = float((ype - yp).abs().max()); res['event_traffic_pk'] = trp
    mk.train(); torch.manual_seed(4); ykt = mk(u); mk.E = 0; torch.manual_seed(4); yk0 = mk(u); mk.E = 2
    res['read_probes_forward_unchanged'] = float((ykt - yk0).abs().max())
    ykt.sum().backward(); gk = mk.rkey.grad.clone(); mk.zero_grad(); yk0.sum().backward(); res['read_probe_credit_to_read_keys'] = float((gk - mk.rkey.grad).abs().max())
    m.train(); torch.manual_seed(3); yt = m(u); m.E = 0; torch.manual_seed(3); y0 = m(u); m.E = 2
    res['probes_forward_unchanged'] = float((yt - y0).abs().max())
    yt.sum().backward(); g_e = m.wkey.grad.clone(); m.zero_grad(); y0.sum().backward(); res['probe_credit_to_write_keys'] = float((g_e - m.wkey.grad).abs().max())
    ok = res['chunked_equals_explicit'] < 1e-4 and res['causal_prefix_max_diff'] == 0 and res['carry_equals_long_lane'] < 1e-4 and res['probes_forward_unchanged'] < 1e-6 and res['probe_credit_to_write_keys'] > 0 \
        and abs(res['write_density'] - res['expected_density']) < 1e-6 and res['long_lane_read_bounded'] <= 1.0 + 1e-5 and res['event_driven_equals_chunked'] < 1e-5 and res['sparse_read_event_equals_chunked'] < 1e-5 \
        and res['pk_topk_exact'] < 1e-5 and res['pk_event_equals_chunked'] < 1e-5 and res['read_probes_forward_unchanged'] < 1e-6 and res['read_probe_credit_to_read_keys'] > 0
    res['status'] = 'passed' if ok else 'failed'; return res


if __name__ == '__main__':
    import json
    print(json.dumps(contracts(), indent=1))
