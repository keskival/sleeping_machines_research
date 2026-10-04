"""Exact expected reception with hard selective writes: a deterministic member of the family (THEORY §418).

Each receiver head delivers the exact expectation of its race, sum_u pi_u v_u (the family's exact-aggregation reception
endpoint, report/model_family_design.md), with the race's mean common clock T = 1/Z (Z = sum exp(s_u)).  Persistent
memory writes stay hard and selective: the highest-score unit commits its proposal and timestamp (argmax write).  The
member is deterministic and smooth in its inputs, which suits precision regression where sampled winners and clock noise
inject output noise (§417).  Scores receive exact gradients through pi (soft-attention credit for the delivered value);
the write address is not credited beyond that.  It pays dense value delivery: every unit's proposal is computed and
pooled at inference as well as in training (all keys scored, all proposals computed; one write per head).

The outer event loop is compiled_episodes.compiled_logits (unmodified) with this layer function; ExpectedStepper
reproduces it one event at a time for autonomous rollouts.  Contracts: tests/test_expected_reception.py.
"""
import math

import torch
from torch.nn import functional as F

from .compiled_episodes import _rotate, _transport, compiled_logits
from .parallel_stream_language import precise_rotate


def expected_layer(x, arrival, m, arr_d, seen_d, active, noise, mix_w, mix_b, query, key, key_read, clock_bias, control_w,
                   control_b, rate, frequency, input_w, output_w, gate_w, gate_b, gain, transport_rate,
                   transport_frequency, linear_credit=False, write_credit=False, written_only=False):
    """Same interface as compiled_episodes.layer_step; noise and credit flags are ignored (deterministic member)."""
    n = x.shape[0]
    H, U, P = m.shape[1], m.shape[2], m.shape[3]
    mixed = F.linear(x, mix_w, mix_b)
    all_features = F.layer_norm(mixed, (H * P,))
    incoming = mixed.view(n, H, P)
    q = torch.einsum('hpd,ld->lhp', query, all_features)
    prev = torch.where(seen_d, arr_d, arrival[:, None, None])
    x_u = incoming[:, :, None, :].expand(n, H, U, P)
    q_u = q[:, :, None, :].expand(n, H, U, P)
    read = key.view(H, U, P) + torch.einsum('hupq,lhuq->lhup', key_read.view(H, U, P, P), m)
    scores = ((q_u * read).sum(-1) / math.sqrt(P) + clock_bias.view(H, U)).clamp(-12, 12)
    controls = torch.einsum('hucp,lhup->lhuc', control_w.view(H, U, 2, P), F.layer_norm(x_u, (P,))) + control_b.view(H, U, 2)
    forget = F.softplus(controls[..., 0]) / math.log(2)
    write = 2 * torch.sigmoid(controls[..., 1])
    age = (arrival[:, None, None] - prev).clamp_min(0)
    decay = torch.exp(-age.to(m.dtype)[..., None] * rate.view(H, U, P // 2) * forget[..., None]).repeat_interleave(2, -1)
    m_new = _rotate(m * decay, age[..., None] * frequency.view(H, U, P // 2))
    m_new = m_new + write[..., None] * torch.einsum('hupq,lhuq->lhup', input_w.view(H, U, P, P), x_u)
    y = F.layer_norm(torch.einsum('hupq,lhuq->lhup', output_w.view(H, U, P, P), m_new) + x_u, (P,))
    gate = torch.einsum('hupq,lhuq->lhup', gate_w.view(H, U, P, P), F.gelu(y)) + gate_b.view(H, U, P)
    proposals = x_u + gain * y * torch.sigmoid(gate)
    s64 = scores.to(torch.float64)
    pi = torch.softmax(s64, -1)                                               # (n, H, U)
    first = 1 / s64.exp().sum(-1)                                             # mean first time 1/Z (scores clamped to +-12)
    delay = .001 + .010 * first / (1 + first)
    values = (pi.to(proposals.dtype)[..., None] * proposals).sum(2)           # exact expected delivery
    winner = scores.argmax(-1)                                                # hard selective write
    onehot = F.one_hot(winner, U).to(torch.bool) & active[:, None, None]
    new_mem = torch.where(onehot[..., None], m_new, m)
    new_arr = torch.where(onehot, arrival[:, None, None], arr_d)
    new_seen = seen_d | onehot
    arrivals = arrival[:, None] + delay
    arrival_out = arrivals.max(-1).values
    age_h = arrival_out[:, None] - arrivals
    x_out = torch.cat([_transport(values[:, h], age_h[:, h], transport_rate[h], transport_frequency[h]) for h in range(H)], -1)
    return x_out, arrival_out, new_mem, new_arr, new_seen, values, arrivals


_COMPILED = {}


def compiled_expected_layer():
    if 'step' not in _COMPILED:
        from torch._dynamo import config as dynamo_config
        for name in ('cache_size_limit', 'recompile_limit'):
            if hasattr(dynamo_config, name):
                setattr(dynamo_config, name, max(getattr(dynamo_config, name), 64))
        _COMPILED['step'] = torch.compile(expected_layer, dynamic=False, fullgraph=True)
    return _COMPILED['step']


def expected_logits(model, rows, seed=0, all_logits=False, feedback=None, compiled=True, **_):
    """Training/evaluation path: compiled_logits' outer loop with the expected-reception layer."""
    return compiled_logits(model, rows, seed, all_logits=all_logits, feedback=feedback,
                           step=compiled_expected_layer() if compiled else expected_layer)


class ExpectedStepper:
    """One event at a time (lanes in parallel), identical to expected_logits' outer loop."""

    @torch.no_grad()
    def __init__(self, model, lanes):
        self.model, self.n = model, lanes
        self.layers = model._stacked(0)
        D, H, U, P = model.depth, model.heads, model.pool, model.payload
        dtype = model.embedding.weight.dtype
        self.mem = [torch.zeros(lanes, H, U, P, dtype=dtype) for _ in range(D)]
        self.arr = [torch.zeros(lanes, H, U, dtype=torch.float64) for _ in range(D)]
        self.seen = [torch.zeros(lanes, H, U, dtype=torch.bool) for _ in range(D)]
        self.ctx_vals = torch.zeros(lanes, H * P, dtype=dtype)
        self.ctx_arr = torch.zeros(lanes, H, dtype=torch.float64)
        self.has_ctx = torch.zeros(lanes, dtype=torch.bool)

    def _transport(self, value, age, depth, head):
        model = self.model
        age = age.clamp_min(0)
        rate = F.softplus(model.transport_rate[depth, head]) + 1e-6
        decayed = value * torch.exp(-age.to(value.dtype)[:, None] * rate).repeat_interleave(2, -1)
        return precise_rotate(decayed, age[:, None] * model.transport_frequency[depth, head].to(torch.float64))

    @torch.no_grad()
    def step(self, stamps, marks):
        model, n = self.model, self.n
        D, H, U, P = model.depth, model.heads, model.pool, model.payload
        dtype = model.embedding.weight.dtype
        active = torch.ones(n, dtype=torch.bool)
        x = model.embedding.weight[0][None] + model.content(marks.to(dtype))
        arrival = stamps.to(torch.float64)
        read_time = torch.where(self.has_ctx, torch.maximum(arrival, self.ctx_arr.max(-1).values), arrival)
        arrival = read_time
        context = torch.cat([self._transport(self.ctx_vals[:, h * P:(h + 1) * P], read_time - self.ctx_arr[:, h], D - 1, h)
                             for h in range(H)], -1)
        x = torch.where(self.has_ctx[:, None], F.layer_norm(x + torch.sigmoid(model.source_gate(x)) * context,
                                                              (model.total_payload,)), x)
        noise = torch.ones(H, U, dtype=torch.float64)
        for depth in range(D):
            Lp = self.layers[depth]
            mix = model.channel_mix[depth]
            x, arrival, self.mem[depth], self.arr[depth], self.seen[depth], values, arrivals = expected_layer(
                x, arrival, self.mem[depth], self.arr[depth], self.seen[depth], active, noise, mix.weight, mix.bias,
                Lp['query'], Lp['key'], Lp['key_read'], Lp['clock_bias'], Lp['control_w'], Lp['control_b'], Lp['rate'],
                Lp['frequency'], Lp['input'], Lp['output'], Lp['gate_w'], Lp['gate_b'], Lp['gain'],
                model.transport_rate[depth], model.transport_frequency[depth])
        self.ctx_vals = values.reshape(n, H * P)
        self.ctx_arr = arrivals
        self.has_ctx = torch.ones(n, dtype=torch.bool)
        return model.head(x)
