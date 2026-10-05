"""Recruitment knobs for the race layer (THEORY §419): a fixed score bonus for never-written slots, a training-time
score temperature, and the race probabilities exposed for an exposure (load-balance) penalty.  Also k-winner writes
(write_k: the k earliest arrivals per head all write; deliver_k: deliver the mean of their proposals, §428).

recruit_layer has compiled_episodes.layer_step's arguments plus free_bias and temperature, and returns layer_step's
seven outputs plus pi (n, H, U).  With free_bias = 0 and temperature = 1 it equals layer_step exactly
(tests/test_recruit_layer.py).  The bonus is part of the model's policy (applied in training and evaluation); the
temperature is an exploration knob for training only (evaluate at 1).  compiled_episodes.py is unchanged (pinned).
"""
import math

import torch
from torch.nn import functional as F

from .compiled_episodes import _rotate, _transport


def recruit_layer(x, arrival, m, arr_d, seen_d, active, noise, mix_w, mix_b, query, key, key_read, clock_bias, control_w,
                  control_b, rate, frequency, input_w, output_w, gate_w, gate_b, gain, transport_rate, transport_frequency,
                  linear_credit=False, write_credit=False, written_only=False, free_bias=0., temperature=1., stale_bias=0.,
                  write_k=1, deliver_k=False):
    n = x.shape[0]
    H, U, P = m.shape[1], m.shape[2], m.shape[3]
    total = H * P
    mixed = F.linear(x, mix_w, mix_b)
    all_features = F.layer_norm(mixed, (total,))
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
    if free_bias:                                  # optimistic prior: a never-written slot overwrites nothing (§419)
        scores = scores + free_bias * (~seen_d).to(scores.dtype)
    if stale_bias:                                 # graded optionality: content already lost to decay costs nothing to replace
        staleness = torch.where(seen_d, 1 - decay.detach().mean(-1), torch.ones_like(scores))
        scores = scores + stale_bias * staleness.to(scores.dtype)
    if temperature != 1.:
        scores = scores / temperature
    m_new = _rotate(m * decay, age[..., None] * frequency.view(H, U, P // 2))
    written = write[..., None] * torch.einsum('hupq,lhuq->lhup', input_w.view(H, U, P, P), x_u)
    m_new = m_new + written
    y = F.layer_norm(torch.einsum('hupq,lhuq->lhup', output_w.view(H, U, P, P), m_new) + x_u, (P,))
    gate = torch.einsum('hupq,lhuq->lhup', gate_w.view(H, U, P, P), F.gelu(y)) + gate_b.view(H, U, P)
    proposals = x_u + gain * y * torch.sigmoid(gate)
    s64 = scores.to(torch.float64)
    times = noise[None] / s64.exp()
    first, winner = times.min(-1)
    first = first.detach()
    lse = torch.logsumexp(s64, -1)
    first_s = first - first * (lse - lse.detach())
    delay = .001 + .010 * first_s / (1 + first_s)
    if write_k > 1:                                # k earliest arrivals all write (write bandwidth; FINDINGS 5 Oct)
        kth = times.topk(write_k, -1, largest=False).indices                    # (n, H, k), first = winner
        onehot = F.one_hot(kth, U).sum(-2).to(torch.bool) & active[:, None, None]
    else:
        onehot = F.one_hot(winner, U).to(torch.bool) & active[:, None, None]
    if write_k > 1 and deliver_k:                  # deliver the mean of the k earliest proposals (k-winner SDM, §428)
        values = torch.gather(proposals, 2, kth[..., None].expand(n, H, write_k, P)).mean(2)
    else:
        values = torch.gather(proposals, 2, winner[:, :, None, None].expand(n, H, 1, P)).squeeze(2)
    pi = torch.softmax(s64, -1).to(proposals.dtype)
    if linear_credit:
        values = values + ((pi - pi.detach())[..., None] * proposals.detach()).sum(-2)
    new_mem = torch.where(onehot[..., None], m_new, m)
    if write_credit:
        delta = written if written_only else m_new - m
        new_mem = new_mem + (pi - pi.detach())[..., None] * delta.detach() * active[:, None, None, None].to(m.dtype)
    new_arr = torch.where(onehot, arrival[:, None, None], arr_d)
    new_seen = seen_d | onehot
    arrivals = arrival[:, None] + delay
    arrival_out = arrivals.max(-1).values
    age_h = arrival_out[:, None] - arrivals
    x_out = torch.cat([_transport(values[:, h], age_h[:, h], transport_rate[h], transport_frequency[h]) for h in range(H)], -1)
    return x_out, arrival_out, new_mem, new_arr, new_seen, values, arrivals, pi


_COMPILED = {}


def compiled_recruit_layer():
    if 'step' not in _COMPILED:
        from torch._dynamo import config as dynamo_config
        for name in ('cache_size_limit', 'recompile_limit'):
            if hasattr(dynamo_config, name):
                setattr(dynamo_config, name, max(getattr(dynamo_config, name), 64))
        _COMPILED['step'] = torch.compile(recruit_layer, dynamic=False, fullgraph=True)
    return _COMPILED['step']


def balance_penalty(pis):
    """U * sum_u (mean pi_u)^2 per (layer, head), averaged: 1 for uniform use, U for collapse onto one slot.
    pis: list over (event, layer) of (n, H, U)."""
    by_layer = {}
    for depth, p in pis:
        by_layer.setdefault(depth, []).append(p)
    vals = []
    for ps in by_layer.values():
        mean = torch.stack(ps).mean((0, 1))                 # (H, U)
        vals.append(mean.shape[-1] * (mean ** 2).sum(-1).mean())
    return torch.stack(vals).mean()
