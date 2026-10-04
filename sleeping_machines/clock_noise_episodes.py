"""Isolated native clock-noise prototype; existing kernels/defaults untouched.

The eager wrapper reuses the exact native event program with a private globals
dictionary. The explicit layer port supplies an independent formulation and a
compile target. No shared module globals, weights, scores or race draws change.
Numerical native admission is required before any quality claim.
"""
from functools import partial
import math
from types import FunctionType

import torch
from torch.nn import functional as F

from . import batched_episodes as native
from .compiled_episodes import _rotate, _transport, compiled_logits, layer_step
from .clock_noise_law import check_temperature


def reshape_first(first, scores, temperature):
    if temperature == 1.:
        return first
    logz = torch.logsumexp(scores.to(torch.float64), -1)
    if temperature == 0.:
        return (-logz).exp()
    logw = first.clamp_min(torch.finfo(torch.float64).tiny).log() + logz
    return (temperature * logw - math.lgamma(1 + temperature) - logz).exp()


class ClockRace(torch.autograd.Function):
    @staticmethod
    def forward(ctx, scores, proposals, noise, forced_alt, temperature):
        rates = scores.to(torch.float64).exp()
        first, winner = (noise[None, :] / rates).min(-1)
        first = reshape_first(first, scores, temperature)
        winner = torch.where(forced_alt >= 0, forced_alt, winner)
        ctx.save_for_backward(rates, first, winner)
        ctx.shape = proposals.shape
        return proposals[torch.arange(len(winner)), winner], .001 + .010 * first / (1 + first), winner

    @staticmethod
    def backward(ctx, error_value, error_delay, error_winner):
        rates, first, winner = ctx.saved_tensors
        n, units, payload = ctx.shape
        value_credit = None
        if error_value is not None:
            value_credit = error_value.new_zeros(n, units, payload)
            value_credit[torch.arange(n), winner] = error_value
        credit = torch.zeros_like(rates)
        if error_delay is not None:
            pi = rates / rates.sum(-1, keepdim=True)
            credit = (error_delay * .010 / (1 + first).square())[:, None] * (-first[:, None] * pi)
        return credit.to(error_value.dtype if error_value is not None else rates.dtype), value_credit, None, None, None


def clock_batched_logits(model, rows, seed, temperature=1., **kwargs):
    temperature = check_temperature(temperature)
    if kwargs.get('deterministic'):
        raise ValueError('Clock-noise control retains sampled routes; greedy is a separate control')
    if kwargs.get('route_credit') not in (None, 'linear'):
        raise ValueError('Initial prototype admits native clock and local message credit only')
    if temperature == 1.:
        return native.batched_logits(model, rows, seed, **kwargs)
    class Adapter:
        @staticmethod
        def apply(scores, proposals, noise, forced_alt):
            return ClockRace.apply(scores, proposals, noise, forced_alt, temperature)
    original = native.batched_logits
    isolated = FunctionType(original.__code__, dict(original.__globals__, LaneRace=Adapter),
                            original.__name__, original.__defaults__, original.__closure__)
    isolated.__kwdefaults__ = original.__kwdefaults__
    return isolated(model, rows, seed, **kwargs)


def clock_layer_step(x, arrival, m, arr_d, seen_d, active, noise, mix_w, mix_b, query, key, key_read, clock_bias, control_w,
                     control_b, rate, frequency, input_w, output_w, gate_w, gate_b, gain, transport_rate, transport_frequency,
                     linear_credit=False, write_credit=False, written_only=False, temperature=1.):
    """Explicit port of the native layer, changing only the common first clock."""
    n = x.shape[0]
    heads, units, payload = m.shape[1], m.shape[2], m.shape[3]
    total = heads * payload
    mixed = F.linear(x, mix_w, mix_b)
    incoming = mixed.view(n, heads, payload)
    q = torch.einsum('hpd,ld->lhp', query, F.layer_norm(mixed, (total,)))
    prev = torch.where(seen_d, arr_d, arrival[:, None, None])
    xu = incoming[:, :, None, :].expand(n, heads, units, payload)
    qu = q[:, :, None, :].expand(n, heads, units, payload)
    read = key.view(heads, units, payload) + torch.einsum('hupq,lhuq->lhup', key_read.view(heads, units, payload, payload), m)
    scores = ((qu * read).sum(-1) / math.sqrt(payload) + clock_bias.view(heads, units)).clamp(-12, 12)
    controls = torch.einsum('hucp,lhup->lhuc', control_w.view(heads, units, 2, payload), F.layer_norm(xu, (payload,))) + control_b.view(heads, units, 2)
    forget, write = F.softplus(controls[..., 0]) / math.log(2), 2 * torch.sigmoid(controls[..., 1])
    age = (arrival[:, None, None] - prev).clamp_min(0)
    decay = torch.exp(-age.to(m.dtype)[..., None] * rate.view(heads, units, payload // 2) * forget[..., None]).repeat_interleave(2, -1)
    new = _rotate(m * decay, age[..., None] * frequency.view(heads, units, payload // 2))
    written = write[..., None] * torch.einsum('hupq,lhuq->lhup', input_w.view(heads, units, payload, payload), xu)
    new = new + written
    y = F.layer_norm(torch.einsum('hupq,lhuq->lhup', output_w.view(heads, units, payload, payload), new) + xu, (payload,))
    gate = torch.einsum('hupq,lhuq->lhup', gate_w.view(heads, units, payload, payload), F.gelu(y)) + gate_b.view(heads, units, payload)
    proposals = xu + gain * y * torch.sigmoid(gate)
    s64 = scores.to(torch.float64)
    first, winner = (noise[None] / s64.exp()).min(-1)
    first = reshape_first(first, s64, temperature).detach()
    lse = torch.logsumexp(s64, -1)
    clock = first - first * (lse - lse.detach())
    delay = .001 + .010 * clock / (1 + clock)
    onehot = F.one_hot(winner, units).bool() & active[:, None, None]
    values = torch.gather(proposals, 2, winner[:, :, None, None].expand(n, heads, 1, payload)).squeeze(2)
    if linear_credit:
        pi = torch.softmax(s64, -1).to(proposals.dtype)
        values = values + ((pi - pi.detach())[..., None] * proposals.detach()).sum(-2)
    new_mem = torch.where(onehot[..., None], new, m)
    if write_credit:
        pi = torch.softmax(s64, -1).to(m.dtype)
        delta = written if written_only else new - m
        new_mem = new_mem + (pi - pi.detach())[..., None] * delta.detach() * active[:, None, None, None].to(m.dtype)
    new_arr = torch.where(onehot, arrival[:, None, None], arr_d)
    new_seen = seen_d | onehot
    arrivals = arrival[:, None] + delay
    arrival_out = arrivals.max(-1).values
    ages = arrival_out[:, None] - arrivals
    out = torch.cat([_transport(values[:, h], ages[:, h], transport_rate[h], transport_frequency[h]) for h in range(heads)], -1)
    return out, arrival_out, new_mem, new_arr, new_seen, values, arrivals


_COMPILED = {}


def clock_compiled_logits(model, rows, seed, temperature=1., compile_step=False, **kwargs):
    temperature = check_temperature(temperature)
    if kwargs.get('deterministic') or kwargs.get('route_credit') not in (None, 'linear'):
        raise ValueError('Retain sampled routes and admitted local message credit')
    step = partial(clock_layer_step, temperature=temperature)
    if compile_step:
        if temperature not in _COMPILED:
            from torch._dynamo import config as dynamo_config
            for name in ('cache_size_limit', 'recompile_limit'):
                if hasattr(dynamo_config, name):
                    setattr(dynamo_config, name, max(getattr(dynamo_config, name), 64))
            _COMPILED[temperature] = torch.compile(step, dynamic=False, fullgraph=True)
        step = _COMPILED[temperature]
    return compiled_logits(model, rows, seed, step=step, **kwargs)
