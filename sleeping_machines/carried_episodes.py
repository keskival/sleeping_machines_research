"""Carried-state event episodes: compiled_logits' outer loop with the native state passed in and returned.

Truncated credit over long streams (PROTOCOL_ASYMMETRY_AUDIT.md item 2; FAS runs of ~1,060 events): a long sequence is processed as consecutive segments; each segment starts from the previous segment's final
state (memories, write times, written flags, last context and its arrival times), detached so credit stays inside the
segment.  The forward computation is exactly the full-sequence computation: with the same layer function and
deterministic races, segmenting with carried state reproduces the unsegmented logits (tests/test_carried_episodes.py).
Timestamps are absolute and continue across segments.  compiled_episodes.py is unchanged (its hashes are pinned by
AWS manifests); this module reuses its layer step and transport.
"""
import torch
from torch.nn import functional as F

from .compiled_episodes import compiled_step
from .parallel_stream_language import precise_rotate


def initial_state(model, n):
    D, H, U, P = model.depth, model.heads, model.pool, model.payload
    dtype = model.embedding.weight.dtype
    return dict(mem=[torch.zeros(n, H, U, P, dtype=dtype) for _ in range(D)],
                arr=[torch.zeros(n, H, U, dtype=torch.float64) for _ in range(D)],
                seen=[torch.zeros(n, H, U, dtype=torch.bool) for _ in range(D)],
                ctx_vals=torch.zeros(n, H * P, dtype=dtype), ctx_arr=torch.zeros(n, H, dtype=torch.float64),
                has_ctx=torch.zeros(n, dtype=torch.bool))


def detach(state):
    return {k: ([x.detach() for x in v] if isinstance(v, list) else v.detach()) for k, v in state.items()}


def carried_logits(model, stamps, marks, state=None, seed=0, step=None, route_credit=None, deterministic=False):
    """stamps (n, T) float64 absolute times; marks (n, T, content).  Every lane has T events in this segment.
    Returns (logits (n, T, classes), final state; not detached: call detach() between segments for truncated credit)."""
    step = step or compiled_step()
    source = 0
    layers = model._stacked(source)
    D, H, U, P = model.depth, model.heads, model.pool, model.payload
    n, T = stamps.shape
    st = state or initial_state(model, n)
    mem, arr, seen = list(st['mem']), list(st['arr']), list(st['seen'])
    ctx_vals, ctx_arr, has_ctx = st['ctx_vals'], st['ctx_arr'], st['has_ctx']
    active = torch.ones(n, dtype=torch.bool)
    every = []

    def transport(value, age, depth, head):
        age = age.clamp_min(0)
        rate = F.softplus(model.transport_rate[depth, head]) + 1e-6
        decayed = value * torch.exp(-age.to(value.dtype)[:, None] * rate).repeat_interleave(2, -1)
        return precise_rotate(decayed, age[:, None] * model.transport_frequency[depth, head].to(torch.float64))

    with torch.random.fork_rng():
        torch.manual_seed(seed)
        for k in range(T):
            x = model.embedding.weight[source][None] + model.content(marks[:, k])
            arrival = stamps[:, k]
            read_time = torch.where(has_ctx, torch.maximum(arrival, ctx_arr.max(-1).values), arrival)
            arrival = read_time
            context = torch.cat([transport(ctx_vals[:, h * P:(h + 1) * P], read_time - ctx_arr[:, h], D - 1, h)
                                 for h in range(H)], -1)
            x = torch.where(has_ctx[:, None], F.layer_norm(x + torch.sigmoid(model.source_gate(x)) * context,
                                                             (model.total_payload,)), x)
            for depth in range(D):
                Lp = layers[depth]
                noise = torch.stack([torch.empty(U, dtype=torch.float64).exponential_() for _ in range(H)])
                if deterministic:
                    noise = torch.ones_like(noise)
                mix = model.channel_mix[depth]
                x, arrival, mem[depth], arr[depth], seen[depth], values, arrivals = step(
                    x, arrival, mem[depth], arr[depth], seen[depth], active, noise, mix.weight, mix.bias, Lp['query'],
                    Lp['key'], Lp['key_read'], Lp['clock_bias'], Lp['control_w'], Lp['control_b'], Lp['rate'],
                    Lp['frequency'], Lp['input'], Lp['output'], Lp['gate_w'], Lp['gate_b'], Lp['gain'],
                    model.transport_rate[depth], model.transport_frequency[depth],
                    route_credit in ('linear', 'linear_rw', 'linear_rwn'), route_credit in ('linear_rw', 'linear_rwn'),
                    route_credit == 'linear_rwn')
            ctx_vals = values.reshape(n, H * P); ctx_arr = arrivals; has_ctx = torch.ones_like(has_ctx)
            every.append(model.head(x))
    return torch.stack(every, 1), dict(mem=mem, arr=arr, seen=seen, ctx_vals=ctx_vals, ctx_arr=ctx_arr, has_ctx=has_ctx)
