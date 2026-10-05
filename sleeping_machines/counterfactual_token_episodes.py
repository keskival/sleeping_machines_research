"""Token lookup and persistent event computation; existing pinned kernels stay intact.

Lookup is algebraically identical to the original one-hot content projection.
Retains temporal transport, hard races, sparse writes, separate keys/values and
local counterfactual value credit. Optional intervention supports bounded future-write outcome credit.
"""
import torch
from torch.nn import functional as F

from .counterfactual_token_layer import counterfactual_layer
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


def token_features(model, tokens, state=None, generator=None, route_credit='linear',
                   deterministic=False, eos=None, free_bias=0., temperature=1., force_site=None, suppress_site=None):
    """CPU token IDs -> integrated event features, carried state and race probabilities.

    Numeric state persists; callers detach it at credit boundaries. The explicit
    generator makes stochastic chunk partitioning invariant. EOS resets only its
    lane before consuming the observed EOS. No target enters state or routing.
    All candidates are scored/proposed; this implementation charges that work.
    """
    if tokens.ndim != 2 or tokens.dtype != torch.long or tokens.shape[1] == 0:
        raise ValueError('Nonempty [lanes,time] int64 token IDs required')
    if model.embedding.weight.device.type != 'cpu':
        raise ValueError('This initial implementation is CPU-only')
    if generator is None and not deterministic:
        raise ValueError('An explicit persistent generator is required')
    if route_credit not in ('linear', 'none'):
        raise ValueError('Use linear local value credit or its diagnostic ablation')
    pis = []
    step = counterfactual_layer
    source = 0
    layers = model._stacked(source)
    D, H, U, P = model.depth, model.heads, model.pool, model.payload
    n, T = tokens.shape
    st = state or initial_state(model, n)
    mem, arr, seen = list(st['mem']), list(st['arr']), list(st['seen'])
    ctx_vals, ctx_arr, has_ctx = st['ctx_vals'], st['ctx_arr'], st['has_ctx']
    position = st.get('position', torch.zeros(n, dtype=torch.float64))
    active = torch.ones(n, dtype=torch.bool)
    every = []
    winners = []
    writes = torch.zeros(D, H, U, dtype=torch.long)

    def transport(value, age, depth, head):
        age = age.clamp_min(0)
        rate = F.softplus(model.transport_rate[depth, head]) + 1e-6
        decayed = value * torch.exp(-age.to(value.dtype)[:, None] * rate).repeat_interleave(2, -1)
        return precise_rotate(decayed, age[:, None] * model.transport_frequency[depth, head].to(torch.float64))

    for k in range(T):
        reset = tokens[:, k] == eos if eos is not None else torch.zeros(n, dtype=torch.bool)
        if bool(reset.any()):
            mem = [torch.where(reset[:, None, None, None], torch.zeros_like(v), v) for v in mem]
            arr = [torch.where(reset[:, None, None], torch.zeros_like(v), v) for v in arr]
            seen = [v & ~reset[:, None, None] for v in seen]
            ctx_vals = torch.where(reset[:, None], torch.zeros_like(ctx_vals), ctx_vals)
            ctx_arr = torch.where(reset[:, None], torch.zeros_like(ctx_arr), ctx_arr)
            has_ctx = has_ctx & ~reset
            position = torch.where(reset, torch.zeros_like(position), position)
        x = model.embedding.weight[source][None] + F.embedding(tokens[:, k], model.content.weight.T)
        arrival = position
        position = position + 1
        read_time = torch.where(has_ctx, torch.maximum(arrival, ctx_arr.max(-1).values), arrival)
        arrival = read_time
        context = torch.cat([transport(ctx_vals[:, h * P:(h + 1) * P], read_time - ctx_arr[:, h], D - 1, h)
                             for h in range(H)], -1)
        x = torch.where(has_ctx[:, None], F.layer_norm(x + torch.sigmoid(model.source_gate(x)) * context,
                                                         (model.total_payload,)), x)
        for depth in range(D):
            Lp = layers[depth]
            noise = torch.stack([torch.empty(U, dtype=torch.float64).exponential_(generator=generator) for _ in range(H)])
            if deterministic:
                noise = torch.ones_like(noise)
            forced = mask = None
            if force_site is not None and (k,depth) == force_site[:2]:
                head, choices = force_site[2:]
                if not 0 <= head < H or choices.shape != (n,) or bool(((choices<0)|(choices>=U)).any()):
                    raise ValueError('Valid per-lane counterfactual receiver choices required')
                forced = torch.full((n,H),-1,dtype=torch.long)
                forced[:,head] = choices
            if suppress_site is not None and (k,depth) == suppress_site[:2]:
                mask = torch.ones(H,dtype=x.dtype); mask[suppress_site[2]] = 0
            mix = model.channel_mix[depth]
            old_arr, old_seen = arr[depth], seen[depth]
            x, arrival, mem[depth], arr[depth], seen[depth], values, arrivals, pi, winner = step(
                x, arrival, mem[depth], arr[depth], seen[depth], active, noise, mix.weight, mix.bias, Lp['query'],
                Lp['key'], Lp['key_read'], Lp['clock_bias'], Lp['control_w'], Lp['control_b'], Lp['rate'],
                Lp['frequency'], Lp['input'], Lp['output'], Lp['gate_w'], Lp['gate_b'], Lp['gain'],
                model.transport_rate[depth], model.transport_frequency[depth],
                route_credit == 'linear', False, False, free_bias=free_bias, temperature=temperature, forced_winner=forced, credit_mask=mask)
            writes[depth] += ((arr[depth] != old_arr) | (seen[depth] & ~old_seen)).sum(0).detach()
            pis.append((depth, pi))
            winners.append(winner.detach())
        ctx_vals = values.reshape(n, H * P); ctx_arr = arrivals; has_ctx = torch.ones_like(has_ctx)
        every.append(x)
    state = dict(mem=mem, arr=arr, seen=seen, ctx_vals=ctx_vals, ctx_arr=ctx_arr, has_ctx=has_ctx, position=position, last_writes=writes, race_winners=torch.stack(winners).reshape(T,D,n,H))
    return torch.stack(every, 1), state, pis
