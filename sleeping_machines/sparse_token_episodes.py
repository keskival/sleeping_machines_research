"""Token lookup and persistent event computation; existing pinned kernels stay intact.

Lookup is algebraically identical to the original one-hot content projection.
Retains temporal transport, hard races, sparse writes, separate keys/values and
sampled local counterfactual value credit. It does not claim full future-write credit.
"""
import torch
from torch.nn import functional as F

from .sparse_token_layer import sparse_token_layer, compiled_sparse_token_layer
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


def token_features(model, tokens, state=None, generator=None, route_credit='sampled',
                   deterministic=False, eos=None, free_bias=0., temperature=1., compiled=False, alternative_generator=None):
    """CPU token IDs -> integrated event features, carried state and race probabilities.

    Numeric state persists; callers detach it at credit boundaries. The explicit
    generator makes stochastic chunk partitioning invariant. EOS resets only its
    lane before consuming the observed EOS. No target enters state or routing.
    All keys are scored; only winner/one alternative proposals are evaluated.
    """
    if tokens.ndim != 2 or tokens.dtype != torch.long or tokens.shape[1] == 0:
        raise ValueError('Nonempty [lanes,time] int64 token IDs required')
    if model.embedding.weight.device.type != 'cpu':
        raise ValueError('This initial implementation is CPU-only')
    if generator is None and not deterministic:
        raise ValueError('An explicit persistent generator is required')
    if route_credit not in ('sampled', 'none'):
        raise ValueError('Use sampled local value credit or its diagnostic ablation')
    if free_bias != 0. or temperature != 1.:
        raise ValueError('This first sparse kernel retains the base race policy')
    if route_credit == 'sampled' and alternative_generator is None:
        raise ValueError('Sampled alternatives need an independent persistent generator')
    pis = []
    step = compiled_sparse_token_layer() if compiled else sparse_token_layer
    source = 0
    layers = model._stacked(source)
    D, H, U, P = model.depth, model.heads, model.pool, model.payload
    n, T = tokens.shape
    st = state or initial_state(model, n)
    mem, arr, seen = list(st['mem']), list(st['arr']), list(st['seen'])
    ctx_vals, ctx_arr, has_ctx = st['ctx_vals'], st['ctx_arr'], st['has_ctx']
    # Rebuild differentiable cached keys from CURRENT parameters each chunk.
    # Persist memories, never cached projections across optimizer updates.
    reads = [Lp['key'].view(H,U,P) + torch.einsum('hupq,lhuq->lhup',
             Lp['key_read'].view(H,U,P,P), m) for Lp,m in zip(layers,mem)]
    position = st.get('position', torch.zeros(n, dtype=torch.float64))
    active = torch.ones(n, dtype=torch.bool)
    every = []
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
            reads = [torch.where(reset[:,None,None,None],Lp['key'].view(H,U,P),r) for Lp,r in zip(layers,reads)]
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
            alt_noise = (torch.empty(H,U,dtype=torch.float64).exponential_(generator=alternative_generator)
                         if route_credit == 'sampled' else noise)
            mix = model.channel_mix[depth]
            old_arr, old_seen = arr[depth], seen[depth]
            x, arrival, mem[depth], arr[depth], seen[depth], reads[depth], values, arrivals, pi = step(
                x, arrival, mem[depth], arr[depth], seen[depth], reads[depth], active, noise, alt_noise, mix.weight, mix.bias, Lp['query'],
                Lp['key'], Lp['key_read'], Lp['clock_bias'], Lp['control_w'], Lp['control_b'], Lp['rate'],
                Lp['frequency'], Lp['input'], Lp['output'], Lp['gate_w'], Lp['gate_b'], Lp['gain'],
                model.transport_rate[depth], model.transport_frequency[depth],
                route_credit == 'sampled')
            writes[depth] += ((arr[depth] != old_arr) | (seen[depth] & ~old_seen)).sum(0).detach()
            pis.append((depth, pi))
        ctx_vals = values.reshape(n, H * P); ctx_arr = arrivals; has_ctx = torch.ones_like(has_ctx)
        every.append(x)
    state = dict(mem=mem, arr=arr, seen=seen, ctx_vals=ctx_vals, ctx_arr=ctx_arr, has_ctx=has_ctx, position=position, last_writes=writes)
    return torch.stack(every, 1), state, pis
