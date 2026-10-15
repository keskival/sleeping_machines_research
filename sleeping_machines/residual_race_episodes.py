"""Residual race stack: each race stage ADDS its transported output to the stream.

Versioned copy of multidepth_counterfactual_episodes (all-depth readout). The only
change: x <- x + g_d * stage_d(x) when the core carries `residual_gain` (D, H*P);
without it the function is the original serial composition, bit for bit. Races,
delays, sparse writes, transport, context and counterfactual credit are unchanged.
The stream keeps the stage output arrival (fast bypass plus delayed correction).

Optional per-stage addressed reads (core carries `read_query`): after stage d, the
stream queries a persistent per-lane store of earlier depth-d stream states with a
sparse top-k race (learned age clock, document-boundary bias from the observed EOS
token, null route) and adds sum_j a_j W_v f_j. Old entries are detached (read-time
credit reaches query/value maps, clock, bias and null for any lag); entries written
inside the current credit window keep their writer gradient. W_v starts at zero, so
the read stack starts exactly at the residual stack.

Lookup is algebraically identical to the original one-hot content projection.
Retains temporal transport, hard races, sparse writes, separate keys/values and
sampled local value credit. Forced-write continuations expose actual suffix
consequences to the driver's bounded future-write teacher.
"""
import math
import torch
from torch.nn import functional as F

from .sparse_counterfactual_layer import sparse_counterfactual_layer, compiled_sparse_counterfactual_layer
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
                   deterministic=False, eos=None, free_bias=0., temperature=1., compiled=False, alternative_generator=None, force_site=None, suppress_site=None):
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
    if temperature <= 0:raise ValueError('Positive race temperature required')
    if route_credit == 'sampled' and alternative_generator is None:
        raise ValueError('Sampled alternatives need an independent persistent generator')
    pis = []
    residual = getattr(model, 'residual_gain', None)
    reader = getattr(model, 'read_query', None) is not None
    step = compiled_sparse_counterfactual_layer() if compiled else sparse_counterfactual_layer
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
    if reader:
        W = H * P; dk = model.read_query.shape[-1]
        store_k = list(st.get('store_k', [torch.zeros(n, 0, dk) for _ in range(D)]))
        store_f = list(st.get('store_f', [torch.zeros(n, 0, W) for _ in range(D)]))
        store_t = list(st.get('store_t', [torch.zeros(n, 0, dtype=torch.long) for _ in range(D)]))
        clock = st.get('store_clock', torch.zeros(n, dtype=torch.long)).clone()
        last_eos = st.get('store_eos', torch.full((n,), -1, dtype=torch.long)).clone()
        win_k = [[] for _ in range(D)]; win_f = [[] for _ in range(D)]; win_t = [[] for _ in range(D)]
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
            reads = [torch.where(reset[:,None,None,None],Lp['key'].view(H,U,P),r) for Lp,r in zip(layers,reads)]
            ctx_vals = torch.where(reset[:, None], torch.zeros_like(ctx_vals), ctx_vals)
            ctx_arr = torch.where(reset[:, None], torch.zeros_like(ctx_arr), ctx_arr)
            has_ctx = has_ctx & ~reset
            position = torch.where(reset, torch.zeros_like(position), position)
        if reader:
            last_eos = torch.where(tokens[:, k] == 50256, clock, last_eos)
        x = model.embedding.weight[source][None] + F.embedding(tokens[:, k], model.content.weight.T)
        arrival = position
        position = position + 1
        read_time = torch.where(has_ctx, torch.maximum(arrival, ctx_arr.max(-1).values), arrival)
        arrival = read_time
        context = torch.cat([transport(ctx_vals[:, h * P:(h + 1) * P], read_time - ctx_arr[:, h], D - 1, h)
                             for h in range(H)], -1)
        x = torch.where(has_ctx[:, None], F.layer_norm(x + torch.sigmoid(model.source_gate(x)) * context,
                                                         (model.total_payload,)), x)
        depth_snapshots=[]
        for depth in range(D):
            Lp = layers[depth]
            noise = torch.stack([torch.empty(U, dtype=torch.float64).exponential_(generator=generator) for _ in range(H)])
            if deterministic:
                noise = torch.ones_like(noise)
            alt_noise = (torch.empty(H,U,dtype=torch.float64).exponential_(generator=alternative_generator)
                         if route_credit == 'sampled' else noise)
            forced=mask=None
            if force_site is not None and (k,depth)==force_site[:2]:
                head,choices=force_site[2:]
                if not 0<=head<H or choices.shape!=(n,) or bool(((choices<0)|(choices>=U)).any()):
                    raise ValueError('Valid per-lane alternative receiver choices required')
                forced=torch.full((n,H),-1,dtype=torch.long);forced[:,head]=choices
            if suppress_site is not None and (k,depth)==suppress_site[:2]:
                mask=torch.ones(H,dtype=x.dtype);mask[suppress_site[2]]=0
            mix = model.channel_mix[depth]
            old_arr, old_seen = arr[depth], seen[depth]
            stream = x
            x, arrival, mem[depth], arr[depth], seen[depth], reads[depth], values, arrivals, pi, winner = step(
                x, arrival, mem[depth], arr[depth], seen[depth], reads[depth], active, noise, alt_noise, mix.weight, mix.bias, Lp['query'],
                Lp['key'], Lp['key_read'], Lp['clock_bias'], Lp['control_w'], Lp['control_b'], Lp['rate'],
                Lp['frequency'], Lp['input'], Lp['output'], Lp['gate_w'], Lp['gate_b'], Lp['gain'],
                model.transport_rate[depth], model.transport_frequency[depth],
                route_credit == 'sampled',forced_winner=forced,credit_mask=mask,free_bias=free_bias,temperature=temperature)
            if residual is not None: x = stream + residual[depth] * x
            if reader:
                f = F.layer_norm(x, (W,)); q = f @ model.read_query[depth]
                parts_s, parts_t = [], []
                M = store_k[depth].shape[1]
                if M: parts_s.append((store_k[depth] @ q[:, :, None]).squeeze(-1)); parts_t.append(store_t[depth])
                if win_k[depth]:
                    wk = torch.stack(win_k[depth], 1); parts_s.append((wk @ q[:, :, None]).squeeze(-1)); parts_t.append(torch.stack(win_t[depth], 1))
                if parts_s:
                    times = torch.cat(parts_t, 1); age = (clock[:, None] - times).clamp_min(0).to(f.dtype)
                    sc = torch.cat(parts_s, 1) / math.sqrt(dk) - F.softplus(model.read_age[depth]) * torch.log1p(age)
                    sc = sc + model.read_boundary[depth] * (times < last_eos[:, None]).to(f.dtype)
                    kk = min(model.read_topk, sc.shape[1]); top, idx = sc.topk(kk, -1)
                    wts = torch.softmax(torch.cat([top, model.read_null[depth].expand(n, 1)], -1), -1)[:, :-1]
                    sel = torch.zeros(n, kk, W, dtype=f.dtype)
                    if M: sel = torch.where((idx < M)[..., None], store_f[depth].gather(1, idx.clamp_max(M - 1)[..., None].expand(-1, -1, W)), sel)
                    if win_f[depth]:
                        wf = torch.stack(win_f[depth], 1)
                        sel = torch.where((idx >= M)[..., None], wf.gather(1, (idx - M).clamp(0, wf.shape[1] - 1)[..., None].expand(-1, -1, W)), sel)
                    x = x + ((wts[..., None] * sel).sum(1)) @ model.read_value[depth]
                win_k[depth].append(f @ model.read_key[depth]); win_f[depth].append(f); win_t[depth].append(clock.clone())
            writes[depth] += ((arr[depth] != old_arr) | (seen[depth] & ~old_seen)).sum(0).detach()
            pis.append((depth, pi))
            winners.append(winner.detach())
            depth_snapshots.append((x,arrival))
        ctx_vals = values.reshape(n, H * P); ctx_arr = arrivals; has_ctx = torch.ones_like(has_ctx)
        if reader: clock = clock + 1
        output=[]
        for depth,(value,issued_at) in enumerate(depth_snapshots):
            if depth==D-1:output.append(value)
            else:output.append(torch.cat([transport(value[:,h*P:(h+1)*P],arrival-issued_at,depth,h) for h in range(H)],-1))
        every.append(torch.cat(output,-1))
    state = dict(mem=mem, arr=arr, seen=seen, ctx_vals=ctx_vals, ctx_arr=ctx_arr, has_ctx=has_ctx, position=position, last_writes=writes,race_winners=torch.stack(winners).reshape(T,D,n,H))
    if reader:
        C = model.read_capacity
        state.update(store_k=[torch.cat([store_k[d], torch.stack(win_k[d], 1)], 1)[:, -C:] for d in range(D)],
                     store_f=[torch.cat([store_f[d], torch.stack(win_f[d], 1)], 1)[:, -C:] for d in range(D)],
                     store_t=[torch.cat([store_t[d], torch.stack(win_t[d], 1)], 1)[:, -C:] for d in range(D)],
                     store_clock=clock, store_eos=last_eos)
    return torch.stack(every, 1), state, pis
