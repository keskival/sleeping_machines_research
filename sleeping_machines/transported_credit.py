"""Transported write credit (THEORY §430, corrected Corollary 430.4): first-order choice credit including write
consequences, with two zero-valued shadow channels per slot.

Race scores read *stored* slot memory (key_read · m); proposals read *transported* memory
m' = A(t - tau) m + w. A counterfactual write of slot j at time t therefore changes:
  - later value reads by A(s - t) w_j (semigroup; value channel sigma_v, transported continuously);
  - later score reads by m'_j - m_j, in stored coordinates (score channel sigma_s), until slot j is really written
    again at t2. From then on both worlds' stored memories differ by A(t2 - t) w_j, so sigma_s resets to the
    transported value channel.

Each race adds (pi - pi.detach()) times the detached differences to both channels. The forward is unchanged (the
channels are exactly zero), so backpropagation gives the race scores pi_u [g.(v_u - v_bar) + write terms], the full
first-order choice credit of Proposition 430.2 (corrected).

transported_layer has compiled_episodes.layer_step's arguments plus (sv, sv_t, ss). It returns layer_step's seven
outputs, the race scores, and the updated (sv, sv_t, ss). transported_logits mirrors carried_episodes.carried_logits
and returns (logits, state, recorded scores). Contracts: tests/test_transported_credit.py.
"""
import math

import torch
from torch.nn import functional as F

from .compiled_episodes import _rotate, _transport
from .parallel_stream_language import precise_rotate


def transported_layer(x, arrival, m, arr_d, seen_d, active, noise, mix_w, mix_b, query, key, key_read, clock_bias,
                      control_w, control_b, rate, frequency, input_w, output_w, gate_w, gate_b, gain, transport_rate,
                      transport_frequency, sv, sv_t, ss, linear_credit=True, write_credit=True):
    n = x.shape[0]
    H, U, P = m.shape[1], m.shape[2], m.shape[3]
    mixed = F.linear(x, mix_w, mix_b)
    all_features = F.layer_norm(mixed, (H * P,))
    incoming = mixed.view(n, H, P)
    q = torch.einsum('hpd,ld->lhp', query, all_features)
    prev = torch.where(seen_d, arr_d, arrival[:, None, None])
    x_u = incoming[:, :, None, :].expand(n, H, U, P)
    q_u = q[:, :, None, :].expand(n, H, U, P)
    read = key.view(H, U, P) + torch.einsum('hupq,lhuq->lhup', key_read.view(H, U, P, P), m + ss)   # score path: stored
    scores = ((q_u * read).sum(-1) / math.sqrt(P) + clock_bias.view(H, U)).clamp(-12, 12)
    controls = torch.einsum('hucp,lhup->lhuc', control_w.view(H, U, 2, P), F.layer_norm(x_u, (P,))) + control_b.view(H, U, 2)
    forget = F.softplus(controls[..., 0]) / math.log(2)
    write = 2 * torch.sigmoid(controls[..., 1])
    age = (arrival[:, None, None] - prev).clamp_min(0)
    decay = torch.exp(-age.to(m.dtype)[..., None] * rate.view(H, U, P // 2) * forget[..., None]).repeat_interleave(2, -1)
    m_new = _rotate(m * decay, age[..., None] * frequency.view(H, U, P // 2))
    written = write[..., None] * torch.einsum('hupq,lhuq->lhup', input_w.view(H, U, P, P), x_u)
    m_new = m_new + written
    age_v = (arrival[:, None, None] - sv_t).clamp_min(0)                     # value channel, transported to now
    decay_v = torch.exp(-age_v.to(m.dtype)[..., None] * rate.view(H, U, P // 2) * forget[..., None]).repeat_interleave(2, -1)
    sv_now = _rotate(sv * decay_v, age_v[..., None] * frequency.view(H, U, P // 2))
    y = F.layer_norm(torch.einsum('hupq,lhuq->lhup', output_w.view(H, U, P, P), m_new + sv_now) + x_u, (P,))
    gate = torch.einsum('hupq,lhuq->lhup', gate_w.view(H, U, P, P), F.gelu(y)) + gate_b.view(H, U, P)
    proposals = x_u + gain * y * torch.sigmoid(gate)
    s64 = scores.to(torch.float64)
    times = noise[None] / s64.exp()
    first, winner = times.min(-1)
    first = first.detach()
    lse = torch.logsumexp(s64, -1)
    first_s = first - first * (lse - lse.detach())
    delay = .001 + .010 * first_s / (1 + first_s)
    onehot = F.one_hot(winner, U).to(torch.bool) & active[:, None, None]
    values = torch.gather(proposals, 2, winner[:, :, None, None].expand(n, H, 1, P)).squeeze(2)
    pi = torch.softmax(s64, -1).to(proposals.dtype)
    dpi = (pi - pi.detach())[..., None]
    if linear_credit:
        values = values + (dpi * proposals.detach()).sum(-2)
    new_mem = torch.where(onehot[..., None], m_new, m)
    new_arr = torch.where(onehot, arrival[:, None, None], arr_d)
    new_seen = seen_d | onehot
    act = active[:, None, None, None].to(m.dtype)
    if write_credit:
        sv_out = sv_now + dpi * written.detach() * act
        ss_out = torch.where(onehot[..., None], sv_now, ss) + dpi * (m_new - m).detach() * act
    else:
        sv_out, ss_out = sv_now, torch.where(onehot[..., None], sv_now, ss)
    sv_t_out = arrival[:, None, None].expand_as(sv_t).to(sv_t.dtype)
    arrivals = arrival[:, None] + delay
    arrival_out = arrivals.max(-1).values
    age_h = arrival_out[:, None] - arrivals
    x_out = torch.cat([_transport(values[:, h], age_h[:, h], transport_rate[h], transport_frequency[h]) for h in range(H)], -1)
    return x_out, arrival_out, new_mem, new_arr, new_seen, values, arrivals, s64, sv_out, sv_t_out, ss_out


_COMPILED = {}


def compiled_transported_layer():
    if 'step' not in _COMPILED:
        from torch._dynamo import config as dynamo_config
        for name in ('cache_size_limit', 'recompile_limit'):
            if hasattr(dynamo_config, name):
                setattr(dynamo_config, name, max(getattr(dynamo_config, name), 64))
        _COMPILED['step'] = torch.compile(transported_layer, dynamic=False, fullgraph=True)
    return _COMPILED['step']


def transported_logits(model, stamps, marks, state=None, seed=0, write_credit=True, compiled=False, deterministic=False):
    """carried_episodes.carried_logits' loop with the transported layer. state additionally holds sv, sv_t, ss.
    Returns (logits (n, T, classes), final state (not detached), recorded race scores [(event, depth, (n, H, U))])."""
    from .carried_episodes import initial_state
    step = compiled_transported_layer() if compiled else transported_layer
    layers = model._stacked(0)
    D, H, U, P = model.depth, model.heads, model.pool, model.payload
    n, T = stamps.shape
    dtype = model.embedding.weight.dtype
    st = state or initial_state(model, n)
    mem, arr, seen = list(st['mem']), list(st['arr']), list(st['seen'])
    sv = list(st.get('sv') or [torch.zeros(n, H, U, P, dtype=dtype) for _ in range(D)])
    sv_t = list(st.get('sv_t') or [torch.zeros(n, H, U, dtype=torch.float64) for _ in range(D)])
    ss = list(st.get('ss') or [torch.zeros(n, H, U, P, dtype=dtype) for _ in range(D)])
    ctx_vals, ctx_arr, has_ctx = st['ctx_vals'], st['ctx_arr'], st['has_ctx']
    active = torch.ones(n, dtype=torch.bool)
    every, record = [], []

    def transport(value, age, depth, head):
        age = age.clamp_min(0)
        r = F.softplus(model.transport_rate[depth, head]) + 1e-6
        decayed = value * torch.exp(-age.to(value.dtype)[:, None] * r).repeat_interleave(2, -1)
        return precise_rotate(decayed, age[:, None] * model.transport_frequency[depth, head].to(torch.float64))

    with torch.random.fork_rng():
        torch.manual_seed(seed)
        for k in range(T):
            x = model.embedding.weight[0][None] + model.content(marks[:, k])
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
                (x, arrival, mem[depth], arr[depth], seen[depth], values, arrivals, s64, sv[depth], sv_t[depth],
                 ss[depth]) = step(x, arrival, mem[depth], arr[depth], seen[depth], active, noise, mix.weight, mix.bias,
                                   Lp['query'], Lp['key'], Lp['key_read'], Lp['clock_bias'], Lp['control_w'],
                                   Lp['control_b'], Lp['rate'], Lp['frequency'], Lp['input'], Lp['output'], Lp['gate_w'],
                                   Lp['gate_b'], Lp['gain'], model.transport_rate[depth],
                                   model.transport_frequency[depth], sv[depth], sv_t[depth], ss[depth], True, write_credit)
                record.append((k, depth, s64))
            ctx_vals = values.reshape(n, H * P); ctx_arr = arrivals; has_ctx = torch.ones_like(has_ctx)
            every.append(model.head(x))
    state = dict(mem=mem, arr=arr, seen=seen, ctx_vals=ctx_vals, ctx_arr=ctx_arr, has_ctx=has_ctx, sv=sv, sv_t=sv_t, ss=ss)
    return torch.stack(every, 1), state, record
