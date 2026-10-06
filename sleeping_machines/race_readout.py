"""Competing-risks race readout and posterior-routed writes (THEORY note 155 §§432–433).

A merged anonymous event log is a superposition of timed processes: the next event is the earliest pending event of
the K processes (§432.1). The race readout makes that the model's likelihood.

After event k, each top-layer slot s (heads x pool) proposes, from its memory, the event context and a learned slot
embedding:
- a type distribution p_s(e);
- a log-normal own-duration law f_s for its next event, measured from t_s, the time of the slot's last write;
- a probability q_s that it has a pending event (defective law: S_s(tau) = 1 - q_s F_s(tau)).

For the next event (type e at time t'), with t the current time:
  log p(e, t') = sum_v [log S_v(t' - t_v) - log S_v(t - t_v)]
               + logsumexp_s [log q_s + log f_s(t' - t_s) + log p_s(e) - log S_s(t' - t_s)].
The survival terms of the slots that did not fire are silence-aware supervision.

The time-only likelihood drops log p_s(e). The type part is the difference. Gap densities are converted to the
driver's log-gap units by the Jacobian: + log(gap + log_eps).

The posterior responsibility of slot s, r_s ∝ q_s f_s p_s(e) / S_s(t' - t_s), is the probability that s's race produced
the event. With posterior=True, the top layer writes the event, per head, to a slot drawn by an exponential race over
log r (exactly a sample from the per-head posterior; argmax when deterministic). This is the filtering binding rule:
the event is written where it was predicted. The routing scores are detached, so binding needs no learned write
credit. The delivered value is that slot's proposal, as in the standard layer.

routed_layer_step is compiled_episodes.layer_step (pinned file, copied here unchanged) with an optional per-lane
route_scores (n, H, U) replacing the query/key scores.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F

from .compiled_episodes import _rotate, _transport, layer_step
from .parallel_stream_language import precise_rotate


class RaceReadout(nn.Module):
    def __init__(self, types, heads, pool, payload, total_payload, hidden=64, eps=1e-3, mu0=2.5):
        super().__init__()
        self.types, self.H, self.U, self.P, self.eps = types, heads, pool, payload, eps
        self.slot = nn.Parameter(torch.randn(heads, pool, hidden) * .1)
        self.mem_in = nn.Linear(payload, hidden)
        self.ctx_in = nn.Linear(total_payload, hidden, bias=False)
        self.out = nn.Linear(hidden, types + 3)
        with torch.no_grad():
            self.out.bias[types] = mu0                   # log seconds: FAS item-own steps are ~5–76 s
            self.out.bias[types + 1] = 0.
            self.out.bias[types + 2] = 0.

    def forward(self, mem, x):
        """mem (n, H, U, P) top-layer memories; x (n, total_payload). Returns per-slot laws, slots flattened."""
        n = mem.shape[0]
        h = F.gelu(self.mem_in(F.layer_norm(mem, (self.P,))) + self.ctx_in(x)[:, None, None] + self.slot)
        o = self.out(h).reshape(n, self.H * self.U, -1).double()
        V = self.types
        return dict(logp=F.log_softmax(o[..., :V], -1), mu=o[..., V], log_sigma=o[..., V + 1].clamp(-4, 3),
                    logq=F.logsigmoid(o[..., V + 2]), log1mq=F.logsigmoid(-o[..., V + 2]))


def log_surv(p, tau, eps):
    z = (torch.log(tau.clamp_min(0) + eps) - p['mu']) / p['log_sigma'].exp()
    return torch.logaddexp(p['log1mq'], p['logq'] + torch.special.log_ndtr(-z))


def log_fire_density(p, tau, eps):
    u = torch.log(tau.clamp_min(0) + eps)
    z = (u - p['mu']) / p['log_sigma'].exp()
    return p['logq'] - .5 * z ** 2 - .5 * math.log(2 * math.pi) - p['log_sigma'] - u


def event_terms(p, t_ref, t_now, t_next, e_next, eps):
    """p: laws after the previous event; t_ref (n, S) last-write times; t_now (n,) previous event time; t_next (n,),
    e_next (n,) long: the event. Returns (log-lik total (n,), time-only (n,), log responsibilities (n, S))."""
    tau_now = (t_now[:, None] - t_ref).clamp_min(0); tau = (t_next[:, None] - t_ref).clamp_min(0)
    ls = log_surv(p, tau, eps)
    base = (ls - log_surv(p, tau_now, eps)).sum(-1)
    fire_t = log_fire_density(p, tau, eps) - ls
    fire = fire_t + p['logp'].gather(-1, e_next[:, None, None].expand(-1, fire_t.shape[1], 1)).squeeze(-1)
    lse = torch.logsumexp(fire, -1)
    return base + lse, base + torch.logsumexp(fire_t, -1), fire - lse[:, None]


def routed_layer_step(x, arrival, m, arr_d, seen_d, active, noise, mix_w, mix_b, query, key, key_read, clock_bias, control_w,
                      control_b, rate, frequency, input_w, output_w, gate_w, gate_b, gain, transport_rate, transport_frequency,
                      linear_credit=False, route_scores=None):
    """layer_step with route_scores (n, H, U) replacing the query/key race scores when given."""
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
    if route_scores is not None:
        scores = route_scores.clamp(-12, 12)
    controls = torch.einsum('hucp,lhup->lhuc', control_w.view(H, U, 2, P), F.layer_norm(x_u, (P,))) + control_b.view(H, U, 2)
    forget = F.softplus(controls[..., 0]) / math.log(2)
    write = 2 * torch.sigmoid(controls[..., 1])
    age = (arrival[:, None, None] - prev).clamp_min(0)
    decay = torch.exp(-age.to(m.dtype)[..., None] * rate.view(H, U, P // 2) * forget[..., None]).repeat_interleave(2, -1)
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
    onehot = F.one_hot(winner, U).to(torch.bool) & active[:, None, None]
    values = torch.gather(proposals, 2, winner[:, :, None, None].expand(n, H, 1, P)).squeeze(2)
    if linear_credit and route_scores is None:
        pi = torch.softmax(s64, -1).to(proposals.dtype)
        values = values + ((pi - pi.detach())[..., None] * proposals.detach()).sum(-2)
    new_mem = torch.where(onehot[..., None], m_new, m)
    new_arr = torch.where(onehot, arrival[:, None, None], arr_d)
    new_seen = seen_d | onehot
    arrivals = arrival[:, None] + delay
    arrival_out = arrivals.max(-1).values
    age_h = arrival_out[:, None] - arrivals
    x_out = torch.cat([_transport(values[:, h], age_h[:, h], transport_rate[h], transport_frequency[h]) for h in range(H)], -1)
    return x_out, arrival_out, new_mem, new_arr, new_seen, values, arrivals


_COMPILED = {}


def compiled_routed_step():
    if 'step' not in _COMPILED:
        from torch._dynamo import config as dynamo_config
        for name in ('cache_size_limit', 'recompile_limit'):
            if hasattr(dynamo_config, name):
                setattr(dynamo_config, name, max(getattr(dynamo_config, name), 64))
        _COMPILED['step'] = torch.compile(routed_layer_step, dynamic=False, fullgraph=True)
    return _COMPILED['step']


def readout_episode(model, readout, stamps, marks, types, state=None, seed=0, route_credit='linear', posterior=False,
                    deterministic=False, compiled=False):
    """carried_episodes.carried_logits' loop with the race readout.
    stamps (n, T) float64 absolute seconds; marks (n, T, content); types (n, T) long event ids.
    Returns ll_total (n, T), ll_time (n, T): the log-likelihood of event j of this segment given everything before it
    (zero, with valid False, for the first event of a stream), valid (n, T) bool, and the state (not detached).
    The state carries the top-layer write times and the previous event's laws, so segments chain exactly."""
    from .carried_episodes import initial_state
    step = compiled_routed_step() if compiled else routed_layer_step
    terms = event_terms
    read = readout
    if compiled:
        if 'terms' not in _COMPILED:
            _COMPILED['terms'] = torch.compile(event_terms, dynamic=False, fullgraph=True)
        terms = _COMPILED['terms']
        read = _COMPILED.setdefault(('readout', id(readout)), torch.compile(readout, dynamic=False, fullgraph=True))
    layers = model._stacked(0)
    D, H, U, P = model.depth, model.heads, model.pool, model.payload
    n, T = stamps.shape
    st = state or initial_state(model, n)
    mem, arr, seen = list(st['mem']), list(st['arr']), list(st['seen'])
    ctx_vals, ctx_arr, has_ctx = st['ctx_vals'], st['ctx_arr'], st['has_ctx']
    t_ref = st.get('t_ref'); prev = st.get('laws'); t_prev = st.get('t_prev')
    active = torch.ones(n, dtype=torch.bool)
    lin = route_credit in ('linear', 'linear_rw', 'linear_rwn')
    ll_tot, ll_time, valid = [], [], []

    def transport(value, age, depth, head):
        age = age.clamp_min(0)
        r = F.softplus(model.transport_rate[depth, head]) + 1e-6
        decayed = value * torch.exp(-age.to(value.dtype)[:, None] * r).repeat_interleave(2, -1)
        return precise_rotate(decayed, age[:, None] * model.transport_frequency[depth, head].to(torch.float64))

    with torch.random.fork_rng():
        torch.manual_seed(seed)
        for k in range(T):
            now = stamps[:, k]
            route = None
            if prev is not None:
                lt, lti, logr = terms(prev, t_ref, t_prev, now, types[:, k], readout.eps)
                ll_tot.append(lt); ll_time.append(lti); valid.append(torch.ones(n, dtype=torch.bool))
                if posterior:                 # per-head posterior over that head's slots
                    route = torch.log_softmax(logr.view(n, H, U), -1).detach()
            else:
                z = torch.zeros(n, dtype=torch.float64)
                ll_tot.append(z); ll_time.append(z); valid.append(torch.zeros(n, dtype=torch.bool))
            x = model.embedding.weight[0][None] + model.content(marks[:, k])
            arrival = torch.where(has_ctx, torch.maximum(now, ctx_arr.max(-1).values), now)
            context = torch.cat([transport(ctx_vals[:, h * P:(h + 1) * P], arrival - ctx_arr[:, h], D - 1, h)
                                 for h in range(H)], -1)
            x = torch.where(has_ctx[:, None], F.layer_norm(x + torch.sigmoid(model.source_gate(x)) * context,
                                                             (model.total_payload,)), x)
            for depth in range(D):
                Lp = layers[depth]
                noise = torch.stack([torch.empty(U, dtype=torch.float64).exponential_() for _ in range(H)])
                if deterministic:
                    noise = torch.ones_like(noise)
                mix = model.channel_mix[depth]
                old_arr, old_seen = arr[depth], seen[depth]
                x, arrival, mem[depth], arr[depth], seen[depth], values, arrivals = step(
                    x, arrival, mem[depth], arr[depth], seen[depth], active, noise, mix.weight, mix.bias, Lp['query'],
                    Lp['key'], Lp['key_read'], Lp['clock_bias'], Lp['control_w'], Lp['control_b'], Lp['rate'],
                    Lp['frequency'], Lp['input'], Lp['output'], Lp['gate_w'], Lp['gate_b'], Lp['gain'],
                    model.transport_rate[depth], model.transport_frequency[depth], lin,
                    route if depth == D - 1 else None)
                if depth == D - 1:
                    wrote = (arr[depth] != old_arr) | (seen[depth] & ~old_seen)
                    base = now[:, None].expand(n, H * U) if t_ref is None else t_ref
                    t_ref = torch.where(wrote.reshape(n, H * U), now[:, None], base)
            ctx_vals = values.reshape(n, H * P); ctx_arr = arrivals; has_ctx = torch.ones_like(has_ctx)
            unseen = ~seen[D - 1].reshape(n, H * U)                  # never-written slots: births timed from now
            t_ref = torch.where(unseen, now[:, None], t_ref)
            prev = read(mem[D - 1], x); t_prev = now
    state = dict(mem=mem, arr=arr, seen=seen, ctx_vals=ctx_vals, ctx_arr=ctx_arr, has_ctx=has_ctx,
                 t_ref=t_ref, laws=prev, t_prev=t_prev)
    return torch.stack(ll_tot, 1), torch.stack(ll_time, 1), torch.stack(valid, 1), state


def detach_state(state):
    out = {}
    for k, v in state.items():
        if isinstance(v, list):
            out[k] = [x.detach() for x in v]
        elif isinstance(v, dict):
            out[k] = {a: b.detach() for a, b in v.items()}
        elif v is None:
            out[k] = None
        else:
            out[k] = v.detach()
    return out
