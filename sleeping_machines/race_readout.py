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
    def __init__(self, types, heads, pool, payload, total_payload, hidden=64, eps=1e-3, mu0=2.5, type_durations=False,
                 classes=0, context=True):
        """type_durations (THEORY §435.2): each slot's own-duration law is conditioned on the next type,
        f_s(tau | e), instead of one law f_s(tau) shared by all types.
        classes = M > 0 (THEORY §436.1): a mixture of M step classes, p_s(e, tau) = sum_c pi_c p_c(e) f_c(tau), which
        couples type and duration (a skipped step changes both) at M instead of V duration laws per slot.
        context=False drops the merged-stream context from the per-slot laws (THEORY §436.3: the context path lets every
        slot predict the merged distribution, which weakens the pressure to bind). context='additive' makes the per-slot
        hidden state depend on slot memory only (computable once per write) and adds a context term shared by all
        slots to the output logits (O(1) per event)."""
        super().__init__()
        self.types, self.H, self.U, self.P, self.eps = types, heads, pool, payload, eps
        self.type_durations, self.classes, self.context = type_durations, classes, context
        self.slot = nn.Parameter(torch.randn(heads, pool, hidden) * .1)
        self.mem_in = nn.Linear(payload, hidden)
        self.ctx_in = nn.Linear(total_payload, hidden, bias=False)
        V = types
        if classes:
            M = classes
            self.out = nn.Linear(hidden, M * V + 3 * M + 1)
            with torch.no_grad():
                self.out.bias.zero_()
                self.out.bias[M * V + M:M * V + 2 * M] = mu0 + torch.linspace(-.5, .5, M)   # spread class durations
        else:
            D = V if type_durations else 1
            self.out = nn.Linear(hidden, V + 2 * D + 1)
            with torch.no_grad():
                self.out.bias[V:V + D] = mu0             # log seconds: FAS item-own steps are ~5–76 s
                self.out.bias[V + D:] = 0.
        if context == 'additive':
            self.ctx_out = nn.Linear(total_payload, self.out.out_features, bias=False)
            nn.init.zeros_(self.ctx_out.weight)

    def forward(self, mem, x):
        """mem (n, H, U, P) top-layer memories; x (n, total_payload). Returns per-slot laws, slots flattened.
        mu and log_sigma are (n, S) or, with type_durations, (n, S, V)."""
        n = mem.shape[0]
        h = self.mem_in(F.layer_norm(mem, (self.P,))) + self.slot
        if self.context is True:
            h = h + self.ctx_in(x)[:, None, None]
        h = F.gelu(h)
        o = self.out(h)
        if self.context == 'additive':
            o = o + self.ctx_out(x)[:, None, None]
        o = o.reshape(n, self.H * self.U, -1).double()
        V = self.types
        if self.classes:
            M = self.classes
            return dict(logp=F.log_softmax(o[..., :M * V].reshape(n, -1, M, V), -1),
                        logpi=F.log_softmax(o[..., M * V:M * V + M], -1), mu=o[..., M * V + M:M * V + 2 * M],
                        log_sigma=o[..., M * V + 2 * M:M * V + 3 * M].clamp(-4, 3),
                        logq=F.logsigmoid(o[..., -1]), log1mq=F.logsigmoid(-o[..., -1]))
        D = V if self.type_durations else 1
        mu, ls = o[..., V:V + D], o[..., V + D:V + 2 * D].clamp(-4, 3)
        if not self.type_durations:
            mu, ls = mu[..., 0], ls[..., 0]
        return dict(logp=F.log_softmax(o[..., :V], -1), mu=mu, log_sigma=ls,
                    logq=F.logsigmoid(o[..., -1]), log1mq=F.logsigmoid(-o[..., -1]))


def _typed(p):
    return p['mu'].dim() == p['logq'].dim() + 1


def log_surv(p, tau, eps):
    """log S_s(tau) = log(1 - q_s F_s(tau)); with type-conditional durations F_s = sum_e p_s(e) F_s(tau | e);
    with step classes F_s = sum_c pi_c F_c(tau)."""
    if 'logpi' in p:
        z = (torch.log(tau.clamp_min(0) + eps)[..., None] - p['mu']) / p['log_sigma'].exp()
        return torch.logaddexp(p['log1mq'], p['logq'] + torch.logsumexp(p['logpi'] + torch.special.log_ndtr(-z), -1))
    if _typed(p):
        z = (torch.log(tau.clamp_min(0) + eps)[..., None] - p['mu']) / p['log_sigma'].exp()
        return torch.logaddexp(p['log1mq'], p['logq'] + torch.logsumexp(p['logp'] + torch.special.log_ndtr(-z), -1))
    z = (torch.log(tau.clamp_min(0) + eps) - p['mu']) / p['log_sigma'].exp()
    return torch.logaddexp(p['log1mq'], p['logq'] + torch.special.log_ndtr(-z))


def log_fire_density(p, tau, eps):
    """log q_s f_s(tau); with type-conditional durations, per type: log q_s p_s(e) f_s(tau | e), shape (..., V)."""
    if _typed(p):
        u = torch.log(tau.clamp_min(0) + eps)[..., None]
        z = (u - p['mu']) / p['log_sigma'].exp()
        return p['logq'][..., None] + p['logp'] - .5 * z ** 2 - .5 * math.log(2 * math.pi) - p['log_sigma'] - u
    u = torch.log(tau.clamp_min(0) + eps)
    z = (u - p['mu']) / p['log_sigma'].exp()
    return p['logq'] - .5 * z ** 2 - .5 * math.log(2 * math.pi) - p['log_sigma'] - u


def log_diff_ndtr(z1, z2):
    """log(Phi(z2) - Phi(z1)) for z2 >= z1, stable in both tails (lower-tail form when z1 < 0, upper-tail otherwise)."""
    a, b = torch.special.log_ndtr(z2), torch.special.log_ndtr(z1)
    lower = a + torch.log1p(-torch.exp((b - a).clamp(max=-1e-12)))
    c, d = torch.special.log_ndtr(-z1), torch.special.log_ndtr(-z2)
    upper = c + torch.log1p(-torch.exp((d - c).clamp(max=-1e-12)))
    return torch.where(z1 < 0, lower, upper)


def event_terms(p, t_ref, t_now, t_next, e_next, eps, cell=None, scale=0.):
    """p: laws after the previous event; t_ref (n, S) last-write times; t_now (n,) previous event time; t_next (n,),
    e_next (n,) long: the event. Returns (log-lik total (n,), time-only (n,), log responsibilities (n, S)).
    cell (seconds): the recording resolution (THEORY §439). The firing term uses the probability of the recorded
    cell [tau, tau + cell), log(S_s(tau) - S_s(tau + cell)) - log S_s(tau) - log cell, instead of the point density,
    so exact ties cannot earn unbounded likelihood. Per-second units are kept.
    scale (THEORY §440): the log slowdown s of the alternative in which every own duration is scaled by e^s. Durations
    enter as tau e^-s; point densities carry the Jacobian -s; the recorded cell is scaled with them (its probability is
    still divided by the unscaled cell). d/ds at s = 0 is the score statistic of a slowdown."""
    tau_now = (t_now[:, None] - t_ref).clamp_min(0); tau = (t_next[:, None] - t_ref).clamp_min(0)
    if scale:
        shrink = math.exp(-scale)
        tau_now, tau = tau_now * shrink, tau * shrink
        cell_s = cell * shrink if cell is not None else None
        return _event_terms_tau(p, tau_now, tau, e_next, eps, cell_s, cell, -scale if cell is None else 0.)
    return _event_terms_tau(p, tau_now, tau, e_next, eps, cell, cell, 0.)


def _event_terms_tau(p, tau_now, tau, e_next, eps, cell, cell_norm, jac):
    """event_terms on given own durations; cell: the (possibly scaled) cell width; cell_norm: the width the cell
    probability is divided by; jac: an additive log-Jacobian on the firing terms (point densities under scaling)."""
    ls = log_surv(p, tau, eps)
    base = (ls - log_surv(p, tau_now, eps)).sum(-1)
    if cell is not None:
        lc = math.log(cell_norm)
        z1 = lambda mu, ls_: (torch.log(tau + eps) - mu) / ls_.exp()
        z2 = lambda mu, ls_: (torch.log(tau + cell + eps) - mu) / ls_.exp()
        if 'logpi' in p:
            u1, u2 = torch.log(tau + eps)[..., None], torch.log(tau + cell + eps)[..., None]
            sig = p['log_sigma'].exp()
            comp = p['logq'][..., None] + p['logpi'] + log_diff_ndtr((u1 - p['mu']) / sig, (u2 - p['mu']) / sig)
            M = comp.shape[-1]
            pe = p['logp'].gather(-1, e_next[:, None, None, None].expand(-1, comp.shape[1], M, 1)).squeeze(-1)
            fire_t = torch.logsumexp(comp, -1) - ls - lc
            fire = torch.logsumexp(comp + pe, -1) - ls - lc
        elif _typed(p):
            u1, u2 = torch.log(tau + eps)[..., None], torch.log(tau + cell + eps)[..., None]
            sig = p['log_sigma'].exp()
            dens = p['logq'][..., None] + p['logp'] + log_diff_ndtr((u1 - p['mu']) / sig, (u2 - p['mu']) / sig)
            fire_t = torch.logsumexp(dens, -1) - ls - lc
            fire = dens.gather(-1, e_next[:, None, None].expand(-1, dens.shape[1], 1)).squeeze(-1) - ls - lc
        else:
            fire_t = p['logq'] + log_diff_ndtr(z1(p['mu'], p['log_sigma']), z2(p['mu'], p['log_sigma'])) - ls - lc
            fire = fire_t + p['logp'].gather(-1, e_next[:, None, None].expand(-1, fire_t.shape[1], 1)).squeeze(-1)
        lse = torch.logsumexp(fire, -1)
        return base + lse, base + torch.logsumexp(fire_t, -1), fire - lse[:, None]
    if 'logpi' in p:
        u = torch.log(tau.clamp_min(0) + eps)[..., None]
        z = (u - p['mu']) / p['log_sigma'].exp()
        comp = p['logq'][..., None] + p['logpi'] - .5 * z ** 2 - .5 * math.log(2 * math.pi) - p['log_sigma'] - u + jac   # (n,S,M)
        M = comp.shape[-1]
        pe = p['logp'].gather(-1, e_next[:, None, None, None].expand(-1, comp.shape[1], M, 1)).squeeze(-1)
        fire_t = torch.logsumexp(comp, -1) - ls
        fire = torch.logsumexp(comp + pe, -1) - ls
        lse = torch.logsumexp(fire, -1)
        return base + lse, base + torch.logsumexp(fire_t, -1), fire - lse[:, None]
    dens = log_fire_density(p, tau, eps) + jac
    if _typed(p):
        fire_t = torch.logsumexp(dens, -1) - ls
        fire = dens.gather(-1, e_next[:, None, None].expand(-1, dens.shape[1], 1)).squeeze(-1) - ls
    else:
        fire_t = dens - ls
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


def slot_reads(m, key, key_read):
    """read vectors key_s + key_read_s m_s for all slots (n, H, U, P): the cache sparse_layer_step keeps exact."""
    H, U, P = m.shape[1], m.shape[2], m.shape[3]
    return key.view(H, U, P) + torch.einsum('hupq,lhuq->lhup', key_read.view(H, U, P, P), m)


@torch.no_grad()
def sparse_layer_step(x, arrival, m, arr_d, seen_d, active, noise, mix_w, mix_b, query, key, key_read, clock_bias,
                      control_w, control_b, rate, frequency, input_w, output_w, gate_w, gate_b, gain, transport_rate,
                      transport_frequency, reads, route_scores=None):
    """Inference-only equivalent of routed_layer_step (THEORY §436 work note).
    - Scores use the cached reads (n, H, U, P). A slot's read changes only when the slot is written.
    - Controls, decay, write, output and gate are computed for the winning slot of each head only.
    - The winner's read is refreshed after its write.
    Returns routed_layer_step's seven outputs plus the updated reads. Per head and event: U*P (scores) plus ~4 P^2
    (winner) multiply-adds, instead of ~5 U P^2."""
    n = x.shape[0]
    H, U, P = m.shape[1], m.shape[2], m.shape[3]
    mixed = F.linear(x, mix_w, mix_b)
    all_features = F.layer_norm(mixed, (H * P,))
    incoming = mixed.view(n, H, P)
    q = torch.einsum('hpd,ld->lhp', query, all_features)
    scores = ((q[:, :, None, :] * reads).sum(-1) / math.sqrt(P) + clock_bias.view(H, U)).clamp(-12, 12)
    if route_scores is not None:
        scores = route_scores.clamp(-12, 12)
    s64 = scores.to(torch.float64)
    times = noise[None] / s64.exp()
    first, winner = times.min(-1)                                              # (n, H)
    delay = .001 + .010 * first / (1 + first)
    hu = torch.arange(H)[None] * U + winner                                    # flat slot index (n, H)
    pick = lambda t: t.view(H * U, *t.shape[1:])[hu] if t.dim() > 1 else t.view(H * U)[hu]
    m_w = torch.gather(m, 2, winner[:, :, None, None].expand(n, H, 1, P)).squeeze(2)        # (n, H, P)
    prev = torch.where(torch.gather(seen_d, 2, winner[..., None]).squeeze(2),
                       torch.gather(arr_d, 2, winner[..., None]).squeeze(2), arrival[:, None])
    xn = F.layer_norm(incoming, (P,))
    controls = torch.einsum('lhcp,lhp->lhc', pick(control_w.view(H * U, 2, P)), xn) + pick(control_b.view(H * U, 2))
    forget = F.softplus(controls[..., 0]) / math.log(2)
    write = 2 * torch.sigmoid(controls[..., 1])
    age = (arrival[:, None] - prev).clamp_min(0)
    decay = torch.exp(-age.to(m.dtype)[..., None] * pick(rate.view(H * U, P // 2)) * forget[..., None]).repeat_interleave(2, -1)
    m_new = _rotate(m_w * decay, age[..., None] * pick(frequency.view(H * U, P // 2)))
    m_new = m_new + write[..., None] * torch.einsum('lhpq,lhq->lhp', pick(input_w.view(H * U, P, P)), incoming)
    y = F.layer_norm(torch.einsum('lhpq,lhq->lhp', pick(output_w.view(H * U, P, P)), m_new) + incoming, (P,))
    gate = torch.einsum('lhpq,lhq->lhp', pick(gate_w.view(H * U, P, P)), F.gelu(y)) + pick(gate_b.view(H * U, P))
    values = incoming + gain * y * torch.sigmoid(gate)
    onehot = F.one_hot(winner, U).to(torch.bool) & active[:, None, None]
    new_mem = torch.where(onehot[..., None], m_new[:, :, None, :], m)
    new_read = pick(key.view(H * U, P)) + torch.einsum('lhpq,lhq->lhp', pick(key_read.view(H * U, P, P)), m_new)
    reads = torch.where(onehot[..., None], new_read[:, :, None, :], reads)
    new_arr = torch.where(onehot, arrival[:, None, None], arr_d)
    new_seen = seen_d | onehot
    arrivals = arrival[:, None] + delay
    arrival_out = arrivals.max(-1).values
    age_h = arrival_out[:, None] - arrivals
    x_out = torch.cat([_transport(values[:, h], age_h[:, h], transport_rate[h], transport_frequency[h]) for h in range(H)], -1)
    return x_out, arrival_out, new_mem, new_arr, new_seen, values, arrivals, reads


_COMPILED = {}


def compiled_routed_step():
    if 'step' not in _COMPILED:
        from torch._dynamo import config as dynamo_config
        for name in ('cache_size_limit', 'recompile_limit'):
            if hasattr(dynamo_config, name):
                setattr(dynamo_config, name, max(getattr(dynamo_config, name), 64))
        _COMPILED['step'] = torch.compile(routed_layer_step, dynamic=False, fullgraph=True)
    return _COMPILED['step']


class BindingMemory(nn.Module):
    """Dedicated binding capacity (THEORY §436): Ub slots above the deep race network, written once per event by the
    posterior race of the readout (one emitting head, so Theorem 434.1 holds exactly).
    The written slot s* is updated as m_s* <- exp(-(t - t_s*) * rate) * m_s* + W x, where x is the deep network's
    output for the event and rate is per dimension, initialized log-spaced to time constants 1..tau_max seconds.
    Per event this costs one O(P * total_payload) write plus the readout's O(Ub * P * hidden) scoring. Binding capacity
    is decoupled from the per-slot proposal computation of the deep layers."""

    def __init__(self, total_payload, slots, payload, tau_max=1000., gated=False, learned=False):
        super().__init__()
        self.slots, self.payload, self.gated, self.learned = slots, payload, gated, learned
        if learned:        # comparison arm (THEORY §437 T1): a query/key write race learned by linear write credit
            self.query = nn.Linear(total_payload, payload)
            self.key = nn.Parameter(torch.randn(slots, payload) * .1)
            self.key_read = nn.Linear(payload, payload, bias=False)
        self.inp = nn.Linear(total_payload, payload)
        tau = torch.logspace(0, math.log10(tau_max), payload)
        self.raw_rate = nn.Parameter(torch.expm1(1 / tau).log())
        if gated:          # per-dimension overwrite gate (THEORY §436.2): a slot can hold its process's latest state
            self.gate = nn.Linear(total_payload, payload)

    def learned_write(self, bmem, t_ref, bseen, x, now, noise):
        """Learned write race: scores q(x) . (key_s + key_read m_s) / sqrt(P); winner by the exponential race; a
        zero-valued linear write credit (pi - pi.detach()) * (m_new_s - m_s).detach() for every slot (§430, value
        channel) carries the readout's gradient to the scores."""
        n, U, P = bmem.shape
        scores = (self.query(x)[:, None] * (self.key[None] + self.key_read(bmem))).sum(-1) / math.sqrt(P)
        s64 = scores.clamp(-12, 12).to(torch.float64)
        slot = (noise / s64.exp()).argmin(-1)
        age = (now[:, None] - t_ref).clamp_min(0)
        kept = bmem * torch.exp(-age[..., None].to(bmem.dtype) * F.softplus(self.raw_rate))
        if self.gated:
            g = torch.sigmoid(self.gate(x))[:, None]
            cand = (1 - g) * kept + g * self.inp(x)[:, None]
        else:
            cand = kept + self.inp(x)[:, None]
        onehot = F.one_hot(slot, U).to(torch.bool)
        pi = torch.softmax(s64, -1).to(bmem.dtype)
        new = torch.where(onehot[..., None], cand, bmem) + (pi - pi.detach())[..., None] * (cand - bmem).detach()
        return new, torch.where(onehot, now[:, None], t_ref), bseen | onehot, slot

    def write(self, bmem, t_ref, bseen, slot, x, now):
        n = bmem.shape[0]
        onehot = F.one_hot(slot, self.slots).to(torch.bool)
        age = (now - torch.gather(t_ref, 1, slot[:, None]).squeeze(1)).clamp_min(0)
        old = torch.gather(bmem, 1, slot[:, None, None].expand(n, 1, self.payload)).squeeze(1)
        kept = old * torch.exp(-age[:, None].to(old.dtype) * F.softplus(self.raw_rate))
        if self.gated:
            g = torch.sigmoid(self.gate(x))
            new = (1 - g) * kept + g * self.inp(x)
        else:
            new = kept + self.inp(x)
        bmem = torch.where(onehot[..., None], new[:, None], bmem)
        return bmem, torch.where(onehot, now[:, None], t_ref), bseen | onehot


class PredictiveLayer(nn.Module):
    """A deep layer with predictive routing (THEORY §438). Its slots form a binding memory over the layer's input event
    stream. Each slot's race readout predicts the next input (type law, own-duration law, pending probability). An
    arriving input is written to the slot drawn by the posterior race of those predictions, so the route needs no
    learned credit. The layer contributes its own superposition log-likelihood (a local objective) and outputs
    LN(x + W [x, m_s*]) from the written slot's updated memory."""

    def __init__(self, types, total_payload, slots, payload, hidden=64, classes=0, tau_max=1000., gated=True):
        super().__init__()
        self.memory = BindingMemory(total_payload, slots, payload, tau_max=tau_max, gated=gated)
        self.readout = RaceReadout(types, 1, slots, payload, total_payload, hidden=hidden, classes=classes)
        self.out = nn.Linear(total_payload + payload, total_payload)
        self.total_payload = total_payload

    def step(self, st, x, now, etype, deterministic):
        """st: dict(bmem, t_ref, bseen, laws, t_prev) or None. Returns x_out, local log-lik (n,), valid (n,), new st."""
        n = x.shape[0]; U, P = self.memory.slots, self.memory.payload
        if st is None:
            st = dict(bmem=torch.zeros(n, U, P, dtype=x.dtype), t_ref=now[:, None].expand(n, U).clone(),
                      bseen=torch.zeros(n, U, dtype=torch.bool), laws=None, t_prev=None)
        if st['laws'] is not None:
            ll, _, logr = event_terms(st['laws'], st['t_ref'], st['t_prev'], now, etype, self.readout.eps)
            noise = torch.ones(n, U, dtype=torch.float64) if deterministic else torch.empty(n, U, dtype=torch.float64).exponential_()
            slot = (logr.detach() - noise.log()).argmax(-1); valid = torch.ones(n, dtype=torch.bool)
        else:
            ll = torch.zeros(n, dtype=torch.float64); slot = torch.zeros(n, dtype=torch.long)
            valid = torch.zeros(n, dtype=torch.bool)
        bmem, t_ref, bseen = self.memory.write(st['bmem'], st['t_ref'], st['bseen'], slot, x, now)
        t_ref = torch.where(~bseen, now[:, None], t_ref)
        m_s = torch.gather(bmem, 1, slot[:, None, None].expand(n, 1, P)).squeeze(1)
        x_out = F.layer_norm(x + self.out(torch.cat([x, m_s], -1)), (self.total_payload,))
        laws = self.readout(bmem[:, None], x)
        return x_out, ll, valid, dict(bmem=bmem, t_ref=t_ref, bseen=bseen, laws=laws, t_prev=now)


def readout_episode(model, readout, stamps, marks, types, state=None, seed=0, route_credit='linear', posterior=False,
                    deterministic=False, compiled=False, binding=None, record=None, sparse=False, skip_deep=False,
                    pred_layers=None, local=None, surprise_gate=None, gate_stats=None, cell=None, late=None):
    """carried_episodes.carried_logits' loop with the race readout.
    stamps (n, T) float64 absolute seconds; marks (n, T, content); types (n, T) long event ids.
    Returns ll_total (n, T), ll_time (n, T): the log-likelihood of event j of this segment given everything before it
    (zero, with valid False, for the first event of a stream), valid (n, T) bool, and the state (not detached).
    The state carries the top-layer write times and the previous event's laws, so segments chain exactly.
    binding: a BindingMemory. The readout then reads the binding slots (readout built with heads=1, pool=Ub). Each event
    is written to one binding slot drawn by the posterior race (argmax when deterministic); the first event of a stream
    takes slot 0. The deep layers keep their learned races (no override).
    pred_layers: optional list of PredictiveLayer applied after the deep race layers (or instead of them with
    skip_deep); their local log-likelihoods are appended to `local` (a list) per event as (ll (n,), valid (n,)).
    late: optional list; per event it receives the slowdown score T_k = d/ds log p_s(event | past) at s = 0 (THEORY §440),
    by a central difference (h = 0.05), detached, for evaluation (None for a stream's first event).
    cell: recording resolution in seconds for the cell likelihood (event_terms; THEORY §439); None = point density.
    surprise_gate: a log-likelihood threshold (nats). When the readout gave the arriving event a log-likelihood above it
    (a predictable event), that lane skips the deep layers: their state is kept and x is the layer input. The binding
    memory and readout still update (THEORY §437 I4). gate_stats (list) receives the skipped fraction per event.
    skip_deep: no deep race layers; the event embedding feeds the binding memory and readout directly (THEORY §438
    ablation: are learned-route layers needed when binding and readout are posterior/likelihood-driven?).
    sparse: inference only (no grad): deep layers run sparse_layer_step (winner-only computation, cached slot reads);
    outputs equal the dense path (tests/test_race_readout.py).
    record: optional list; per event it receives dict(slot=written binding slot (n,) or None, hazard=the rescaled
    interval ΔΛ = -base (n,) or None for a stream's first event), detached, for evaluation diagnostics."""
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
    bwrite = None
    if binding is not None:
        Ub = binding.slots
        bwrite = binding.write
        if compiled:
            bwrite = _COMPILED.setdefault(('bwrite', id(binding)), torch.compile(binding.write, dynamic=False, fullgraph=True))
        bmem = st.get('bmem'); bseen = st.get('bseen')
        if bmem is None:
            bmem = torch.zeros(n, Ub, binding.payload, dtype=model.embedding.weight.dtype)
            bseen = torch.zeros(n, Ub, dtype=torch.bool)
    active = torch.ones(n, dtype=torch.bool)
    lin = route_credit in ('linear', 'linear_rw', 'linear_rwn')
    ll_tot, ll_time, valid = [], [], []
    reads = None
    if sparse:
        reads = list(st.get('reads') or [slot_reads(mem[d], layers[d]['key'], layers[d]['key_read']) for d in range(D)])

    def transport(value, age, depth, head):
        age = age.clamp_min(0)
        r = F.softplus(model.transport_rate[depth, head]) + 1e-6
        decayed = value * torch.exp(-age.to(value.dtype)[:, None] * r).repeat_interleave(2, -1)
        return precise_rotate(decayed, age[:, None] * model.transport_frequency[depth, head].to(torch.float64))

    with torch.random.fork_rng():
        torch.manual_seed(seed)
        for k in range(T):
            now = stamps[:, k]
            route = None; bslot = None; hazard = None
            if prev is not None:
                lt, lti, logr = terms(prev, t_ref, t_prev, now, types[:, k], readout.eps, cell)
                if late is not None:
                    with torch.no_grad():
                        up = event_terms(prev, t_ref, t_prev, now, types[:, k], readout.eps, cell, .05)[0]
                        dn = event_terms(prev, t_ref, t_prev, now, types[:, k], readout.eps, cell, -.05)[0]
                    late.append((up - dn) / .1)
            elif late is not None:
                late.append(None)
            if prev is not None:
                if record is not None:
                    tau_now = (t_prev[:, None] - t_ref).clamp_min(0); tau = (now[:, None] - t_ref).clamp_min(0)
                    hazard = -(log_surv(prev, tau, readout.eps) - log_surv(prev, tau_now, readout.eps)).sum(-1).detach()
                ll_tot.append(lt); ll_time.append(lti); valid.append(torch.ones(n, dtype=torch.bool))
                if binding is not None and not binding.learned:   # exponential race over log responsibilities = a posterior sample
                    noise_b = torch.ones(n, Ub, dtype=torch.float64) if deterministic else torch.empty(n, Ub, dtype=torch.float64).exponential_()
                    bslot = (logr.detach() - noise_b.log()).argmax(-1)
                elif posterior:               # per-head posterior over that head's slots
                    route = torch.log_softmax(logr.view(n, H, U), -1).detach()
            else:
                z = torch.zeros(n, dtype=torch.float64)
                ll_tot.append(z); ll_time.append(z); valid.append(torch.zeros(n, dtype=torch.bool))
            gate = torch.zeros(n, dtype=torch.bool)
            if surprise_gate is not None and prev is not None:
                gate = lt.detach() > surprise_gate
            if gate_stats is not None:
                gate_stats.append(float(gate.float().mean()))
            x = model.embedding.weight[0][None] + model.content(marks[:, k])
            arrival = torch.where(has_ctx, torch.maximum(now, ctx_arr.max(-1).values), now)
            context = torch.cat([transport(ctx_vals[:, h * P:(h + 1) * P], arrival - ctx_arr[:, h], D - 1, h)
                                 for h in range(H)], -1)
            x = torch.where(has_ctx[:, None], F.layer_norm(x + torch.sigmoid(model.source_gate(x)) * context,
                                                             (model.total_payload,)), x)
            if gate.any():
                saved = ([t.clone() for t in mem], [t.clone() for t in arr], [t.clone() for t in seen],
                         ctx_vals, ctx_arr, x, arrival)
            for depth in (() if skip_deep else range(D)):
                Lp = layers[depth]
                noise = torch.stack([torch.empty(U, dtype=torch.float64).exponential_() for _ in range(H)])
                if deterministic:
                    noise = torch.ones_like(noise)
                mix = model.channel_mix[depth]
                old_arr, old_seen = arr[depth], seen[depth]
                common = (x, arrival, mem[depth], arr[depth], seen[depth], active, noise, mix.weight, mix.bias, Lp['query'],
                          Lp['key'], Lp['key_read'], Lp['clock_bias'], Lp['control_w'], Lp['control_b'], Lp['rate'],
                          Lp['frequency'], Lp['input'], Lp['output'], Lp['gate_w'], Lp['gate_b'], Lp['gain'],
                          model.transport_rate[depth], model.transport_frequency[depth])
                if sparse:
                    x, arrival, mem[depth], arr[depth], seen[depth], values, arrivals, reads[depth] = sparse_layer_step(
                        *common, reads[depth], route if depth == D - 1 else None)
                else:
                    x, arrival, mem[depth], arr[depth], seen[depth], values, arrivals = step(
                        *common, lin, route if depth == D - 1 else None)
                if depth == D - 1 and binding is None:
                    wrote = (arr[depth] != old_arr) | (seen[depth] & ~old_seen)
                    base = now[:, None].expand(n, H * U) if t_ref is None else t_ref
                    t_ref = torch.where(wrote.reshape(n, H * U), now[:, None], base)
            if not skip_deep:
                ctx_vals = values.reshape(n, H * P); ctx_arr = arrivals; has_ctx = torch.ones_like(has_ctx)
            if gate.any() and not skip_deep:          # gated lanes keep their deep state; x is the layer input
                g4, g3, g1 = gate[:, None, None, None], gate[:, None, None], gate[:, None]
                for dd in range(D):
                    mem[dd] = torch.where(g4, saved[0][dd], mem[dd]); arr[dd] = torch.where(g3, saved[1][dd], arr[dd])
                    seen[dd] = torch.where(g3, saved[2][dd], seen[dd])
                    if sparse:
                        reads[dd] = torch.where(g4, slot_reads(mem[dd], layers[dd]['key'], layers[dd]['key_read']), reads[dd])
                ctx_vals = torch.where(g1, saved[3], ctx_vals); ctx_arr = torch.where(g1, saved[4], ctx_arr)
                x = torch.where(g1, saved[5], x)
            if pred_layers:
                pst = list(st.get('pred') or [None] * len(pred_layers)) if k == 0 else pst
                for li, layer in enumerate(pred_layers):
                    x, lll, lv, pst[li] = layer.step(pst[li], x, now, types[:, k], deterministic)
                    if local is not None:
                        local.append((lll, lv))
            if binding is not None:
                if t_ref is None:
                    t_ref = now[:, None].expand(n, Ub).clone()
                if binding.learned:
                    noise_l = torch.ones(n, Ub, dtype=torch.float64) if deterministic else torch.empty(n, Ub, dtype=torch.float64).exponential_()
                    bmem, t_ref, bseen, bslot = binding.learned_write(bmem, t_ref, bseen, x, now, noise_l)
                else:
                    if bslot is None:
                        bslot = torch.zeros(n, dtype=torch.long)
                    bmem, t_ref, bseen = bwrite(bmem, t_ref, bseen, bslot, x, now)
                if record is not None:
                    record.append(dict(slot=bslot.detach(), hazard=hazard))
                t_ref = torch.where(~bseen, now[:, None], t_ref)        # never-written slots: births timed from now
                prev = read(bmem[:, None], x); t_prev = now
                continue
            if record is not None:
                record.append(dict(slot=None, hazard=hazard))
            unseen = ~seen[D - 1].reshape(n, H * U)                  # never-written slots: births timed from now
            t_ref = torch.where(unseen, now[:, None], t_ref)
            prev = read(mem[D - 1], x); t_prev = now
    state = dict(mem=mem, arr=arr, seen=seen, ctx_vals=ctx_vals, ctx_arr=ctx_arr, has_ctx=has_ctx,
                 t_ref=t_ref, laws=prev, t_prev=t_prev)
    if binding is not None:
        state.update(bmem=bmem, bseen=bseen)
    if pred_layers:
        state['pred'] = pst
    if sparse:
        state['reads'] = reads
    return torch.stack(ll_tot, 1), torch.stack(ll_time, 1), torch.stack(valid, 1), state


def detach_state(state):
    """detach every tensor in a (nested) state of dicts, lists and tensors; None stays None."""
    if state is None:
        return None
    if isinstance(state, dict):
        return {k: detach_state(v) for k, v in state.items()}
    if isinstance(state, (list, tuple)):
        return [detach_state(v) for v in state]
    return state.detach() if torch.is_tensor(state) else state
