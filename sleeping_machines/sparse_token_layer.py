"""Token-compatible sparse layer: one factual and one optional sampled proposal.

Arithmetic and estimator are inherited from sparse_training.py; all score keys
are still charged. Adds probability diagnostics without extra proposal work.
"""
import math
import torch
from torch.nn import functional as F
from .sparse_training import _unit, _transport


def sparse_token_layer(x, arrival, m, arr_d, seen_d, reads_d, active, noise, alt_noise, mix_w, mix_b, query, key, key_read,
                      clock_bias, control_w, control_b, rate, frequency, input_w, output_w, gate_w, gate_b, gain,
                      transport_rate, transport_frequency, sampled_credit=False):
    n = x.shape[0]
    H, U, P = m.shape[1], m.shape[2], m.shape[3]
    mixed = F.linear(x, mix_w, mix_b)
    all_features = F.layer_norm(mixed, (H * P,))
    incoming = mixed.view(n, H, P)
    q = torch.einsum('hpd,ld->lhp', query, all_features)
    scores = ((q[:, :, None, :] * reads_d).sum(-1) / math.sqrt(P) + clock_bias.view(H, U)).clamp(-12, 12)
    s64 = scores.to(torch.float64)
    times = noise[None] / s64.exp()
    first, winner = times.min(-1)
    first = first.detach()
    lse = s64.exp().sum(-1).log()                                              # scores are clamped to +-12
    first_s = first - first * (lse - lse.detach())                            # clock credit -T pi_i
    delay = .001 + .010 * first_s / (1 + first_s)
    unit_args = (U, P, control_w, control_b, rate, frequency, input_w, output_w, gate_w, gate_b, gain)
    value_w, m_new_w, idx_w = _unit(winner, m, arr_d, seen_d, arrival, incoming, *unit_args)
    values = value_w
    pi = torch.softmax(s64, -1)
    if sampled_credit and U > 1:
        unused_pi = pi                                            # (n, H, U) float64
        alt_times = alt_noise[None] / pi.detach()
        alt_times = alt_times.masked_fill(F.one_hot(winner, U).to(torch.bool), float('inf'))
        alt = alt_times.argmin(-1)                                             # j ~ pi_j / (1 - pi_w) among the losers
        value_j, _, _ = _unit(alt, m, arr_d, seen_d, arrival, incoming, *unit_args)
        pi_j = torch.gather(pi, 2, alt[:, :, None]).squeeze(2)
        pi_w = torch.gather(pi, 2, winner[:, :, None]).squeeze(2)
        q_j = (pi_j / (pi.sum(-1) - pi_w)).detach()                            # pi_j / (1 - pi_w)
        coef = ((pi_j - pi_j.detach()) / q_j).to(value_w.dtype)
        values = values + coef[..., None] * (value_j - value_w).detach()
    onehot = F.one_hot(winner, U).to(torch.bool) & active[:, None, None]
    new_mem = torch.where(onehot[..., None], m_new_w[:, :, None, :], m)
    new_arr = torch.where(onehot, arrival[:, None, None], arr_d)
    new_seen = seen_d | onehot
    refreshed = key.view(H * U, P)[idx_w] + torch.einsum('nhpq,nhq->nhp', key_read.view(H * U, P, P)[idx_w], m_new_w)
    new_reads = torch.where(onehot[..., None], refreshed[:, :, None, :], reads_d)
    arrivals = arrival[:, None] + delay
    arrival_out = arrivals.max(-1).values
    age_h = arrival_out[:, None] - arrivals
    x_out = torch.cat([_transport(values[:, h], age_h[:, h], transport_rate[h], transport_frequency[h]) for h in range(H)], -1)
    return x_out, arrival_out, new_mem, new_arr, new_seen, new_reads, values, arrivals, pi.to(x.dtype)


_COMPILED = None

def compiled_sparse_token_layer():
    global _COMPILED
    if _COMPILED is None:
        _COMPILED = torch.compile(sparse_token_layer, dynamic=False, fullgraph=True)
    return _COMPILED
