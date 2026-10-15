"""Addressed episodic memory with read-time credit for the native token learner.

Every observed token event appends one entry to a persistent per-lane store:
key source = the learner's feature after that token (stored detached once it
leaves the current credit chunk), value = the token that followed it. A query
from the current feature scores stored keys with the CURRENT key map, adds a
learned age clock and a learned document-boundary bias, keeps only the top-k
candidates (sparse read) and races them against a null route that returns mass
to the native readout. A learned gate mixes the resulting pointer distribution
with the native probabilities, so the output is a normalized distribution.

Learning rule: an entry written long ago receives credit when it is READ.
Gradients reach the query/key maps, clock, boundary bias, null route and gate
from reads of arbitrarily old entries without backpropagating through the
intervening state evolution; persistent numerical memory is separate from the
gradient window. Entries written inside the current chunk keep their path to
the writer features, so the core also learns to write retrievable keys.
"""
import math
import torch
from torch import nn
import torch.nn.functional as F

EOS = 50256


class EpisodicPointerMemory(nn.Module):
    def __init__(self, feature_dim, key_dim=32, capacity=1024, topk=32, gate_init=0.15):
        super().__init__()
        self.capacity, self.topk, self.key_dim = capacity, topk, key_dim
        self.query = nn.Linear(feature_dim, key_dim, bias=False)
        self.key = nn.Linear(feature_dim, key_dim, bias=False)
        nn.init.normal_(self.query.weight, std=feature_dim ** -.5)
        nn.init.normal_(self.key.weight, std=feature_dim ** -.5)
        self.age_rate = nn.Parameter(torch.tensor(math.log(math.e ** .5 - 1)))  # softplus -> 0.5
        self.boundary = nn.Parameter(torch.zeros(()))
        self.null = nn.Parameter(torch.zeros(()))
        self.gate = nn.Linear(feature_dim + 1, 1)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, math.log(gate_init / (1 - gate_init)))

    def empty(self, lanes, feature_dim):
        return dict(feats=torch.zeros(lanes, 0, feature_dim), vals=torch.zeros(lanes, 0, dtype=torch.long),
                    pos=torch.zeros(lanes, 0, dtype=torch.long), eos=torch.full((lanes,), -1, dtype=torch.long),
                    start=torch.zeros(lanes, dtype=torch.long))

    def read(self, x, ids, targets, memory):
        """x (L,n,d) features after ids; targets (L,n) next tokens.

        Returns pointer weights over candidate values, null weight, gate and
        the updated (detached) memory. Query t may read old entries and chunk
        entries j<t (whose value targets[j]=ids[j+1] is already observed)."""
        L, n, _ = x.shape
        positions = memory['start'][:, None] + torch.arange(n)
        feats = torch.cat([memory['feats'], x], 1)
        vals = torch.cat([memory['vals'], targets], 1)
        pos = torch.cat([memory['pos'], positions], 1)
        keys = self.key(feats); q = self.query(x)
        scores = q @ keys.transpose(1, 2) / math.sqrt(self.key_dim)               # (L,n,M)
        age = (positions[:, :, None] - pos[:, None, :]).clamp_min(0).float()
        scores = scores - F.softplus(self.age_rate) * torch.log1p(age)
        eos_seen = torch.where(ids == EOS, positions, torch.full_like(positions, -1))
        last_eos = torch.maximum(torch.cummax(eos_seen, 1).values, memory['eos'][:, None])
        scores = scores + self.boundary * (pos[:, None, :] < last_eos[:, :, None]).float()
        valid = pos[:, None, :] < positions[:, :, None]
        scores = scores.masked_fill(~valid, -math.inf)
        k = min(self.topk, scores.shape[-1])
        top, index = scores.topk(k, -1)
        logits = torch.cat([top, self.null.expand(L, n, 1)], -1)
        weights = torch.softmax(logits, -1)                                         # -inf -> 0
        candidate_vals = vals[:, None, :].expand(L, n, -1).gather(2, index)
        peak = torch.where(torch.isfinite(top[..., :1]), top[..., :1], torch.zeros_like(top[..., :1]))
        gate = torch.sigmoid(self.gate(torch.cat([x, peak], -1))).squeeze(-1)
        keep = self.capacity
        new = dict(feats=feats[:, -keep:].detach(), vals=vals[:, -keep:], pos=pos[:, -keep:],
                   eos=last_eos[:, -1], start=memory['start'] + n)
        return weights[..., :-1], candidate_vals, weights[..., -1], gate, new

    def mixed_nll(self, native_nll, x, ids, targets, memory):
        weights, candidate_vals, null, gate, new = self.read(x, ids, targets, memory)
        pointer = (weights * (candidate_vals == targets[..., None]).float()).sum(-1)
        native_log = -native_nll.reshape(pointer.shape)
        keep = torch.log((1 - gate + gate * null).clamp_min(1e-30)) + native_log
        copy = torch.log((gate * pointer).clamp_min(1e-30))
        log_p = torch.where(pointer > 0, torch.logaddexp(keep, copy), keep)
        return -log_p, new, dict(gate=float(gate.detach().mean()), pointer_hit=float((pointer > 0).float().mean()),
                                 null=float(null.detach().mean()))

    def full_log_prob(self, native_log_prob, x, ids, targets, memory):
        """Normalized mixed distribution over the vocabulary (contracts/inference)."""
        weights, candidate_vals, null, gate, new = self.read(x, ids, targets, memory)
        pointer = torch.zeros_like(native_log_prob).scatter_add_(-1, candidate_vals, weights)
        p = (1 - gate + gate * null)[..., None] * native_log_prob.exp() + gate[..., None] * pointer
        return p.clamp_min(1e-30).log(), new

    def work(self, lanes, n, stored, feature_dim):
        m = stored + n
        return dict(key_map_macs=lanes * m * feature_dim * self.key_dim, query_macs=lanes * n * feature_dim * self.key_dim,
                    score_macs=lanes * n * m * self.key_dim, selected_reads=lanes * n * min(self.topk, m))
