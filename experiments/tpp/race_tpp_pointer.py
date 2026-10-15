"""B4 fighter model: log-domain clocks + addressed mark memory (race_tpp_logspread_marks) + episodic pointer race over the
sequence's own past events (the addressed episodic memory that improved native language by 0.345 nats/token).

For event i (issuing the law of event i+1), candidates in a causal window of W past events race a null route:
  recency vote j <= i  -> mark m_j        score q_i.k_j/sqrt(dk) + b_rec + c * [m_j == m_i] - softplus(r) * log1p(age_ij/scale)
  induction vote j < i -> mark m_{j+1}    score q_i.k_j/sqrt(dk) + b_ind + c * [m_j == m_i] - softplus(r) * log1p(age_ij/scale)
  null                 -> the native conditional mark law p_native(k | tau, history)
p(k | tau, history) = a_null p_native(k) + sum_j a_j [value_j == k]: exactly normalized, time density unchanged.
The null starts dominant (b_null = 12, about 99.8% mass), so an untrained pointer leaves the parent's predictions essentially unchanged.
Gradients reach the query/key maps, biases, match bonus and age clock from reads of any entry in the window."""
import math
import torch
from torch import nn
import torch.nn.functional as F
from race_tpp_logspread_marks import RaceTPP as Parent

WINDOW = 64


class RaceTPP(Parent):
    def __init__(self, *args, key_dim=16, **kwargs):
        super().__init__(*args, **kwargs)
        d = self.embed.embedding_dim; self.key_dim = key_dim
        self.p_query = nn.Linear(d, key_dim, bias=False); self.p_key = nn.Linear(d, key_dim, bias=False)
        nn.init.normal_(self.p_query.weight, std=d ** -.5); nn.init.normal_(self.p_key.weight, std=d ** -.5)
        self.p_rec = nn.Parameter(torch.zeros(())); self.p_ind = nn.Parameter(torch.zeros(())); self.p_null = nn.Parameter(torch.tensor(12.))
        self.p_match = nn.Parameter(torch.tensor(1.)); self.p_age = nn.Parameter(torch.tensor(math.log(math.e ** .5 - 1)))

    def event_terms(self, t, m, mask, target_jitter=0.0):
        time_lp, joint_lp, cache = super().event_terms(t, m, mask, target_jitter)
        h = cache[0]; B, L, _ = h.shape; n = L - 1; W = WINDOW
        q = self.p_query(h[:, :n]); k = self.p_key(h)                       # queries at issuing events 0..n-1; keys at all events
        idx = torch.arange(n, device=h.device)[:, None] - torch.arange(W, device=h.device)[None, :]   # j = i - w, w = 0..W-1
        valid_j = (idx >= 0)                                                # [n, W]
        jj = idx.clamp_min(0)
        kj = k[:, jj]                                                        # [B, n, W, dk]
        dot = (q.unsqueeze(2) * kj).sum(-1) / math.sqrt(self.key_dim)
        mi = m[:, :n]; mj = m[:, jj]; same = (mj == mi.unsqueeze(-1)).to(dot.dtype)
        age = (t[:, :n].unsqueeze(-1) - t[:, jj]).clamp_min(0) / self.scale
        base = dot + self.p_match * same - F.softplus(self.p_age) * torch.log1p(age)
        okj = valid_j.unsqueeze(0) & mask[:, jj] & mask[:, :n].unsqueeze(-1)
        rec_ok = okj & (mj != self.K - 1)                                    # the start marker is never a target
        ind_ok = okj & (idx >= 0).unsqueeze(0) & (torch.arange(W, device=h.device) >= 1)[None, None, :]   # j < i: next mark m_{j+1} known
        s_rec = torch.where(rec_ok, base + self.p_rec, torch.full_like(base, -1e30))
        s_ind = torch.where(ind_ok, base + self.p_ind, torch.full_like(base, -1e30))
        s_null = self.p_null.expand(B, n, 1)
        logits = torch.cat([s_null, s_rec, s_ind], -1); logz = torch.logsumexp(logits, -1, keepdim=True); la = logits - logz
        y = m[:, 1:]                                                        # target marks
        nxt = m[:, (jj + 1).clamp_max(L - 1)]                               # value of induction vote
        hit_rec = (mj == y.unsqueeze(-1)); hit_ind = (nxt == y.unsqueeze(-1))
        neg = torch.full_like(base, -1e30)
        mark_native = joint_lp - time_lp
        parts = torch.cat([la[..., :1] + mark_native.unsqueeze(-1), torch.where(hit_rec, la[..., 1:1 + W], neg), torch.where(hit_ind, la[..., 1 + W:], neg)], -1)
        mark_lp = torch.logsumexp(parts, -1)
        return time_lp, time_lp + mark_lp, cache
