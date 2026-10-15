"""B10-P models: the matched B10 encoders (native race / LSTM / Transformer) with causal price-feature inputs and a
horizon return-distribution head scored on exact tick cells. The trade-tape likelihood (time + side/move mark) is kept."""
import math
import torch
from torch import nn
import torch.nn.functional as F
from b10_matched_models import MatchedRace
from b10_depth_tpp_v1 import CELL, log_interval
from b10p_data import NFEAT, HORIZONS

COMPONENTS = 3


def log_cell_mixture(lo, hi, logits, mu, log_sigma):
    """log P(lo < r <= hi) under a Gaussian mixture; stable in both tails. lo/hi: [...]; params: [..., C]."""
    sigma = F.softplus(log_sigma) + 1e-3
    a = (lo.unsqueeze(-1) - mu) / sigma; b = (hi.unsqueeze(-1) - mu) / sigma
    upper = a > 0                                                     # use survival tails above the mean
    x = torch.where(upper, -b, a); y = torch.where(upper, -a, b)        # P = Phi(y) - Phi(x), x < y
    ly, lx = torch.special.log_ndtr(y), torch.special.log_ndtr(x)
    comp = ly + torch.log(-torch.expm1((lx - ly).clamp_max(-1e-300)))
    return torch.logsumexp(F.log_softmax(logits, -1) + comp, -1)


class PriceRace(MatchedRace):
    def __init__(self, gaps, family, scales):
        super().__init__(gaps, family)
        d = 32
        self.feat = nn.Linear(NFEAT, d); nn.init.zeros_(self.feat.weight); nn.init.zeros_(self.feat.bias)
        width = d + self._slot_width()
        self.price = nn.Sequential(nn.Linear(width, 64), nn.GELU(), nn.Linear(64, len(HORIZONS) * COMPONENTS * 3))
        nn.init.zeros_(self.price[-1].weight); nn.init.zeros_(self.price[-1].bias)
        self.register_buffer('log_scale', torch.log(torch.tensor(scales, dtype=torch.float64)))
        with torch.no_grad():                                       # components start at 0.5x/1x/2x TRAIN scale
            b = self.price[-1].bias.view(len(HORIZONS), COMPONENTS, 3)
            for h, s in enumerate(scales):
                for c, k in enumerate((0.5, 1.0, 2.0)):
                    b[h, c, 2] = math.log(math.expm1(s * k))

    def _slot_width(self):
        return self.m.K * self.m.marks.dv

    def encode_features(self, t, m, f):
        dt = torch.diff(t, dim=1, prepend=t[:, :1]).clamp_min(0) / self.m.scale
        extra = self.feat(f)
        if self.family == 'native':
            x = self.m.embed(m) + self.m.gap(torch.stack([torch.log1p(dt), (dt > 0).to(t.dtype)], -1)) + extra
            for layer in self.m.layers:
                x = layer(x, dt)
            return x, self.m.marks(x, m, dt)
        x = self.m.embed(m) + self.m.gap(torch.log1p(dt).unsqueeze(-1)) + extra
        if self.family == 'transformer':
            frequency = torch.exp(torch.arange(8, dtype=t.dtype, device=t.device) * (-math.log(10000) / 8))
            elapsed = (t - t[:, :1]) / self.m.scale
            position = torch.arange(t.shape[1], dtype=t.dtype, device=t.device).expand_as(t)
            def pe(z):
                a = z.unsqueeze(-1) * frequency
                return torch.stack((a.sin(), a.cos()), -1).flatten(-2)
            x = x + torch.cat((pe(elapsed), pe(position)), -1)
        h = self.encoder(x)
        return h, self.m.marks(h, m, dt)

    def all_terms(self, t, m, f, lo, hi):
        """Per event: tape time/mark log-likelihood for the next event, and per-horizon tick-cell return log-likelihood
        from the state after the event (labels NaN where invalid -> 0 with mask)."""
        h, slots = self.encode_features(t, m, f)
        p = self.m.clocks(h[:, :-1], slots[:, :-1])
        g = (t[:, 1:] - t[:, :-1]).clamp_min(0)
        _, ls0 = self.m.clock_terms(g, *p[:4]); _, ls1 = self.m.clock_terms(g + CELL, *p[:4])
        lh, _ = self.m.clock_terms(g + CELL / 2, *p[:4])
        log_pk = p[4].gather(-1, m[:, 1:, None, None].expand(-1, -1, self.m.M, 1)).squeeze(-1)
        mark_ll = torch.logsumexp(lh + log_pk, -1) - torch.logsumexp(lh, -1)
        time_ll = log_interval(ls0.sum(-1), ls1.sum(-1))
        raw = self.price(torch.cat([h, slots.flatten(-2)], -1)).view(*h.shape[:2], len(HORIZONS), COMPONENTS, 3)
        valid = torch.isfinite(lo)
        lo0 = torch.where(valid, lo, torch.zeros_like(lo)); hi0 = torch.where(valid, hi, torch.ones_like(hi))
        price_ll = log_cell_mixture(lo0, hi0, raw[..., 0], raw[..., 1], raw[..., 2])
        return time_ll, mark_ll, torch.where(valid, price_ll, torch.zeros_like(price_ll)), valid

    def predictive(self, t, m, f):
        """Mixture parameters per event and horizon (for secondary metrics)."""
        h, slots = self.encode_features(t, m, f)
        raw = self.price(torch.cat([h, slots.flatten(-2)], -1)).view(*h.shape[:2], len(HORIZONS), COMPONENTS, 3)
        return F.softmax(raw[..., 0], -1), raw[..., 1], F.softplus(raw[..., 2]) + 1e-3
