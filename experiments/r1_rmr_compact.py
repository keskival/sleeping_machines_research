"""Exact current-slot-only RMR execution without zero quadratic old-write state.

The selected dense-stack variant has top_m=0 at every level. Its old-slot
write matrices are identically zero. Eliminate their storage and products,
retaining the current-slot law, trainable objects and zero old-branch gradients.
Models that actually refine old slots retain the original implementation.
"""
import math
import torch
from torch.nn import functional as F
from experiments.r1_rmr import RMR


class CompactRMR(RMR):
    def build_memory(self, x):
        if any(level.top_m for level in self.levels):
            return super().build_memory(x)
        total = self.base.states(x)
        batch, length, _ = total.shape
        positions = torch.arange(length, device=x.device)
        stats = []
        for level in self.levels:
            own_new = torch.zeros_like(total)
            own_old = torch.zeros_like(total)
            for start in range(0, length, self.chunk):
                stop = min(start + self.chunk, length)
                rows = positions[start:stop]
                previous = (positions < start).to(total.dtype)
                available = total + own_new * previous[None, :, None]
                u = level.nq(total[:, start:stop])
                scores = level.q(u) @ level.k(available).transpose(1, 2)
                distance = (rows[:, None] - positions[None, :]).to(total.dtype)
                scores = (scores / math.sqrt(level.dk)
                          - F.softplus(level.log_rate) * torch.log1p(distance.clamp_min(0)))
                scores = scores.masked_fill(distance < 0, float('-inf'))
                if level.read_topk and level.read_topk < length:
                    threshold = scores.topk(level.read_topk, -1).values[..., -1:]
                    scores = scores.masked_fill(scores < threshold, float('-inf'))
                read = torch.softmax(scores, -1) @ available
                new, old = level.g(torch.cat([u, level.nr(read)], -1)).chunk(2, -1)
                if level.rezero:
                    new = level.alpha_new * new
                    old = level.alpha_old * torch.tanh(old)
                own_new = own_new.index_copy(1, rows, new)
                own_old = own_old.index_copy(1, rows, old)
            # Original zero-matrix products generate zero tensors for old-write
            # gradients. Preserve those tensors (and AdamW behavior), not None.
            total = total + own_new + own_old.sum() * 0
            stats.append(dict(new_norm=float(own_new.detach().norm(dim=-1).mean()),
                              old_norm=float(own_old.detach().norm(dim=-1).mean())))
        return total, [], stats
