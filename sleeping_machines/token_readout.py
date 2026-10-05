"""Exactly normalized adaptive token readout; supporting primitive, not the event core."""
import torch
from torch import nn
from torch.nn import functional as F


class TokenReadout(nn.Module):
    def __init__(self, width, vocab, cutoffs=(), rank_to_token=None):
        super().__init__()
        order = torch.arange(vocab) if rank_to_token is None else torch.as_tensor(rank_to_token, dtype=torch.long)
        if not torch.equal(order.sort().values, torch.arange(vocab)):
            raise ValueError('Frequency order must be a vocabulary permutation fitted on training only')
        self.register_buffer('rank_to_token', order)
        self.register_buffer('token_to_rank', order.argsort())
        self.adaptive = bool(cutoffs)
        self.output = (nn.AdaptiveLogSoftmaxWithLoss(width, vocab, list(cutoffs), div_value=2.)
                       if cutoffs else nn.Linear(width, vocab))

    def nll(self, features, targets):
        x = features.reshape(-1, features.shape[-1])
        y = self.token_to_rank[targets.reshape(-1)]
        if self.adaptive:
            return -self.output(x, y).output
        return F.cross_entropy(self.output(x), y, reduction='none')

    def log_prob(self, features):
        x = features.reshape(-1, features.shape[-1])
        z = self.output.log_prob(x) if self.adaptive else F.log_softmax(self.output(x), -1)
        return z[:, self.token_to_rank]
