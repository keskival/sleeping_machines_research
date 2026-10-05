"""Normalized frequency initialization for dense/adaptive readouts.

A train-only unigram starting distribution separates marginal-token learning
from contextual learning. All biases remain trainable; the event core is intact.
"""
import torch
from torch import nn
from .token_readout import TokenReadout


class FrequencyTokenReadout(TokenReadout):
    def __init__(self, width, counts, cutoffs=(), rank_to_token=None, smoothing=.1):
        if smoothing <= 0 or torch.any(counts < 0) or counts.ndim != 1:
            raise ValueError('Nonnegative training counts and positive smoothing required')
        super().__init__(width, len(counts), cutoffs, rank_to_token)
        prior = counts[self.rank_to_token].float() + smoothing
        with torch.no_grad():
            if self.adaptive:
                # The original primitive omits tail intercepts. Add trainable
                # intercepts so every branch can express its token marginals.
                old = self.output.head
                self.output.head = nn.Linear(old.in_features,old.out_features,bias=True)
                nn.init.zeros_(self.output.head.weight)
                masses = [prior[:cutoffs[0]]]
                for low, high in zip(self.output.cutoffs[:-1],self.output.cutoffs[1:]):
                    masses.append(prior[low:high].sum().reshape(1))
                self.output.head.bias.copy_(torch.cat(masses).log())
                for tail, low, high in zip(self.output.tail,self.output.cutoffs[:-1],self.output.cutoffs[1:]):
                    old = tail[1]
                    tail[1] = nn.Linear(old.in_features,old.out_features,bias=True)
                    nn.init.zeros_(tail[1].weight)
                    tail[1].bias.copy_(prior[low:high].log())
            else:
                nn.init.zeros_(self.output.weight)
                self.output.bias.copy_(prior.log())
