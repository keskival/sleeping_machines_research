"""Selectable contextual rank for normalized frequency-initialized tails."""
import torch
from torch import nn
from .frequency_token_readout import FrequencyTokenReadout


def frequency_readout(width, counts, cutoffs, order, minimum_tail_width=0):
    if minimum_tail_width < 0 or minimum_tail_width > width:
        raise ValueError('Tail minimum must be between zero and the feature width')
    result = FrequencyTokenReadout(width, counts, cutoffs, order)
    if result.adaptive:
        prior = counts[result.rank_to_token].float() + .1
        for i, (tail, low, high) in enumerate(zip(result.output.tail,
                result.output.cutoffs[:-1], result.output.cutoffs[1:])):
            existing = tail[0].out_features
            if existing < minimum_tail_width:
                replacement = nn.Sequential(nn.Linear(width, minimum_tail_width, bias=False),
                    nn.Linear(minimum_tail_width, high-low, bias=True))
                with torch.no_grad():
                    replacement[1].weight.zero_()
                    replacement[1].bias.copy_(prior[low:high].log())
                result.output.tail[i] = replacement
    result.tail_widths = ([tail[0].out_features for tail in result.output.tail]
                         if result.adaptive else [])
    return result
