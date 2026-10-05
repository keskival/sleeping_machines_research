"""Outcome-independent site support and chunk-mean utility weighting."""
import torch


def sample_site(length, depth, heads, generator):
    if min(length, depth, heads) < 1:
        raise ValueError('Nonempty event-site dimensions required')
    total = length * depth * heads
    index = int(torch.randint(total, (), generator=generator))
    token, inner = divmod(index, depth * heads)
    layer, head = divmod(inner, heads)
    return token, layer, head, total


def utility_weight(length, token, inverse_site_probability):
    if not 0 <= token < length or inverse_site_probability <= 0:
        raise ValueError('Valid event position and positive sampling support required')
    return inverse_site_probability * (length - token) / length
