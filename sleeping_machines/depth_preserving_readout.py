"""Extend a trained token decoder to timestamp-aligned native depth outputs.

Known zero-initialized linear extension; no new normalization or tail rank.
This utility does not alter native routing, memory or the training procedure.
"""
import copy
import torch
from torch import nn


def extend_depth_readout(readout, depth):
    """Place original input weights on the deepest leg; earlier legs start zero.

    Returns an independent decoder. Floating-point kernels can introduce small
    rounding differences despite algebraic equality. Integration requires the
    guarded probability/gradient/RNG contract before any fit.
    """
    if not isinstance(depth, int) or depth < 1:
        raise ValueError('Positive integer depth required')
    result = copy.deepcopy(readout)
    if depth == 1:
        return result

    def extended(layer):
        if not isinstance(layer, nn.Linear):
            raise TypeError('Expected a linear decoder input')
        width = layer.in_features
        # Creating the extra interface must not advance the training RNG.
        devices = [] if layer.weight.device.type == 'cpu' else [layer.weight.device.index]
        with torch.random.fork_rng(devices=devices):
            replacement = nn.Linear(width * depth, layer.out_features,
                                    bias=layer.bias is not None,
                                    device=layer.weight.device, dtype=layer.weight.dtype)
        with torch.no_grad():
            replacement.weight.zero_()
            replacement.weight[:, -width:].copy_(layer.weight)
            if layer.bias is not None:
                replacement.bias.copy_(layer.bias)
        replacement.weight.requires_grad_(layer.weight.requires_grad)
        if layer.bias is not None:
            replacement.bias.requires_grad_(layer.bias.requires_grad)
        return replacement

    if result.adaptive:
        result.output.head = extended(result.output.head)
        for tail in result.output.tail:
            tail[0] = extended(tail[0])
        result.output.in_features *= depth
    else:
        result.output = extended(result.output)
    return result
