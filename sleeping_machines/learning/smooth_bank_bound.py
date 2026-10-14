"""Direction-preserving smooth radial bound inside one declared bank.

A shared cap preserves mathematical ordering of bank norms, not the direction
of the concatenated whole-model gradient or raw gradient ratios.
"""
import math
import torch

def smooth_bank_bound(gradient,cap):
    if not math.isfinite(cap) or cap<=0:raise ValueError('Positive finite cap required')
    if not gradient.is_floating_point() or not bool(torch.isfinite(gradient).all()):raise ValueError('Finite floating bank gradient required')
    bound=gradient.new_tensor(cap)
    if not bool(torch.isfinite(bound)) or float(bound)<=0.:raise ValueError('Cap is not representable in bank dtype')
    norm=torch.linalg.vector_norm(gradient)
    if not bool(torch.isfinite(norm)):raise ValueError('Bank norm overflow')
    if float(norm)==0.:return gradient.clone()
    return gradient*((bound*torch.tanh(norm/bound))/norm)
