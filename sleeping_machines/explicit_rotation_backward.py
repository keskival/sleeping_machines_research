"""Experimental first-order rotation kernel; existing model sources unchanged.

Forward retains float64 trigonometry and payload-precision multiplication.
Backward preserves the original cast/reduction boundary. Higher derivatives
are explicitly unsupported; this kernel is not a default model substitution.
"""
from contextlib import contextmanager
import torch
from torch.autograd.function import once_differentiable


class ExplicitRotation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, state, angles):
        pairs = state.reshape(*state.shape[:-1], -1, 2)
        a, b = pairs.unbind(-1)
        cos64, sin64 = angles.cos(), angles.sin()
        cos, sin = cos64.to(state.dtype), sin64.to(state.dtype)
        ctx.save_for_backward(a, b, cos, sin, cos64, sin64)
        ctx.state_shape = state.shape
        return torch.stack((cos*a-sin*b, sin*a+cos*b), -1).flatten(-2)

    @staticmethod
    @once_differentiable
    def backward(ctx, grad):
        a, b, cos, sin, cos64, sin64 = ctx.saved_tensors
        ga, gb = grad.reshape(*grad.shape[:-1], -1, 2).unbind(-1)
        state_grad = angle_grad = None
        if ctx.needs_input_grad[0]:
            state_grad = torch.stack((ga*cos+gb*sin, gb*cos-ga*sin), -1).reshape(ctx.state_shape)
        if ctx.needs_input_grad[1]:
            # Reduce each multiply adjoint BEFORE summing and casting, as in
            # the original broadcast-multiply -> cast -> trig graph.
            dc = (ga*a).sum_to_size(cos.shape)+(gb*b).sum_to_size(cos.shape)
            ds = (gb*a).sum_to_size(sin.shape)-(ga*b).sum_to_size(sin.shape)
            angle_grad = -dc.to(cos64.dtype)*sin64+ds.to(sin64.dtype)*cos64
        return state_grad, angle_grad


def explicit_rotate(state, angles):
    return ExplicitRotation.apply(state, angles)


@contextmanager
def rotation_kernel():
    """Process-local probe only; no pinned source edits or persistent patch."""
    from . import sparse_training, sparse_counterfactual_episodes
    original_unit = sparse_training._rotate
    original_context = sparse_counterfactual_episodes.precise_rotate
    sparse_training._rotate = explicit_rotate
    sparse_counterfactual_episodes.precise_rotate = explicit_rotate
    try:
        yield
    finally:
        sparse_training._rotate = original_unit
        sparse_counterfactual_episodes.precise_rotate = original_context
