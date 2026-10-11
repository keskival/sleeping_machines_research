"""Differentiable positive-support audit probabilities with an exact budget.

p_i=floor+(1-floor)*sigmoid(score_i-threshold). The shared scalar threshold
enforces sum p=budget. Its implicit adjoint closes analytically, without a
learned trainer for the policy's credit. Higher-order differentiation is
unsupported and rejected explicitly.
"""
import torch


class _BudgetSigmoid(torch.autograd.Function):
    @staticmethod
    def forward(ctx, scores, budget, floor):
        if scores.ndim != 1 or not torch.isfinite(scores).all():
            raise ValueError('finite vector scores required')
        if not (0 < floor < 1 and len(scores)*floor < budget < len(scores)):
            raise ValueError('budget outside positive-support interval')
        low, high = scores.min()-80, scores.max()+80
        for _ in range(80):
            middle = (low+high)/2
            p = floor+(1-floor)*torch.sigmoid(scores-middle)
            if float(p.sum()) > budget: low = middle
            else: high = middle
        sigmoid = torch.sigmoid(scores-(low+high)/2)
        p = floor+(1-floor)*sigmoid
        q = (1-floor)*sigmoid*(1-sigmoid)
        ctx.save_for_backward(q)
        return p

    @staticmethod
    def backward(ctx, cotangent):
        if torch.is_grad_enabled():
            raise RuntimeError('first-order budget adjoint only')
        q, = ctx.saved_tensors
        return q*(cotangent-(cotangent*q).sum()/q.sum()), None, None


def budgeted_probabilities(scores, budget, floor=.01):
    return _BudgetSigmoid.apply(scores, float(budget), float(floor))


def forward_audit_policy_loss(scores, squared_residual_labels, budget, floor=.01):
    if scores.shape != squared_residual_labels.shape:
        raise ValueError('one frozen forward label per score required')
    if not torch.isfinite(squared_residual_labels).all() or (squared_residual_labels < 0).any():
        raise ValueError('finite nonnegative labels required')
    p = budgeted_probabilities(scores, budget, floor)
    return (squared_residual_labels.detach()*(1/p-1)).mean()
