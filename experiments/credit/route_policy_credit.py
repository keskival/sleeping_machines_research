"""Exact factual-residual correction for issued categorical route policies."""
import torch


def route_credit_objective(logits, projected_costs, action, factual_cost, exploration=.1):
    if not 0 <= exploration <= 1:
        raise ValueError('Exploration outside probability range')
    if logits.ndim != 1 or projected_costs.shape != logits.shape:
        raise ValueError('One issued cost per route is required')
    if not 0 <= action < len(logits):
        raise ValueError('Unissued factual route')
    probabilities = (1-exploration)*logits.softmax(0)+exploration/len(logits)
    h = projected_costs.detach()
    correction = torch.as_tensor(factual_cost, dtype=logits.dtype,
                                 device=logits.device).detach()-h[action]
    return (probabilities*h).sum()+correction*probabilities[action].log()
