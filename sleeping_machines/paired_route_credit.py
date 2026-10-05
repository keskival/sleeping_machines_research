"""Conditional alternative utility credit with a zero-forward surrogate."""
import torch


def paired_route_credit(probabilities, alternative, proposal_probability, loss_difference):
    """Return mean conditional route credit; only probabilities carry derivatives.

    The caller samples a nonwinner independently of outcomes, evaluates its
    actual continuation, and supplies q for that sample. This estimator credits
    choice scores, not differentiable alternative values or future parameters.
    """
    if probabilities.ndim != 2 or alternative.shape != probabilities.shape[:1]:
        raise ValueError('Per-lane probabilities and alternative indices required')
    if proposal_probability.shape != alternative.shape or loss_difference.shape != alternative.shape:
        raise ValueError('Per-lane proposal probabilities and utility differences required')
    q = proposal_probability.detach()
    if not bool(torch.isfinite(q).all()) or not bool((q > 0).all()):
        raise ValueError('Sampled alternatives require finite positive proposal support')
    pj = probabilities.gather(1, alternative[:, None]).squeeze(1)
    return ((pj - pj.detach()) * loss_difference.detach() / q).mean()
