"""Predictable route-effect credit with unbiased admitted residual correction.

The caller supplies a causal factual frame, winner, outcome-independent
admission/proposal probabilities, and separately charges all predictor work.
Site/horizon weights remain the caller's responsibility.
"""
import torch


def anchored_predictions(predicted, winner):
    if predicted.ndim!=2 or winner.shape!=predicted.shape[:1]:raise ValueError('Per-lane predicted alternatives and winners required')
    return predicted*(1-torch.nn.functional.one_hot(winner,predicted.shape[1]).to(predicted.dtype))


def predictable_route_credit(probabilities,predicted,winner,*,admitted,admission_probability,alternative=None,proposal_probability=None,loss_difference=None):
    if probabilities.shape!=predicted.shape:raise ValueError('Prediction/route shape mismatch')
    p=float(admission_probability)
    if not 0<p<=1:raise ValueError('Positive causal admission probability required')
    b=anchored_predictions(predicted,winner)
    # Actor receives no derivative through prediction values, even for shared
    # actor/critic leaves. Supervised predictor learning is a separate term.
    credit=((probabilities-probabilities.detach())*b.detach()).sum(-1).mean()
    if not admitted:return credit
    if alternative is None or alternative.shape!=winner.shape:raise ValueError('Admitted per-lane alternative required')
    if bool((alternative==winner).any()):raise ValueError('Teacher alternative must differ from factual winner')
    if proposal_probability is None or loss_difference is None or proposal_probability.shape!=winner.shape or loss_difference.shape!=winner.shape:raise ValueError('Admitted per-lane propensity and observed difference required')
    q=proposal_probability.detach()
    if not bool(torch.isfinite(q).all()) or not bool((q>0).all()):raise ValueError('Finite positive proposal required')
    pj=probabilities.gather(1,alternative[:,None]).squeeze(1)
    residual=loss_difference.detach()-b.detach().gather(1,alternative[:,None]).squeeze(1)
    return credit+((pj-pj.detach())*residual/(p*q)).mean()


def admitted_prediction_loss(predicted,winner,alternative,proposal_probability,loss_difference,admission_probability):
    p=float(admission_probability)
    if not 0<p<=1:raise ValueError('Positive admission probability required')
    b=anchored_predictions(predicted,winner)
    q=proposal_probability.detach()
    if q.shape!=winner.shape or loss_difference.shape!=winner.shape or alternative.shape!=winner.shape:raise ValueError('Per-lane teacher receipt required')
    if bool((alternative==winner).any()) or not bool(torch.isfinite(q).all()) or not bool((q>0).all()):raise ValueError('Positive legal alternative propensity required')
    residual=b.gather(1,alternative[:,None]).squeeze(1)-loss_difference.detach()
    return (residual.square()/(p*q)).mean()
