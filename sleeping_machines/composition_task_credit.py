"""Exact local joint-choice score surrogate from actual conditional suffix losses.

Does not obtain branch losses or install itself in a model. Callers must replay
the real conditional suffix under a legal joint alternative and supply q.
Factual direct/pathwise derivatives and clock coupling remain separate.
"""
import torch
from .coupled_native_query_credit import ordered_pair_probabilities


def actual_joint_loss_credit(scores,factual,alternative,factual_loss,alternative_loss,proposal=None):
    pi=ordered_pair_probabilities(scores).flatten();n=scores.shape[1]
    for pair in [factual,alternative]:
        if len(pair)!=2 or pair[0]==pair[1] or any(not 0<=i<n for i in pair):raise ValueError('legal distinct ordered pair required')
    w=factual[0]*n+factual[1];a=alternative[0]*n+alternative[1]
    if a==w:raise ValueError('nonfactual alternative required')
    q=(pi[a]/(1-pi[w])).detach() if proposal is None else torch.as_tensor(proposal,dtype=pi.dtype,device=pi.device).detach()
    if q.ndim or not bool(torch.isfinite(q)) or not 0<float(q)<=1:raise ValueError('actual positive sampling probability required')
    delta=torch.as_tensor(alternative_loss,dtype=pi.dtype,device=pi.device).detach()-torch.as_tensor(factual_loss,dtype=pi.dtype,device=pi.device).detach()
    if delta.ndim or not bool(torch.isfinite(delta)):raise ValueError('finite actual scalar suffix losses required')
    return (pi[a]-pi[a].detach())*delta/q
