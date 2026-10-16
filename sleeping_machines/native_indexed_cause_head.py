"""Exactly normalized native cause law over actual indexed event records.

Reuses the learned CERL word law/clocks; pointer age and successor are those
of the original event, never its position in the retrieved candidate list.
"""
import math
import torch
from torch.nn import functional as F


def indexed_cause_nll(head,views,target,candidates,*,explore=False,selected=None):
    """All depths are a batch of local predictors; no recurrence or carrier.

    views [depth,width], bank features [candidates,depth,width]. The observed
    target enters only likelihood/posterior credit,never rates or discovery.
    Returned messages are for a subsequent event after this target is revealed.
    """
    depth,width=views.shape;k=len(candidates['tokens']);h=views
    if candidates['features'].shape!=(k,depth,width) or k<1:
        raise ValueError('actual indexed native views required')
    values=candidates['features'].transpose(0,1).to(h.dtype).detach()
    ages=candidates['ages'].to(h.dtype).clamp_min(0);valid=candidates['valid']
    repeat=(head.q_rep(h)[:,None]*head.k_rep(values)).sum(-1)/math.sqrt(head.dk)
    repeat=repeat+head.b_rep-F.softplus(head.age_rep)*torch.log1p(ages)
    repeat=repeat.masked_fill(~valid[None],float('-inf'))
    continuation=(head.q_cont(h)[:,None]*head.k_cont(values)).sum(-1)/math.sqrt(head.dk)
    continuation=continuation+head.b_cont-F.softplus(head.age_cont)*torch.log1p(ages)
    continuation=continuation+head.span*torch.log1p(candidates['suffix_matches'].to(h.dtype))
    continuation=continuation.masked_fill(~(valid&(candidates['successors']>=0))[None],float('-inf'))
    if selected is None:selected,s_unit=head.wake(head.q_unit(h))
    else:s_unit=head.unit_score(head.q_unit(h),selected)
    s_background=head.bg(h)
    rates=torch.cat((s_background,s_unit,repeat,continuation),-1)
    H=torch.cat((h[:,None],h[:,None]+head.unit_out[selected]),1)
    targets=torch.full((depth,),int(target),dtype=torch.long,device=h.device)
    lp=head.two_level(H,targets)
    neg=torch.full_like(repeat,float('-inf'))
    hit_repeat=candidates['tokens']==int(target);hit_cont=candidates['successors']==int(target)
    parts=torch.cat((s_background+lp[:,:1],s_unit+lp[:,1:],
        torch.where(hit_repeat[None],repeat,neg),torch.where(hit_cont[None],continuation,neg)),-1)
    nll=torch.logsumexp(rates,-1)-torch.logsumexp(parts,-1)
    aux=nll.new_zeros(())
    if explore and head.explore:
        ex=torch.randint(0,head.U,(depth,head.explore),device=h.device);hd=h.detach()
        s_ex=head.unit_score(head.q_unit(hd),ex)
        s_ex=s_ex.masked_fill((ex[:,:,None]==selected[:,None]).any(-1),float('-inf'))
        lp_ex=head.two_level(hd[:,None]+head.unit_out[ex],targets)
        aux=(torch.logsumexp(torch.cat((rates.detach(),s_ex),-1),-1)
             -torch.logsumexp(torch.cat((parts.detach(),s_ex+lp_ex),-1),-1)).mean()
    with torch.no_grad():
        post=torch.softmax(parts,-1);n=head.topk
        episodic=post[:,1+n:1+n+k]+post[:,1+n+k:]
        message=torch.einsum('dk,dkw->dw',episodic,values)
        message=message+(post[:,1:1+n,None]*head.unit_out[selected]).sum(1)
        responsibilities=torch.stack((post[:,0],post[:,1:1+n].sum(-1),
            post[:,1+n:1+n+k].sum(-1),post[:,1+n+k:].sum(-1)),-1)
    return nll,aux,dict(background_nll=-lp[:,0],posterior_message=message,
        cause_responsibility=responsibilities,concept_indices=selected.detach(),
        candidate_events=k,clock_candidates_per_depth=rates.shape[-1])
