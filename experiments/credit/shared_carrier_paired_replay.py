"""Exact addressed branch learners share a route-independent fixed carrier."""
from copy import deepcopy
from dataclasses import dataclass
import math
import torch
from torch.nn import functional as F
from optimizer_aware_route_credit import BranchReadout,native


@dataclass
class CarrierPacket:
    model:object
    h:torch.Tensor
    slot_logits:torch.Tensor
    bonus:torch.Tensor
    log_h:torch.Tensor
    log_s:torch.Tensor
    frozen_pk:torch.Tensor
    marks:torch.Tensor


def packet(source,t,m,receivers,actions):
    if len(t)!=1 or t.shape!=m.shape or m.shape[1]<5:
        raise ValueError('One observed sequence with at least one post-write target required')
    if not len(receivers) or len(receivers)!=len(actions) or actions.shape!=(len(receivers),source.dk):
        raise ValueError('Each addressed world needs one route and action')
    if source.qk_norm:raise ValueError('Linear unnormalized key transport required')
    model=deepcopy(source);model.requires_grad_(False);routes=torch.tensor(receivers)
    with torch.no_grad():
        _,cache=native.losses(model,t,m)
        h,slots,params,tau,_=cache;lh,ls=model.clock_terms(tau,*params[:4])
        slot_logits=torch.einsum('blkv,mv->blmk',slots[:,:-1],model.mark_slot)
        base_bonus=torch.einsum('bld,blkd->blk',model._query[:,:-1],model._keyed[:,:-1])
        base_bonus=base_bonus/math.sqrt(model.dk)
        live=torch.zeros_like(actions);betas=[]
        for j in range(m.shape[1]-1):
            live=live*model.key_decay[0,j]
            if j==3:live=live+actions
            betas.append((live*model._query[0,j]).sum(-1)/math.sqrt(model.dk))
        beta=torch.stack(betas,1)
        addressed=F.one_hot(routes,model.K).to(h.dtype)
        bonus=base_bonus.expand(len(routes),-1,-1)+beta[:,:,None]*addressed[:,None]
        frozen_pk=F.log_softmax(params[4].expand(len(routes),-1,-1,-1)+beta[:,:,None,None]*addressed[:,None,None],-1)
    return CarrierPacket(model,h,slot_logits,bonus,lh,ls,frozen_pk,m.expand(len(routes),-1))


def frozen_scores(p):
    selected=p.frozen_pk.gather(-1,p.marks[:,1:,None,None].expand(-1,-1,p.model.M,1)).squeeze(-1)
    return -(torch.logsumexp(p.log_h+selected,-1)+p.log_s.sum(-1))[:,3:]


def replay_pairs(source,t,m,receivers,actions,lr,diagnostics=False):
    p=packet(source,t,m,receivers,actions);model=p.model;worlds=len(receivers)
    model.mark_ctx=BranchReadout(model.mark_ctx,worlds);opt=torch.optim.Adam(model.mark_ctx.parameters(),lr=lr)
    scores=[];gradients=[]
    for target in range(4,m.shape[1]):
        j=target-1;opt.zero_grad()
        h=p.h[:,j:j+1].expand(worlds,-1,-1)
        logits=model.mark_ctx(h).view(worlds,1,model.M,model.K)
        pk=F.log_softmax(F.log_softmax(logits+p.slot_logits[:,j:j+1],-1)+p.bonus[:,j:j+1,None],-1)
        selected=pk[:,0].gather(-1,p.marks[:,target,None,None].expand(-1,model.M,1)).squeeze(-1)
        current=-(torch.logsumexp(p.log_h[:,j]+selected,-1)+p.log_s[:,j].sum(-1))
        assert torch.isfinite(current).all();scores.append(current.detach());current.sum().backward()
        if diagnostics:gradients.append({k:v.grad.detach().clone() for k,v in model.mark_ctx.named_parameters()})
        opt.step()
    result=(torch.stack(scores,1),{k:v.detach().clone() for k,v in model.mark_ctx.state_dict().items()})
    if diagnostics:result+=({'gradients':gradients,'optimizer':deepcopy(opt.state_dict()),'frozen_scores':frozen_scores(p),
        'shared_carrier_values':sum(v.numel() for v in [p.h,p.slot_logits,p.log_h,p.log_s]),
        'route_specific_values':p.bonus.numel()+p.frozen_pk.numel()},)
    return result
