"""B1/R1 exact fixed-carrier replay with causal adapting branch readouts."""
from copy import deepcopy
import math
import torch
from torch.nn import functional as F
from optimizer_aware_route_credit import BranchReadout, native


def replay_pairs(source,t,m,receivers,actions,lr,diagnostics=False):
    if len(receivers)!=len(actions) or not len(receivers):
        raise ValueError('Each addressed world needs one route and action')
    # Same parameter/state snapshot; no initialization or global RNG mutation.
    model=deepcopy(source); model.requires_grad_(False)
    model.mark_ctx=BranchReadout(model.mark_ctx,len(receivers))
    opt=torch.optim.Adam(model.mark_ctx.parameters(),lr=lr)
    tt,mm=t.repeat(len(receivers),1),m.repeat(len(receivers),1)
    with torch.no_grad():
        _,cache=native.losses(model,tt,mm,3,torch.tensor(receivers),actions)
        h,slots,params,tau,_=cache
        log_h,log_s=model.clock_terms(tau,*params[:4])
        slot_logits=torch.einsum('blkv,mv->blmk',slots[:,:-1],model.mark_slot)
        bonus=torch.einsum('bld,blkd->blk',model._query[:,:-1],model._keyed[:,:-1])
        if not model.qk_norm:bonus=bonus/math.sqrt(model.dk)
    values=[];gradients=[]
    for target in range(4,m.shape[1]):
        j=target-1;opt.zero_grad()
        logits=model.mark_ctx(h[:,j:j+1]).view(len(receivers),1,model.M,model.K)
        # Preserve both original normalizations, including finite precision.
        pk=F.log_softmax(F.log_softmax(logits+slot_logits[:,j:j+1],-1)+bonus[:,j:j+1,None,:],-1)
        selected=pk[:,0].gather(-1,mm[:,target,None,None].expand(-1,model.M,1)).squeeze(-1)
        current=-(torch.logsumexp(log_h[:,j]+selected,-1)+log_s[:,j].sum(-1))
        assert torch.isfinite(current).all();values.append(current.detach())
        current.sum().backward()
        if diagnostics:gradients.append({k:p.grad.detach().clone() for k,p in model.mark_ctx.named_parameters()})
        opt.step()
    weights={k:v.detach().clone() for k,v in model.mark_ctx.state_dict().items()}
    result=(torch.stack(values,1),weights)
    if diagnostics:
        result+=({'gradients':gradients,'optimizer':deepcopy(opt.state_dict()),
                  'retained_carrier_values':sum(x.numel() for x in [h,slot_logits,bonus,log_h,log_s])},)
    return result
