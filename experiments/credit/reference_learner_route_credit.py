"""Causal reference learner supplies all addressed route effects from one carrier."""
from copy import deepcopy
import torch
from torch.nn import functional as F
from shared_carrier_paired_replay import packet,frozen_scores


def shadow_scores(source,t,m,receivers,actions,reference_index,lr,diagnostics=False):
    p=packet(source,t,m,receivers,actions)
    if not 0<=reference_index<len(receivers):raise ValueError('Reference must be an addressed world')
    head=deepcopy(p.model.mark_ctx);head.requires_grad_(True)
    opt=torch.optim.Adam(head.parameters(),lr=lr);scores=[];gradients=[]
    for target in range(4,m.shape[1]):
        j=target-1;opt.zero_grad()
        logits=head(p.h[:,j:j+1]).view(1,1,p.model.M,p.model.K)
        pk=F.log_softmax(F.log_softmax(logits+p.slot_logits[:,j:j+1],-1)+p.bonus[:,j:j+1,None],-1)
        selected=pk[:,0].gather(-1,p.marks[:,target,None,None].expand(-1,p.model.M,1)).squeeze(-1)
        loss=-(torch.logsumexp(p.log_h[:,j]+selected,-1)+p.log_s[:,j].sum(-1))
        assert torch.isfinite(loss).all();scores.append(loss.detach())
        # Only the fixed reference trains this common head. Other routes are
        # projections for credit, not secretly independent branch learners.
        loss[reference_index].backward()
        if diagnostics:gradients.append({k:v.grad.detach().clone() for k,v in head.named_parameters()})
        opt.step()
    result=(torch.stack(scores,1),{k:v.detach().clone() for k,v in head.state_dict().items()})
    if diagnostics:result+=({'gradients':gradients,'optimizer':deepcopy(opt.state_dict()),'frozen_scores':frozen_scores(p)},)
    return result
