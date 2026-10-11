"""Two actual causal learners; one carrier and reference-derived route projections."""
from copy import deepcopy
import torch
from torch.nn import functional as F
from shared_carrier_paired_replay import packet, frozen_scores


def episode(source,t,m,receivers,actions,factual,reference,lr,project=True,diagnostics=False):
    if not 0<=factual<len(receivers) or not 0<=reference<len(receivers):
        raise ValueError('Actual routes must belong to the issued candidate set')
    identity=receivers[factual]==receivers[reference] and torch.equal(actions[factual],actions[reference])
    identity=identity or bool((actions[factual]==0).all() and (actions[reference]==0).all())
    p=packet(source,t,m,receivers,actions)
    ids=[reference] if identity else [reference,factual]
    heads=[deepcopy(p.model.mark_ctx).requires_grad_(True) for _ in ids]
    opts=[torch.optim.Adam(h.parameters(),lr=lr) for h in heads]
    actual=[];projected=[];gradients=[]
    def costs(logits,j,rows):
        pk=F.log_softmax(F.log_softmax(logits+p.slot_logits[:,j:j+1],-1)+p.bonus[rows,j:j+1,None],-1)
        selected=pk[:,0].gather(-1,p.marks[rows,j+1,None,None].expand(-1,p.model.M,1)).squeeze(-1)
        return -(torch.logsumexp(p.log_h[:,j]+selected,-1)+p.log_s[:,j].sum(-1))
    for target in range(4,m.shape[1]):
        j=target-1;values=[];gs=[]
        for k,(head,opt,row) in enumerate(zip(heads,opts,ids)):
            opt.zero_grad();logits=head(p.h[:,j:j+1]).view(1,1,p.model.M,p.model.K)
            if k==0 and project:
                with torch.no_grad(): projected.append(costs(logits.detach(),j,list(range(len(receivers)))))
            loss=costs(logits,j,[row]);values.append(loss.detach()[0]);loss.sum().backward()
            if diagnostics:gs.append({n:v.grad.detach().clone() for n,v in head.named_parameters()})
            opt.step()
        actual.append(torch.stack([values[-1],values[0]]));gradients.append(gs)
    result=dict(actual_scores=torch.stack(actual,1),frozen_scores=frozen_scores(p),
        projected_scores=torch.stack(projected,1) if project else None,
        actual_worlds=len(ids),readout_updates=len(ids)*(m.shape[1]-4),projected_route_targets=len(receivers)*(m.shape[1]-4) if project else 0)
    if diagnostics:result.update(head_states=[deepcopy(h.state_dict()) for h in heads],optimizers=[deepcopy(o.state_dict()) for o in opts],gradients=gradients)
    return result
