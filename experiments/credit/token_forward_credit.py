"""Exact proper-token key-write projections with independent fast readout learners.

The temporal/key carrier is fixed. Each learner adapts only a shared lexical
readout vector z, logits += E@z; all full-vocabulary normalization stays paid.
"""
from dataclasses import dataclass
import math
import torch
from torch.nn import functional as F
from token_episode_state import consume

@dataclass
class TokenPacket:
    log_probabilities:torch.Tensor
    queries:torch.Tensor
    targets:torch.Tensor
    reset:torch.Tensor
    effect_live:torch.Tensor
    embedding:torch.Tensor
    rates:torch.Tensor
    dk:int
    index:int
    state:dict
    final_eos:bool


def packet(model,tokens,index=3,initial_state=None,eos_token=50256):
    if tokens.ndim!=1 or len(tokens)<index+2 or not 0<=index<len(tokens)-1:
        raise ValueError('One observed token stream and at least one post-write target required')
    if not model.keyed:raise ValueError('Keyed token model required')
    probabilities=[];queries=[];resets=[];live=[];state=initial_state;active=False
    captured=[];hook=model.query_w.register_forward_hook(lambda module,args,result:captured.append(result.detach()))
    try:
        with torch.no_grad():
            for j in range(len(tokens)-1):
                reset=int(tokens[j])==eos_token
                if reset:state=None;active=False
                forecast,state=consume(model,tokens[j:j+1],state,parameter_version=0)
                if j==index:active=True
                if j>=index:
                    probabilities.append(forecast['log_probabilities'][0]);queries.append(captured[-1].reshape(model.heads,model.dk))
                    resets.append(reset);live.append(active)
    finally:hook.remove()
    return TokenPacket(torch.stack(probabilities),torch.stack(queries),tokens[index+1:],torch.tensor(resets),torch.tensor(live),model.embed.weight.detach(),model.log_rate.detach().exp(),model.dk,index,state,int(tokens[-1])==eos_token)


def bonuses(p,actions):
    if actions.ndim!=3 or actions.shape[1:]!=p.queries.shape[1:]:raise ValueError('One head/dimension message per route')
    age=torch.arange(len(p.targets),dtype=p.embedding.dtype)
    transported=p.queries*torch.exp(-age[:,None]*p.rates[None])[:,:,None]
    return torch.einsum('thd,whd->wt',transported,actions)/math.sqrt(p.dk)*p.effect_live[None]


def projected_costs(logits,receivers,beta,target):
    """Stable one-address log normalization, O(V) preprocessing + O(routes)."""
    if logits.ndim!=1 or len(logits)<2:raise ValueError('Full vocabulary logits required')
    prefix=torch.logcumsumexp(logits,0);suffix=torch.logcumsumexp(logits.flip(0),0).flip(0)
    left=torch.cat((logits.new_full((1,),-torch.inf),prefix[:-1]))
    right=torch.cat((suffix[1:],logits.new_full((1,),-torch.inf)))
    other=torch.logaddexp(left,right);r=torch.as_tensor(receivers,dtype=torch.long)
    return torch.logaddexp(other[r],logits[r]+beta)-logits[target]-beta*(r==target)


def joint_replay(p,receivers,actions,factual,reference,lr,initial_z=None,diagnostics=False):
    if len(receivers)!=len(actions) or not 0<=factual<len(actions) or not 0<=reference<len(actions):raise ValueError('Actual routes must be issued')
    beta=bonuses(p,actions);r=torch.tensor(receivers,dtype=torch.long)
    if min(receivers)<0 or max(receivers)>=len(p.embedding):raise ValueError('Vocabulary address outside model')
    identity=(receivers[factual]==receivers[reference] and torch.equal(actions[factual],actions[reference])) or bool((actions[factual]==0).all() and (actions[reference]==0).all())
    ids=[reference] if identity else [reference,factual]
    initial=p.embedding.new_zeros(p.embedding.shape[1]) if initial_z is None else initial_z.detach()
    def learner(value):
        z=torch.nn.Parameter(value.clone());return z,torch.optim.Adam([z],lr=lr)
    heads=[learner(initial) for _ in ids];scores=[];projected=[];frozen=[];gradients=[]
    for j,target in enumerate(p.targets.tolist()):
        if bool(p.reset[j]):heads=[learner(torch.zeros_like(initial)) for _ in ids]
        values=[];gs=[]
        for k,(row,(z,opt)) in enumerate(zip(ids,heads)):
            opt.zero_grad();logits=p.log_probabilities[j]+p.embedding@z
            if k==0:
                with torch.no_grad():
                    projected.append(projected_costs(logits.detach(),receivers,beta[:,j],target))
                    frozen.append(projected_costs(p.log_probabilities[j],receivers,beta[:,j],target))
            # Each actual learner uses standard full normalized vocabulary CE.
            changed=logits+F.one_hot(r[row],len(logits)).to(logits.dtype)*beta[row,j]
            loss=F.cross_entropy(changed[None],torch.tensor([target]));values.append(loss.detach())
            loss.backward();gs.append(z.grad.detach().clone());opt.step()
        scores.append(torch.stack([values[-1],values[0]]));gradients.append(gs)
    if p.final_eos:heads=[learner(torch.zeros_like(initial)) for _ in ids]
    result=dict(actual_scores=torch.stack(scores,1),projected_scores=torch.stack(projected,1),frozen_scores=torch.stack(frozen,1),actual_worlds=len(ids),readout_updates=len(ids)*len(p.targets),final_z=[z.detach().clone() for z,opt in heads])
    if diagnostics:result.update(gradients=gradients,optimizers=[opt.state_dict() for z,opt in heads])
    return result
