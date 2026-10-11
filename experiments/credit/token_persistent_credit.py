"""Causal carried token carrier/readout states across credit windows.
The copied readout math is source-preserving; incoming optimizer snapshots are
independently deep-cloned and only the selected factual state is committed.
"""
from copy import deepcopy
import torch
from torch.nn import functional as F
from token_forward_credit import packet,bonuses,projected_costs


def carry_carrier(p,receiver,action):
    # Packet state is after the last issued event, before its scored target.
    # The target is consumed exactly once as the next window's first event.
    if p.final_eos:return None
    state=dict(p.state)
    if not bool(p.effect_live[-1]) or not bool((action!=0).any()):return state
    keys=dict(state['keys']);now=state['position']-1
    elapsed=len(p.targets)-1
    delta=action*torch.exp(-p.rates*elapsed)[:,None]
    if receiver in keys:
        value,when=keys[receiver];delta=delta+value*torch.exp(-p.rates*(now-when))[:,None]
    keys[receiver]=(delta.detach(),now)
    return dict(state,keys=keys)


def joint_replay_carry(p,receivers,actions,factual,reference,lr,initial_learner=None,diagnostics=False):
    if len(receivers)!=len(actions) or not 0<=factual<len(actions) or not 0<=reference<len(actions):raise ValueError('Actual routes must be issued')
    beta=bonuses(p,actions);r=torch.tensor(receivers,dtype=torch.long)
    if min(receivers)<0 or max(receivers)>=len(p.embedding):raise ValueError('Vocabulary address outside model')
    identity=(receivers[factual]==receivers[reference] and torch.equal(actions[factual],actions[reference])) or bool((actions[factual]==0).all() and (actions[reference]==0).all())
    ids=[reference] if identity else [reference,factual]
    initial=p.embedding.new_zeros(p.embedding.shape[1]) if initial_learner is None else initial_learner['z'].detach()
    def learner(value):
        z=torch.nn.Parameter(value.clone());return z,torch.optim.Adam([z],lr=lr)
    heads=[learner(initial) for _ in ids]
    if initial_learner is not None:
        for z,opt in heads:opt.load_state_dict(deepcopy(initial_learner['optimizer']))
    scores=[];projected=[];frozen=[];gradients=[]
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
    result['factual_learner']=dict(z=heads[-1][0].detach().clone(),optimizer=deepcopy(heads[-1][1].state_dict()))
    result['factual_carrier']=carry_carrier(p,receivers[factual],actions[factual])
    if diagnostics:result.update(gradients=gradients,optimizers=[opt.state_dict() for z,opt in heads])
    return result
