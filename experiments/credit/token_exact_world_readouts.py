"""Exact independent lexical readout worlds batched over one fixed token carrier."""
import torch
from torch.nn import functional as F
from token_forward_credit import bonuses

def replay(p,receivers,actions,lr,diagnostics=False):
    if len(receivers)!=len(actions) or not len(receivers):raise ValueError('Each world needs an address and message')
    beta=bonuses(p,actions);r=torch.tensor(receivers,dtype=torch.long);worlds=len(r)
    def learner():
        z=torch.nn.Parameter(p.embedding.new_zeros(worlds,p.embedding.shape[1]));return z,torch.optim.Adam([z],lr=lr)
    z,opt=learner();values=[];gradients=[]
    for j,target in enumerate(p.targets.tolist()):
        if bool(p.reset[j]):z,opt=learner()
        opt.zero_grad();logits=p.log_probabilities[j][None]+z@p.embedding.T
        logits=logits.scatter_add(1,r[:,None],beta[:,j:j+1])
        loss=F.cross_entropy(logits,torch.full((worlds,),target,dtype=torch.long),reduction='none')
        values.append(loss.detach());loss.sum().backward()
        if diagnostics:gradients.append(z.grad.detach().clone())
        opt.step()
    if p.final_eos:z,opt=learner()
    result=dict(scores=torch.stack(values,1),z=z.detach().clone(),actual_worlds=worlds,readout_updates=worlds*len(p.targets))
    if diagnostics:result.update(gradients=gradients,optimizer=opt.state_dict())
    return result
