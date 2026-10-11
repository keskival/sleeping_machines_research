"""Causal lexical parameters and complete issued Adam statistics for credit inputs."""
import torch

def learner_features(snapshot,like,dimension,lr=.01):
    z=like.new_zeros(dimension);m=z.clone();v=z.clone();step=0.;b1=.9;b2=.999;eps=1e-8
    if snapshot is not None:
        z=snapshot['z'].detach();groups=snapshot['optimizer']['param_groups']
        if len(groups)!=1:raise ValueError('Single lexical learner required')
        group=groups[0];lr=group['lr'];b1,b2=group['betas'];eps=group['eps']
        fields=list(snapshot['optimizer']['state'].values())
        if fields:
            if len(fields)!=1:raise ValueError('Single lexical parameter required')
            m=fields[0]['exp_avg'].detach();v=fields[0]['exp_avg_sq'].detach();step=float(fields[0]['step'])
    if z.shape!=(dimension,) or m.shape!=z.shape or v.shape!=z.shape or bool((v<0).any()):raise ValueError('Invalid issued learner state')
    signed=lambda x:x.sign()*torch.log1p(x.abs())
    direction=m/(v.sqrt()+eps)
    common=torch.cat([signed(z/.01),signed(m/.01),torch.log1p(v.sqrt()/.001),signed(direction),like.new_tensor([lr/.01,b1,b2,torch.log10(like.new_tensor(eps)).item(),torch.log1p(like.new_tensor(step)).item()])])
    assert bool(torch.isfinite(common).all());return common
