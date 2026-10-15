"""TRAIN-only binary native readout conditioning, with a fixed ridge penalty.

Standard convex logistic regression for diagnosis/block fitting; it does not
replace native representations, timing or memory. No DEV selection is used.
"""
import torch
from torch.nn import functional as F


def fit_binary_readout(features,labels,ridge=.01,max_steps=64):
    x=features.detach().double();y=labels.to(x.device).double()
    if x.ndim!=2 or y.shape!=(len(x),) or set(y.tolist())!={0.,1.}:
        raise ValueError('finite features and both binary classes required')
    if ridge<=0 or not torch.isfinite(x).all():raise ValueError('positive ridge and finite features required')
    center=x.mean(0);scale=x.std(0,unbiased=False).clamp_min(1e-6)
    design=torch.cat(((x-center)/scale,torch.ones_like(x[:,:1])),1)
    penalty=x.new_full((design.shape[1],),ridge);penalty[-1]=0
    beta=x.new_zeros(design.shape[1]);beta[-1]=torch.logit(y.mean())
    def objective(value):
        logits=design@value
        return (F.softplus(logits)-y*logits).mean()+.5*(penalty*value.square()).sum()
    for iteration in range(max_steps):
        prob=(design@beta).sigmoid()
        gradient=design.T@(prob-y)/len(y)+penalty*beta
        if float(gradient.abs().max())<1e-9:break
        hessian=(design.T*(prob*(1-prob)))@design/len(y)+torch.diag(penalty)
        direction=torch.linalg.solve(hessian,gradient)
        old=objective(beta);step=1.
        for _ in range(40):
            candidate=beta-step*direction
            if objective(candidate)<=old-1e-4*step*(gradient@direction):break
            step*=.5
        else:raise RuntimeError('convex readout line search failed')
        beta=candidate
    weight=beta[:-1]/scale;bias=beta[-1]-weight@center
    return dict(weight=weight,bias=bias,ridge=ridge,iterations=iteration+1,
        objective=float(objective(beta)),max_gradient=float(gradient.abs().max()),
        center=center,scale=scale)


def binary_logits(features,fit):
    score=features.double()@fit['weight']+fit['bias']
    return torch.stack((torch.zeros_like(score),score),-1)
