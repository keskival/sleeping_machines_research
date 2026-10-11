"""Differentiable lexical learner state and forward-conditioned block geometry."""
import math
import torch

def unpack(state):
    d=(len(state)-1)//2;return state[:d],state[d:2*d],state[-1]

def forward_geometry(probability,embedding,squares,cache):
    mean=probability@embedding;diagonal=(probability@squares-mean.square()).clamp_min(0)
    u=cache['u'];b=cache['projected'];m=probability@b;entries=probability@cache['pair_products'];rank=b.shape[1]
    covariance=probability.new_zeros(rank,rank);i,j=cache['pair_indices'];covariance[i,j]=entries;covariance[j,i]=entries;covariance=covariance-m[:,None]*m[None,:]
    low=u@covariance@u.T;lowdiag=low.diag().clamp_min(0)
    ratios=torch.where(lowdiag>0,diagonal/lowdiag.clamp_min(1e-30),torch.ones_like(diagonal));scale=ratios.min().clamp(0,1)
    matrix=scale*low+torch.diag((diagonal-scale*lowdiag).clamp_min(0))
    return (matrix+matrix.T)/2

def future_step(function,state,embedding,squares,base,hidden,target,position,cache):
    z,moment,product=unpack(state);logits=base+embedding@z;probability=logits.softmax(0);loss=-logits.log_softmax(0)[target]
    gradient=probability@embedding-embedding[target];curvature=forward_geometry(probability,embedding,squares,cache);diagonal=curvature.diag()
    norm=lambda value:value/value.square().mean().clamp_min(1e-16).sqrt()
    context=torch.cat([norm(hidden),z,norm(moment),norm(gradient),torch.log1p(diagonal/diagonal.mean().clamp_min(1e-8)),torch.stack([loss,state.new_tensor(math.log1p(position))])])
    eta,damping,beta=function(context);updated_m=beta*moment+(1-beta)*gradient;updated_product=beta*product
    scale=curvature.diag().mean().clamp_min(1e-8);matrix=curvature+damping*scale*torch.eye(len(z),dtype=z.dtype,device=z.device)
    delta=eta*torch.linalg.solve(matrix,updated_m/(1-updated_product).clamp_min(1e-8))
    delta=delta*(delta.new_tensor(.08)/delta.norm().clamp_min(1e-12)).clamp_max(1)
    next_state=torch.cat([z-delta,updated_m,updated_product[None]])
    return loss,next_state,dict(eta=eta,damping=damping,beta=beta,z_norm=(z-delta).norm())
