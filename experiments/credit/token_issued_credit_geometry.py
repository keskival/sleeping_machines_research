"""Causal native-feature mean shifts for issued route controller inputs."""
import math
import torch
from token_issued_route_pool import issue_with_history


def route_mean_shifts(logits, embedding, receivers, beta):
    probabilities=logits.softmax(0);mean=probabilities@embedding
    r=torch.tensor(receivers,dtype=torch.long);p=probabilities[r]
    # Stable normalizer for a single-address exponential tilt.
    logden=torch.logaddexp(torch.log1p(-p),p.log()+beta)
    q=(p.log()+beta-logden).exp()
    q=torch.where(beta==0,p,q)
    if bool((p>=1).any()):
        raise ValueError('Saturated route probability needs a leave-one-out feature normalizer')
    mean_except=(mean[None]-p[:,None]*embedding[r])/(1-p)[:,None]
    return (q-p)[:,None]*(embedding[r]-mean_except),q-p,mean


def normalized(v):
    return v/v.square().mean(-1,keepdim=True).sqrt().clamp_min(1e-8)


def issue_geometry(model,observed_tokens,observed_history,initial_state=None,fast_state=None,eos=50256):
    issued=issue_with_history(model,observed_tokens,observed_history,initial_state,fast_state,eos)
    d=model.embed.embedding_dim;k=model.heads*model.dk
    context=issued['context'];h,q,w,z=torch.split(context,[d,k,k,d])
    beta=(issued['actions']*q.reshape(model.heads,model.dk)[None]).sum((1,2))/math.sqrt(model.dk)
    with torch.no_grad():
        logits=issued['forecast']['log_probabilities'][0]+model.embed.weight.detach()@z
        shift,probability_shift,mean=route_mean_shifts(logits,model.embed.weight.detach(),issued['receivers'],beta)
        base=issued['descriptors'];route_e,key,message,scalars=torch.split(base,[d,k,k,3],dim=1)
        # Preserve relative head activation/amplitude via beta and shift strength;
        # normalize vector blocks to avoid raw age/key magnitudes dominating tanh.
        route_features=torch.cat([normalized(route_e),normalized(key),normalized(message),
                                  torch.log1p(scalars[:,0:1].clamp_min(0)),scalars[:,1:3],
                                  normalized(shift),beta[:,None],
                                  torch.log1p(probability_shift.abs()[:,None]/1e-4)],1)
        common=torch.cat([normalized(h),normalized(q),normalized(w),normalized(z),normalized(mean)])
    return dict(issued,context=common,descriptors=route_features,mean_shifts=shift)
