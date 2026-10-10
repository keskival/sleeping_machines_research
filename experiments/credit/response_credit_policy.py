"""State-sized credit from learned response activations; bounded write policy.
This policy changes a key write, not all forward weights. Bounds require the
validated fixed-query response setting and externally certified gradient error.
"""
import math
import torch

def activations(critic,x,actor_version,credit_version):
    h=critic.encoder(x.tanh())[0][:,-1];v=critic.head(h).reshape(len(x),critic.R,6+2*critic.M)
    raw=v[:,:,:4];u=critic.U*raw/(1+raw.norm(dim=-1,keepdim=True))
    return dict(u=u,p=v[:,:,4:4+critic.M].sigmoid(),w=v[:,:,4+critic.M:4+2*critic.M].softmax(-1),b=v[:,:,-2].sigmoid(),mix=v[:,:,-1].softmax(-1),actor_version=actor_version,credit_version=credit_version,U=critic.U)

def cotangent(packet):
    difference=(packet['w']*packet['p']).sum(-1)-packet['b']
    return (packet['mix'][:,:,None]*difference[:,:,None]*packet['u']).sum(1)

def propose(packet,actor_version,credit_version,radius=.25,gradient_error=None,event_cost=0.,cost_weight=0.):
    if not math.isfinite(radius) or radius<=0:raise ValueError('radius')
    if any(not math.isfinite(v) or v<0 for v in (event_cost,cost_weight)):raise ValueError('cost')
    if gradient_error is not None and (not math.isfinite(gradient_error) or gradient_error<0):raise ValueError('error')
    g=cotangent(packet);zero=torch.zeros_like(g)
    if actor_version!=packet['actor_version'] or credit_version!=packet['credit_version']:
        return zero,dict(status='stale_recompute_required')
    U=packet['U']
    if not math.isfinite(U) or U<=0:raise ValueError('bound')
    L=U*U/4;norm=g.norm(dim=-1);eta=torch.minimum(torch.full_like(norm,1/L),radius/norm.clamp_min(1e-30))
    delta=-eta[:,None]*g
    if gradient_error is None:return delta,dict(status='bounded_uncertified',steps=eta.detach().tolist())
    upper=-eta*(1-L*eta/2)*norm.square()+eta*gradient_error*norm
    emit=upper+cost_weight*event_cost<0
    return torch.where(emit[:,None],delta,zero),dict(status='conditional_descent_certificate',emit=emit.tolist(),future_loss_upper_bound=upper.detach().tolist(),scope='External error bound must cover current conditional future gradient; past MSE or sample point gain does not supply it. Event cost is incremental emission cost; prediction/training overhead is charged separately.')
