"""B1/R1 shared critic for optional writes to never-labelled native receivers.
Frozen actor diagnostic; all receiver discovery and teacher work are charged.
"""
import argparse,hashlib,json,math,resource,time
from copy import deepcopy
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.flop_counter import FlopCounterMode
import structured_future_credit_v5 as prior
base=prior.base;ROOT=base.ROOT
SOURCES=['experiments/credit/unseen_receiver_credit.py',*prior.SOURCES]
CALLS=0
class RouteActor(base.ReceiverCuts):
    route_intervention=None
    def encode(self,t,m,mask):
        h,slots=super().encode(t,m,mask)
        if self.route_intervention is not None:
            index,receiver,delta=self.route_intervention;state=h.new_zeros(len(t),self.K,self.dk);history=[]
            for j in range(m.shape[1]):
                addressed=torch.nn.functional.one_hot(m[:,j],self.K).to(h.dtype)
                state=state*self.key_decay[:,j,None,:]+addressed[:,:,None]*self.written_keys[:,j,None,:]
                if j==index:state=state+torch.nn.functional.one_hot(receiver,self.K).to(h.dtype)[:,:,None]*delta[:,None,:]
                history.append(state)
            self._keyed=torch.stack(history,1)
        return h,slots

def actor(depth,seed):
    original=base.actor(depth,seed);model=RouteActor(8,8,3,depth,2,2,2,0.,1.,[-1.,0.],dk=4,local=True,prev_msg=True,qk_norm=False).double();model.load_state_dict(original.state_dict());return model

def losses(model,t,m,index=None,receiver=None,delta=None):
    global CALLS;CALLS+=1;model.route_intervention=None if index is None else (index,receiver,delta)
    try:return base.event_losses(model,t,m)
    finally:model.route_intervention=None

def prefix(model,t,m,index,receivers):
    global CALLS;CALLS+=1;x=base.prefix_packet(model,t,m,index);_,cache=losses(model,t[:,:index+1],m[:,:index+1]);params=model.clocks(cache[0],cache[1]);packets=[]
    for r in receivers:
        key=model._keyed[:,:,r];prob=params[4][...,r].exp().mean(-1,keepdim=True);match=(m[:,:index+1]==r).to(x.dtype).unsqueeze(-1);count=match.cumsum(1)/(torch.arange(index+1,dtype=x.dtype)[None,:,None]+1);role=torch.full_like(match,float(r>=4))
        packets.append(torch.cat((x,key,prob,match,count,role),-1))
    return torch.stack(packets,1).detach()

def responses(model,t,m,index,receivers,actions):
    _,cache=losses(model,t,m);query=model._query[:,:-1];age=(t[:,:-1]-t[:,index:index+1]).clamp_min(0)/model.scale;u=query*torch.exp(-age[:,:,None]*model.key_log_rate.exp())/math.sqrt(model.dk);lh,_=model.clock_terms(cache[3],*cache[2][:4]);target=cache[2][4].gather(-1,m[:,1:,None,None].expand(-1,-1,model.M,1)).squeeze(-1);logw=(lh+target).log_softmax(-1);values=[]
    for r in receivers:
        p=cache[2][4][...,r].exp();b=(m[:,1:]==r).to(query.dtype);values.append(prior.decode(u[:,index:],p[:,index:],logw[:,index:],b[:,index:],actions[None]).mean(1))
    return torch.stack(values,1)

def relabel(model,permutation):
    new=deepcopy(model);old=model.state_dict();state=new.state_dict();d=model.embed.embedding_dim;dv=model.marks.dv
    state['embed.weight'][permutation]=old['embed.weight'];state['mark_ctx.weight'].view(model.M,model.K,d)[:,permutation]=old['mark_ctx.weight'].view(model.M,model.K,d);state['mark_ctx.bias'].view(model.M,model.K)[:,permutation]=old['mark_ctx.bias'].view(model.M,model.K)
    state['clock.weight'][:,d:].view(-1,model.K,dv)[:,permutation]=old['clock.weight'][:,d:].reshape(-1,model.K,dv);new.load_state_dict(state);return new

def contracts():
    rows=[]
    for depth in (2,4,8):
        model=actor(depth,173);t,m=base.batch(173,size=1,pairs=2);actions=base.actions();index=3;exact=responses(model,t,m,index,list(range(8)),actions);original,_=losses(model,t,m);error=0.
        for receiver in range(8):
            action=actions[(receiver%8)+1][None];changed,_=losses(model,t,m,index,torch.tensor([receiver]),action);error=max(error,abs(float((changed[:,index:]-original[:,index:]).mean()-exact[0,receiver,(receiver%8)+1])));assert torch.equal(original[:,:index],changed[:,:index])
        assert error<1e-10,error
        delta=torch.zeros(1,4,dtype=torch.float64,requires_grad=True);changed,_=losses(model,t,m,index,torch.tensor([7]),delta);native_grad=torch.autograd.grad(changed[:,index:].mean(),delta)[0]
        d=delta.detach().clone().requires_grad_();g=torch.autograd.grad(responses(model,t,m,index,[7],d).mean(),d)[0];gradient_error=float((g-native_grad).abs().max());assert gradient_error<1e-10
        receivers=list(range(8));x=prefix(model,t,m,index,receivers);future=m.clone();future[:,index+1:]=(future[:,index+1:]+1)%8;causal=float((x-prefix(model,t,future,index,receivers)).abs().max());assert causal==0
        perm=torch.tensor([0,1,2,3,6,7,4,5]);new=relabel(model,perm);px=prefix(new,t,perm[m],index,perm.tolist());feature_error=float((x-px).abs().max());rx=responses(new,t,perm[m],index,perm.tolist(),actions);response_error=float((rx-exact).abs().max());assert feature_error<1e-10 and response_error<1e-10,(feature_error,response_error)
        rows.append(dict(depth=depth,optional_route_value_error=error,optional_route_gradient_error=gradient_error,causal_prefix_error=causal,value_relabel_descriptor_error=feature_error,value_relabel_response_error=response_error,receivers=8,prefix_feature_width=x.shape[-1]))
    return rows

def collect(model,seed,count,rollouts,receivers,dev=False):
    xs=[];ys=[];records=[]
    with torch.no_grad():
        for i in range(count):
            t,m=base.batch((2100000 if dev else 2000000)+seed*1000+i,size=1,pairs=2);xs.append(prefix(model,t,m,3,receivers));rng=np.random.default_rng((22000000 if dev else 21000000)+seed*10000+i);values=[]
            for j in range(rollouts):
                ft,fm=prior.continuation(t,m,2,rng);values.append(responses(model,ft,fm,3,receivers,base.actions())[0])
                if j==0:records.append((ft,fm,values[-1]))
            ys.append(torch.stack(values).mean(0))
    return torch.cat(xs),torch.stack(ys),records
