"""B4 one-core conditional mark readout; candidate time is a density argument.
Features never include the scored target mark. Original time density unchanged.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

@torch.no_grad()
def distributions(model,t,m,mask):
    h,slots=model.encode(t,m,mask);hp=h[:,:-1];params=model.clocks(hp,slots[:,:-1]);tau=(t[:,1:]-t[:,:-1]).clamp_min(0)
    lh,ls=model.clock_terms(tau,*params[:4]);hazards=lh;marked=lh.unsqueeze(-1)+params[4];survival=ls.sum(-1);aged=None
    if model.ns:
        dt=torch.diff(t,dim=1,prepend=t[:,:1]).clamp_min(0)/model.scale
        zr,zi=model.state_seq(h,dt);zr,zi=zr[:,:-1],zi[:,:-1]
        state_tau=model.state_tau(tau)
        sl,sp=model.state_eval(hp,zr,zi,state_tau.unsqueeze(-1));sl,sp=sl[...,0],sp[...,0,:]
        hazards=torch.cat([lh,sl.unsqueeze(-1)],-1);marked=torch.cat([marked,(sl.unsqueeze(-1)+sp).unsqueeze(-2)],-2)
        survival=survival-model.state_comp(hp,zr,zi,tau)
        rate,freq=model.state_dynamics();elapsed=state_tau.unsqueeze(-1)/model.scale
        decay=torch.exp(-rate*elapsed);angle=freq*elapsed;cr=decay*torch.cos(angle);ci=decay*torch.sin(angle)
        aged=torch.cat([cr*zr-ci*zi,ci*zr+cr*zi],-1)
    total_h=torch.logsumexp(hazards,-1);lp=torch.logsumexp(marked,-2)-total_h.unsqueeze(-1)
    components=[hp,slots[:,:-1].flatten(-2),model.embed(m[:,:-1]),lp]
    if aged is not None:components.append(aged)
    gap=tau/model.scale;components.append(torch.stack([torch.log1p(gap),torch.exp(-gap),(gap==0).to(gap.dtype)],-1))
    features=torch.cat(components,-1)
    return total_h+survival,lp,features

class SingleMarkReadout(nn.Module):
    def __init__(self,core,mean,std,coefficients):
        super().__init__();self.core=core
        self.register_buffer('feature_mean',mean.clone());self.register_buffer('feature_std',std.clone())
        self.readout=nn.Linear(len(mean),core.K).to(dtype=mean.dtype)
        with torch.no_grad():
            self.readout.weight.copy_(coefficients[:-1].T);self.readout.bias.copy_(coefficients[-1])
    @torch.no_grad()
    def event_distributions(self,t,m,mask):
        time,lp,features=distributions(self.core,t,m,mask)
        correction=self.readout((features-self.feature_mean)/self.feature_std)
        return time,F.log_softmax(lp+correction,-1)
