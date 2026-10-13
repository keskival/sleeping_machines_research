"""Single event-local derivative sweep produces own and shared-path endpoint gradients."""
import numpy as np
import torch
from experiments.credit.market_streaming_clock_updates import control
from experiments.market.local_eligibility_state import MarketLegEligibility
from experiments.market.projected_clock_state import ProjectedClockState

def pair_window(model,state,t,m,start,group):
    arm='projected'
    params=tuple(model.parameters());names=[n for n,_ in model.named_parameters()];scores=[]
    if arm=='bptt':
        losses=[]
        for k in range(start,start+group):
            forecast,state=control.market_consume(model,t[k:k+1],m[k:k+1],state)
            loss=-control.revealed_log_probability(model,forecast,t[k+1:k+2],m[k+1:k+2]).mean();scores.append(float(loss.detach()));losses.append(loss)
        grads=torch.autograd.grad(torch.stack(losses).mean(),params,allow_unused=True)
        return control.detach_state(state),[torch.zeros_like(p) if g is None else g.detach() for p,g in zip(params,grads)],scores
    projected=ProjectedClockState(model,state) if arm=='projected' else None
    def consumer(actor,time,mark,old,parameter_version=0):return projected.advance(time,mark,old,control.market_consume)
    observer=MarketLegEligibility(model,state,consumer if projected else control.market_consume) if arm in ['own','projected'] else None
    total=[torch.zeros_like(p) for p in params];own_total=[torch.zeros_like(p) for p in params]
    for k in range(start,start+group):
        if projected:
            state['layers']=[z.detach().requires_grad_() for z in state['layers']];state['slots']=state['slots'].detach().requires_grad_()
        old_tensors=tuple(control.tensors(state)) if projected else ()
        forecast,end=observer.consume(t[k:k+1],m[k:k+1],state) if observer else control.market_consume(model,t[k:k+1],m[k:k+1],state)
        loss=-control.revealed_log_probability(model,forecast,t[k+1:k+2],m[k+1:k+2]).mean();scores.append(float(loss.detach()))
        targets=tuple(control.tensors(end)) if observer else ()
        grads=torch.autograd.grad(loss/group,params+targets+old_tensors,allow_unused=True)
        remote=observer.historical_gradient(grads[len(params):len(params)+len(targets)]) if observer else {}
        own_remote={n:g.copy() for n,g in remote.items()}
        if projected:
            correction=projected.historical_projection(grads[len(params)+len(targets):])-projected.project_parameters({n:torch.as_tensor(g) for n,g in remote.items()})
            for j,l in enumerate(model.layers):
                for offset,key in enumerate(['log_rate','freq']):remote[f'layers.{j}.{key}']+=correction[2*j+offset]/np.sqrt(l.n)
        for i,(name,p,g) in enumerate(zip(names,params,grads[:len(params)])):
            if g is not None:total[i]+=g.detach();own_total[i]+=g.detach()
            if name in remote:total[i]+=torch.as_tensor(remote[name],dtype=p.dtype)
            if name in own_remote:own_total[i]+=torch.as_tensor(own_remote[name],dtype=p.dtype)
        state=control.detach_state(end)
    return state,own_total,total,scores
