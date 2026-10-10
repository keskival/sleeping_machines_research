"""Explicit causal episode state for native temporal/keyed market models.
Serial CPU reference for event-time operation, not a clockless hardware claim.
"""
import math
import torch
from torch.nn import functional as F
from b10_tpp import CELL,log_interval
from race_tpp_v5 import RaceTPP

def detach_state(value):
    if isinstance(value,torch.Tensor):return value.detach()
    if isinstance(value,dict):return {k:detach_state(v) for k,v in value.items()}
    if isinstance(value,tuple):return tuple(detach_state(v) for v in value)
    if isinstance(value,list):return [detach_state(v) for v in value]
    return value

def initial(model,batch):
    zero=model.embed.weight
    return dict(layers=[zero.new_zeros(batch,2*l.n) for l in model.layers],slots=zero.new_zeros(batch,model.K,model.marks.dv),keys=zero.new_zeros(batch,model.K,model.dk) if hasattr(model,'key_w') else None,last_time=None,last_mark=None,parameter_version=None)

def consume(model,time,mark,state=None,parameter_version=0,key_delta=None):
    if time.ndim!=1 or mark.shape!=time.shape or mark.dtype!=torch.long:raise ValueError('batch event shape')
    if not torch.isfinite(time).all():raise ValueError('nonfinite time')
    st=initial(model,len(time)) if state is None else state
    elapsed=torch.zeros_like(time) if st['last_time'] is None else time-st['last_time']
    if (elapsed<0).any():raise ValueError('causal time reversed')
    dt=elapsed/model.scale;h=model.embed(mark)+model.gap(torch.stack((dt.log1p(),(dt>0).to(dt.dtype)),-1));layers=[]
    for layer,z in zip(model.layers,st['layers']):
        w=layer.write(h);g=layer.gate(h).sigmoid();decay=torch.exp(-layer.log_rate.exp()*dt[:,None]);angle=layer.freq*dt[:,None];cr=decay*angle.cos();ci=decay*angle.sin();r,i=z.chunk(2,-1)
        nr=cr*r-ci*i+w[:,:layer.n]*g;ni=ci*r+cr*i+w[:,layer.n:]*g;z=torch.cat((nr,ni),-1);layers.append(z)
        h=layer.norm1(h+layer.drop(layer.read(z)));h=layer.norm2(h+layer.drop(layer.mlp(h)))
    onehot=F.one_hot(mark,model.K).to(h.dtype);slots=st['slots']*torch.exp(-model.marks.log_rate.exp()*dt[:,None])[:,None,:]+onehot[:,:,None]*F.softplus(model.marks.value(h))[:,None,:]
    keys=None
    clocks=RaceTPP.clocks(model,h[:,None],slots[:,None])
    if hasattr(model,'key_w'):
        producer=h.detach() if model.local else h
        previous=torch.zeros_like(h) if st['last_mark'] is None else model.embed(st['last_mark'])
        if model.local:previous=previous.detach()
        kin=torch.cat((producer,previous),-1) if model.prev_msg else producer
        write=model.key_w(kin)
        if model.qk_norm:write=F.normalize(write,dim=-1)
        if key_delta is not None:
            if key_delta.shape!=write.shape:raise ValueError('write intervention shape')
            write=write+key_delta
        keys=st['keys']*torch.exp(-model.key_log_rate.exp()*dt[:,None])[:,None,:]+onehot[:,:,None]*write[:,None,:]
        query=model.query_w(producer)
        if model.qk_norm:query=F.normalize(query,dim=-1)*model.log_sharp.exp()
        bonus=torch.einsum('bd,bkd->bk',query,keys)
        if not model.qk_norm:bonus=bonus/math.sqrt(model.dk)
        clocks=(*clocks[:4],F.log_softmax(clocks[4]+bonus[:,None,None,:],-1))
    elif key_delta is not None:raise ValueError('model has no key-write port')
    nextstate=dict(layers=layers,slots=slots,keys=keys,last_time=time,last_mark=mark,parameter_version=parameter_version)
    forecast=dict(clocks=clocks,time=time,parameter_version=parameter_version)
    return forecast,nextstate

def revealed_log_probability(model,forecast,time,mark):
    gap=time-forecast['time']
    if (gap<0).any():raise ValueError('future event precedes forecast')
    params=forecast['clocks'];g=gap[:,None]
    _,s0=model.clock_terms(g,*params[:4]);_,s1=model.clock_terms(g+CELL,*params[:4]);hazard,_=model.clock_terms(g+CELL/2,*params[:4])
    selected=params[4].gather(-1,mark[:,None,None,None].expand(-1,1,model.M,1)).squeeze(-1)
    return (log_interval(s0.sum(-1),s1.sum(-1))+torch.logsumexp(hazard+selected,-1)-torch.logsumexp(hazard,-1)).squeeze(1)
