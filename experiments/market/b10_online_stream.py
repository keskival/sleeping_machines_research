"""Continuous causal native/LSTM state and rolling-history Transformer forecasts.

Numeric state persists across optimizer updates. Gradient detaches do not reset
memory. All models retain the same addressed mark-output memory.
"""
import math
import torch
from torch.nn import functional as F
from b10_depth_tpp_v1 import CELL,log_interval

def detach(value):
    if torch.is_tensor(value):return value.detach()
    if isinstance(value,dict):return {k:detach(v) for k,v in value.items()}
    if isinstance(value,list):return [detach(v) for v in value]
    if isinstance(value,tuple):return tuple(detach(v) for v in value)
    return value

def forecast(model,t,m,state=None,transformer_context=1024):
    """Issue parameters from observed inputs ONLY; future labels are no argument."""
    assert t.ndim==m.ndim==2 and t.shape==m.shape and t.shape[0]==1
    st={} if state is None else state;core=model.m
    previous=st.get('last_time',t[:,:1]);dt=torch.diff(torch.cat((previous,t),1),dim=1)/core.scale
    assert bool((dt>=0).all());x=core.embed(m)
    if model.family=='native':
        x=x+core.gap(torch.stack((dt.log1p(),(dt>0).to(dt.dtype)),-1));new=[]
        for depth,layer in enumerate(core.layers):
            z=st.get('layers',[None]*len(core.layers))[depth]
            if z is None:z=x.new_zeros(1,2*layer.n)
            writes=layer.write(x);gates=layer.gate(x).sigmoid();decay=torch.exp(-layer.log_rate.exp()*dt[:,:,None]);angle=layer.freq*dt[:,:,None];cr=decay*angle.cos();ci=decay*angle.sin();out=[]
            for i in range(x.shape[1]):
                r,j=z.chunk(2,-1);w=writes[:,i]*gates[:,i].repeat(1,2);z=torch.cat((cr[:,i]*r-ci[:,i]*j+w[:,:layer.n],ci[:,i]*r+cr[:,i]*j+w[:,layer.n:]),-1);out.append(z)
            values=torch.stack(out,1);new.append(z);x=layer.norm1(x+layer.drop(layer.read(values)));x=layer.norm2(x+layer.drop(layer.mlp(x)))
        st={**st,'layers':new}
    elif model.family=='lstm':
        x=x+core.gap(dt.log1p()[:,:,None]);new=[]
        for depth,(a,b) in enumerate(zip(model.encoder.input,model.encoder.hidden)):
            h,c=st.get('lstm',[(None,None)]*2)[depth]
            if h is None:h=x.new_zeros(1,32);c=torch.zeros_like(h)
            projected=a(x);out=[]
            for i in range(x.shape[1]):
                ig,fg,cg,og=(projected[:,i]+b(h)).chunk(4,-1);c=fg.sigmoid()*c+ig.sigmoid()*cg.tanh();h=og.sigmoid()*c.tanh();out.append(h)
            x=torch.stack(out,1);new.append((h,c))
        st={**st,'lstm':new}
    elif model.family=='transformer':
        # Raw observed inputs survive weight updates; keys/values are recomputed
        # under the current weights, so an old-parameter KV cache is never reused.
        origin=st.get('origin',t[:,:1]);count=st.get('count',0);positions=torch.arange(count,count+t.shape[1],dtype=t.dtype,device=t.device)[None]
        ht=torch.cat((st.get('history_t',t[:,:0]),t),1);hm=torch.cat((st.get('history_m',m[:,:0]),m),1);hd=torch.cat((st.get('history_dt',dt[:,:0]),dt),1);hp=torch.cat((st.get('history_position',positions[:,:0]),positions),1)
        x=core.embed(hm)+core.gap(hd.log1p()[:,:,None]);frequency=torch.exp(torch.arange(8,dtype=t.dtype,device=t.device)*(-math.log(10000)/8))
        def pe(z):
            a=z[:,:,None]*frequency;return torch.stack((a.sin(),a.cos()),-1).flatten(-2)
        x=x+torch.cat((pe((ht-origin)/core.scale),pe(hp)),-1);x=model.encoder(x)[:,-t.shape[1]:]
        st={**st,'origin':origin,'count':count+t.shape[1],'history_t':ht[:,-transformer_context:],'history_m':hm[:,-transformer_context:],'history_dt':hd[:,-transformer_context:],'history_position':hp[:,-transformer_context:]}
    else:raise ValueError(model.family)
    slots=st.get('slots',x.new_zeros(1,core.K,core.marks.dv));values=F.softplus(core.marks.value(x));decay=torch.exp(-core.marks.log_rate.exp()*dt[:,:,None]);onehot=F.one_hot(m,core.K).to(x.dtype);out=[]
    for i in range(t.shape[1]):
        slots=slots*decay[:,i,None,:]+onehot[:,i,:,None]*values[:,i,None,:];out.append(slots)
    parameters=core.clocks(x,torch.stack(out,1));return parameters,{**st,'slots':slots,'last_time':t[:,-1:]}

def revealed_terms(model,parameters,issued_times,next_times,next_marks):
    g=(next_times-issued_times).clamp_min(0);core=model.m
    _,s0=core.clock_terms(g,*parameters[:4]);_,s1=core.clock_terms(g+CELL,*parameters[:4]);hazard,_=core.clock_terms(g+CELL/2,*parameters[:4]);selected=parameters[4].gather(-1,next_marks[:,:,None,None].expand(-1,-1,core.M,1)).squeeze(-1)
    return log_interval(s0.sum(-1),s1.sum(-1))+torch.logsumexp(hazard+selected,-1)-torch.logsumexp(hazard,-1)
