"""Stateful online Transformer: persist past-version KV, truncate credit not state.

With fixed weights, partitioned forward equals causal full attention. After
online updates, stored KV remain observations encoded by their producing weight
version, analogous to recurrent state. This is a declared online-memory policy,
not an assertion of equality to re-encoding every past event with new weights.
"""
import math
import torch
from b10_online_stream import forecast as ordinary

def forecast(model,t,m,state=None,transformer_context=2048):
    if model.family!='transformer':return ordinary(model,t,m,state,transformer_context)
    st={} if state is None else state;core=model.m;previous=st.get('last_time',t[:,:1]);dt=torch.diff(torch.cat((previous,t),1),dim=1)/core.scale;assert bool((dt>=0).all());origin=st.get('origin',t[:,:1]);count=st.get('count',0);positions=torch.arange(count,count+t.shape[1],dtype=t.dtype,device=t.device)[None];frequency=torch.exp(torch.arange(8,dtype=t.dtype,device=t.device)*(-math.log(10000)/8))
    def pe(z):
        a=z[:,:,None]*frequency;return torch.stack((a.sin(),a.cos()),-1).flatten(-2)
    x=core.embed(m)+core.gap(dt.log1p()[:,:,None])+torch.cat((pe((t-origin)/core.scale),pe(positions)),-1);cache=[]
    for depth,layer in enumerate(model.encoder):
        B,L,_=x.shape;q,k,v=[z.reshape(B,L,4,8).transpose(1,2) for z in layer.qkv(x).chunk(3,-1)]
        if 'layer_kv' in st:
            oldk,oldv=st['layer_kv'][depth];k=torch.cat((oldk,k),2);v=torch.cat((oldv,v),2)
        old=k.shape[2]-L;mask=torch.arange(k.shape[2],device=x.device)[None,:]>old+torch.arange(L,device=x.device)[:,None];a=(q@k.transpose(-1,-2)/math.sqrt(8)).masked_fill(mask,-math.inf).softmax(-1);h=layer.norm1(x+layer.output((a@v).transpose(1,2).reshape(B,L,32)));x=layer.norm2(h+layer.ff(h));cache.append((k[:,:,-transformer_context:],v[:,:,-transformer_context:]))
    slots=st.get('slots',x.new_zeros(1,core.K,core.marks.dv));values=torch.nn.functional.softplus(core.marks.value(x));decay=torch.exp(-core.marks.log_rate.exp()*dt[:,:,None]);onehot=torch.nn.functional.one_hot(m,core.K).to(x.dtype);outputs=[]
    for i in range(t.shape[1]):slots=slots*decay[:,i,None,:]+onehot[:,i,:,None]*values[:,i,None,:];outputs.append(slots)
    return core.clocks(x,torch.stack(outputs,1)),dict(layer_kv=cache,origin=origin,count=count+t.shape[1],slots=slots,last_time=t[:,-1:])
