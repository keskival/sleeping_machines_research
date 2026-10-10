"""Explicit native keyed-token episode state with lazy addressed key transport.
Output is full normalized GPT-2 vocabulary; dense output work remains charged.
"""
import math
import torch

def detach_state(value):
    if isinstance(value,torch.Tensor):return value.detach()
    if isinstance(value,dict):return {k:detach_state(v) for k,v in value.items()}
    if isinstance(value,list):return [detach_state(v) for v in value]
    if isinstance(value,tuple):return tuple(detach_state(v) for v in value)
    return value

def consume(model,token,state=None,parameter_version=0,reset=False,key_delta=None):
    if token.ndim!=1 or len(token)!=1 or token.dtype!=torch.long:raise ValueError('one contiguous document lane required')
    if state is None or reset:state=dict(layers=[model.embed.weight.new_zeros(1,2*l.n) for l in model.layers],keys={},position=0,previous=None,parameter_version=None)
    h=model.embed(token);layers=[]
    for layer,z in zip(model.layers,state['layers']):
        w=layer.write(h);g=layer.gate(h).sigmoid();decay=torch.exp(-layer.log_rate.exp());cr=decay*layer.freq.cos();ci=decay*layer.freq.sin();r,i=z.chunk(2,-1);z=torch.cat((cr*r-ci*i+w[:,:layer.n]*g,ci*r+cr*i+w[:,layer.n:]*g),-1);layers.append(z)
        h=layer.norm1(h+layer.drop(layer.read(z)));h=layer.norm2(h+layer.drop(layer.mlp(h)))
    h=model.out_norm(h);logits=h@model.embed.weight.T+model.bias;keys=dict(state['keys']);position=state['position'];address=int(token[0]);rates=model.log_rate.exp() if model.keyed else None
    if model.keyed:
        producer=h.detach() if model.keyed==2 else h
        previous=torch.zeros_like(h) if state['previous'] is None else model.embed(state['previous'])
        if model.keyed==2:previous=previous.detach()
        write=model.key_w(torch.cat((producer,previous),-1)).reshape(model.heads,model.dk)
        if key_delta is not None:
            if key_delta.shape!=write.shape:raise ValueError('key intervention shape')
            write=write+key_delta
        if address in keys:
            value,when=keys[address];write=write+value*torch.exp(-rates*(position-when))[:,None]
        keys[address]=(write,position)
        addresses=list(keys);transported=torch.stack([v*torch.exp(-rates*(position-t))[:,None] for v,t in keys.values()]);query=model.query_w(producer).reshape(model.heads,model.dk)
        scores=torch.einsum('uhd,hd->u',transported,query)/math.sqrt(model.dk)
        logits=logits.scatter_add(1,torch.tensor(addresses,dtype=torch.long)[None],scores[None])
    elif key_delta is not None:raise ValueError('no key port')
    nextstate=dict(layers=layers,keys=keys,position=position+1,previous=token,parameter_version=parameter_version)
    # Cache the issued normalized distribution before any parameter update.
    forecast=dict(log_probabilities=logits.log_softmax(-1),parameter_version=parameter_version,position=position)
    return forecast,nextstate

def revealed_nll(forecast,target):
    return -forecast['log_probabilities'].gather(1,target[:,None]).squeeze(1)
