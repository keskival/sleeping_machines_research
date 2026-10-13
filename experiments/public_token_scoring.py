"""R1 normalized token scorer with the pinned public reference's window/target policy."""
from types import FunctionType
import numpy as np
import torch

def windows(tokens,start,targets,length=1024,lanes=8):
    if start<0 or targets<=0 or targets%length or start+targets>=len(tokens):raise ValueError('Complete disjoint windows plus the final target required')
    for begin in range(start,start+targets,length*lanes):
        count=min(lanes,(start+targets-begin)//length)
        ids=torch.stack([torch.from_numpy(np.array(tokens[pos:pos+length+1],dtype=np.int64)) for pos in range(begin,begin+count*length,length)])
        yield ids[:,:-1],ids[:,1:],begin

def reference_forward(model,driver):
    # Reference Transformer has a causal1024-token window and no EOS attention
    # reset. Preserve all native numeric paths inside each such window.
    original=driver.Model.forward;namespace=dict(original.__globals__);real=namespace['token_features']
    def features(*args,**kwargs):kwargs['eos']=None;return real(*args,**kwargs)
    namespace['token_features']=features
    fn=FunctionType(original.__code__,namespace,original.__name__,original.__defaults__,original.__closure__);fn.__kwdefaults__=original.__kwdefaults__
    return fn.__get__(model,type(model))

@torch.no_grad()
def score(model,tokens,args,driver,start,targets,length=1024,lanes=8):
    model.eval();forward=reference_forward(model,driver);gen=torch.Generator().manual_seed(args.seed+100000)
    total=0.;count=0
    for inputs,labels,begin in windows(tokens,start,targets,length,lanes):
        state=None
        for pos in range(0,length,args.chunk):
            x,state,_=forward(inputs[:,pos:pos+args.chunk],state,gen,args,deterministic=args.deterministic_eval)
            losses=model.readout.nll(x,labels[:,pos:pos+args.chunk]);total+=float(losses.double().sum());count+=losses.numel()
    assert count==targets
    return dict(nll=total/count,loss_sum=total,targets=count,target_start=start+1,target_end_inclusive=start+targets,window=length,lanes=lanes,eos_state_reset=False)
