"""Carried-state development scoring; no fixed context ceiling.

Keep public-reference window scoring separately. Warmup precedes the identical
scored interval; lane layout, EOS policy and warmup are explicit in the result.
"""
import torch
import numpy as np

@torch.no_grad()
def score(model,tokens,args,driver,start,targets,warmup=0,lanes=1,eos_reset=False):
    if min(start,warmup)<0 or targets<=0 or targets%lanes or start<warmup or start+targets>=len(tokens):
        raise ValueError('Invalid causal scored interval')
    length=targets//lanes;total=0.;count=0;was_training=model.training;model.eval()
    try:
        for lane in range(lanes):
            begin=start+lane*length;state=None;gen=torch.Generator().manual_seed(args.seed+100000+lane)
            from types import SimpleNamespace
            policy=SimpleNamespace(**vars(args));policy.eos_reset=eos_reset
            for pos in range(begin-warmup,begin+length,args.chunk):
                end=min(pos+args.chunk,begin+length)
                ids=torch.from_numpy(np.array(tokens[pos:end],dtype=np.int64)).reshape(1,-1)
                features,state,_=model(ids,state,gen,policy,deterministic=args.deterministic_eval)
                lo=max(begin-pos,0)
                if lo<end-pos:
                    labels=torch.from_numpy(np.array(tokens[pos+lo+1:end+1],dtype=np.int64)).reshape(1,-1)
                    losses=model.readout.nll(features[:,lo:],labels);total+=float(losses.double().sum());count+=losses.numel()
        assert count==targets
        return dict(nll=total/count,targets=count,target_start=start+1,target_end_inclusive=start+targets,warmup=warmup,lanes=lanes,eos_state_reset=eos_reset,policy='continuous carried state; no periodic context reset')
    finally:model.train(was_training)
