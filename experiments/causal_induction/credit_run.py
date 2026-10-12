"""Same causal execution, with separate actor/critic clipping and paid work."""
import math,time
import numpy as np
import torch
from public_stream import records

def clip(model):
 actor=[p for name,p in model.named_parameters() if not name.startswith('critic.')]
 critic=[p for name,p in model.named_parameters() if name.startswith('critic.')]
 torch.nn.utils.clip_grad_norm_(actor,1.)
 if critic:torch.nn.utils.clip_grad_norm_(critic,1.)

def run(model,path,arm,optimizer=None,seed=71):
 st=model.fresh();g=torch.Generator().manual_seed(seed);torch.manual_seed(seed);scores=[];losses=[];begin=time.monotonic();updates=0;model.train(optimizer is not None)
 # The entire stream is causal. A batch closes only after feedback and no pending queries.
 with torch.set_grad_enabled(optimizer is not None):
  for r in records(path):
   if r['kind']=='event':model.observe(r,st)
   elif r['kind']=='query':model.issue(r,st,g,arm,optimizer is not None)
   else:
    loss,p=model.feedback(r,st,optimizer is not None);scores.append(-math.log(max(1e-12,p if r['target'] else 1-p)))
    if optimizer is not None:
     losses.append(loss)
     if len(losses)>=8 and not st['issued']:
      optimizer.zero_grad();torch.stack(losses).mean().backward();clip(model);optimizer.step();losses=[];model.detach(st);updates+=1
  assert not st['issued']
  if losses:
   optimizer.zero_grad();torch.stack(losses).mean().backward();clip(model);optimizer.step();model.detach(st);updates+=1
 return dict(nll=float(np.mean(scores)),targets=len(scores),wall_s=time.monotonic()-begin,optimizer_updates=updates,peak_occupied_nodes=st['peak_nodes'],**st['counts'])
