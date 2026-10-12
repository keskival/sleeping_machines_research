"""Causal target learning with explicit capacity and historical resource receipt."""
import math,time
import numpy as np
import torch
from credit_run import clip
from public_stream import records

def run(model,path,arm,optimizer,seed,max_queries=None):
 st=model.fresh();g=torch.Generator().manual_seed(seed);torch.manual_seed(seed);scores=[];losses=[];begin=time.monotonic();updates=0;model.train();peak_bytes=0;peak_credit=0;peak_pairs=0;stopped=False
 for r in records(path):
  if r['kind']=='event':model.observe(r,st)
  elif r['kind']=='query':model.issue(r,st,g,arm,True)
  else:
   loss,p=model.feedback(r,st,True);scores.append(-math.log(max(1e-12,p if r['target'] else 1-p)));losses.append(loss)
   if len(losses)>=8 and not st['issued']:
    optimizer.zero_grad();torch.stack(losses).mean().backward();clip(model);optimizer.step();losses=[];model.detach(st);updates+=1
    states=list(st['nodes'].values())+list(st['credit_nodes'].values());peak_bytes=max(peak_bytes,sum(x.storage()['persistent_tensor_bytes'] for x in states));peak_credit=max(peak_credit,len(st['credit_nodes']));peak_pairs=max(peak_pairs,len(st['edges']))
   if max_queries is not None and len(scores)>=max_queries:
    assert not st['issued'];stopped=True;break
 assert not st['issued']
 if losses:
  optimizer.zero_grad();torch.stack(losses).mean().backward();clip(model);optimizer.step();model.detach(st);updates+=1
 return dict(nll=float(np.mean(scores)),targets=len(scores),first_half_nll=float(np.mean(scores[:64])),second_half_nll=float(np.mean(scores[64:])) if len(scores)>64 else None,wall_s=time.monotonic()-begin,optimizer_updates=updates,peak_occupied_nodes=st['peak_nodes'],peak_credit_relationships=peak_credit,peak_evidence_pairs=peak_pairs,peak_persistent_tensor_bytes=peak_bytes,scope_capacity='Live state tensors; graph/allocator/Python metadata excluded. Evidence pairs store four Python floats each; counted, not included in tensor bytes.',stopped_for_smoke=stopped,**st['counts'])
