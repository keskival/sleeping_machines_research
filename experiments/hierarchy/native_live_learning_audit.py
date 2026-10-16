"""R1 read-only checkpoint audit plus one discarded TRAIN learner update.

Does not edit the running model/checkpoint,score DEV/TEST,or train a comparator.
"""
import argparse,hashlib,io,json,math,resource,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT),str(ROOT/'experiments')]
import numpy as np
import torch
from torch.nn import functional as F
from r1_cerl import load_fineweb
from r1_local_native_deep import LocalNativeDeepLM
from experiments.native_query_migration import migrate


def summary(x):
 x=torch.as_tensor(x).detach().double().flatten()
 return dict(n=x.numel(),finite=int(torch.isfinite(x).sum()),minimum=float(x.min()),median=float(x.median()),maximum=float(x.max()),rms=float(x.square().mean().sqrt()))


def main():
 p=argparse.ArgumentParser();p.add_argument('--tag',required=True);a=p.parse_args();torch.set_num_threads(1);began=time.monotonic()
 checkpoint=ROOT/'experiments/results/token_language/aws_r1_queryable_native_4m_s0_20261016_v1.resume.pt'
 raw=checkpoint.read_bytes();digest=hashlib.sha256(raw).hexdigest();saved=torch.load(io.BytesIO(raw),map_location='cpu',weights_only=False);del raw
 weights=saved['model'];model=LocalNativeDeepLM(50257,weights['causes.class_of'],weights['causes.class_words'],width=64,depth=8,heads=4,pool=16,concept_units=4096,topk=4,explore=2,dk=32,packed=True)
 opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01)
 model,opt,state,_=migrate(model,opt,saved['state'],publication_capacity=16)
 model.load_state_dict(weights);opt.load_state_dict(saved['optimizer']);g=torch.Generator();g.set_state(saved['g']);alt=torch.Generator();alt.set_state(saved['alt']);torch.set_rng_state(saved['rng'])
 cursor=saved['cursor'];del weights,saved
 data=load_fineweb(ROOT/'data/fineweb_gpt2/fineweb_train_000001.bin');tokens=torch.from_numpy(np.asarray(data[cursor:cursor+33]).astype(np.int64))
 assert state['events']==cursor and state['expected_next_input']==int(tokens[0])
 query_scores=[];queries={};controls=[];head_scores=[];cause_parts=[];causal_reads=[];handles=[]
 for d,module in enumerate(model.core.memory_queries):
  handles.append(module.register_forward_hook(lambda m,args,out,d=d:queries.__setitem__(d,out.detach().reshape(2,-1))))
 for d,module in enumerate(model.core.memory_keys):
  def capture(m,args,out,d=d):query_scores.append((d,(queries[d]@out.detach().T/math.sqrt(model.core.payload)).flatten()))
  handles.append(module.register_forward_hook(capture))
 original_proposal=model.core.proposal
 def proposal(old,params,incoming,elapsed,*,traces):
  if traces:
   c=torch.einsum('hcp,hp->hc',params['control_w'],F.layer_norm(incoming,(model.core.payload,)))+params['control_b']
   controls.append(c.detach())
  return original_proposal(old,params,incoming,elapsed,traces=traces)
 model.core.proposal=proposal
 original_read=model.core.read_publication
 def read(state,key,now,packets,*,learn=True):
  causal_reads.append(float(now.detach())-float(state['bindings'].entries[key].available))
  return original_read(state,key,now,packets,learn=learn)
 model.core.read_publication=read
 before={n:p.detach().clone() for n,p in model.named_parameters()};opt.zero_grad(set_to_none=True)
 nll,obj,_,packets,report=model.stream_nll(tokens[:-1],tokens[1:],state,g,alt,stats=True)
 obj.backward();model.deliver_credit(packets,state)
 groups={};parameters=[]
 for name,param in model.named_parameters():
  key='.'.join(name.split('.')[:2]);item=groups.setdefault(key,dict(parameters=0,with_gradient=0,gradient_sq=0.,weight_sq=0.))
  item['parameters']+=param.numel();item['weight_sq']+=float(param.detach().double().square().sum())
  if param.grad is not None:
   item['with_gradient']+=param.numel();item['gradient_sq']+=float(param.grad.detach().double().square().sum())
  parameters.append(dict(name=name,has_gradient=param.grad is not None,gradient_norm=0. if param.grad is None else float(param.grad.norm()),weight_norm=float(param.detach().norm())))
 norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.);assert torch.isfinite(norm)
 assert all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in model.parameters())
 opt.step()
 for row in parameters:
  name=row['name'];value=dict(model.named_parameters())[name].detach();row['update_norm']=float((value-before[name]).norm());row['relative_update']=row['update_norm']/max(row['weight_norm'],1e-12)
 raw_scores=torch.cat([v for d,v in query_scores]);c=torch.cat(controls)
 concerns=[]
 if float(norm)>100:concerns.append('Global clipping scales all gradients by less than1/100 on this update;inspect trace/core dominance before changing policy.')
 saturated=float((raw_scores.abs()>=12).double().mean())
 if saturated>.05:concerns.append('More than5%query logits hit the hardclamp;those entries lose local logit sensitivity outside it.')
 dead=[r['name'] for r in parameters if (r['name'].startswith(('core.assemblers.','core.memory_queries.','core.memory_keys.'))) and (not r['has_gradient'] or r['gradient_norm']==0)]
 if dead:concerns.append('Some query/assembly tensors have no task credit: '+str(dead))
 for group,item in groups.items():
  item['gradient_norm']=item.pop('gradient_sq')**.5;item['weight_norm']=item.pop('weight_sq')**.5
 result=dict(status='completed',battle='R1',stage='native_live_learning_audit',checkpoint_sha256=digest,checkpoint_cursor=cursor,discarded_train_targets=32,
  nll=float(nll.detach().mean()),objective=float(obj.detach()),preclip_gradient_norm=float(norm),global_clip_multiplier=min(1.,1./(float(norm)+1e-6)),groups=groups,parameters=parameters,
  query_raw_logits=summary(raw_scores),query_clamped_fraction=saturated,receiver_forget_controls=summary(c[:,0]),receiver_write_controls=summary(c[:,1]),receiver_write_sigmoid=summary(c[:,1].sigmoid()),
  causal_read_age=summary(causal_reads),native_report=report,concerns=concerns,storage=model.core.storage(state),max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,wall_s=time.monotonic()-began,
  scope='One acquiredcheckpoint snapshot/one discarded32targetTRAINupdate;running learner/checkpoint untouched.No DEV/TEST,ablation,tuningor architecture verdict. Gradients/updates describe this batch,not an entiretrajectory. Current maps and clocks/trace credit unchanged.')
 out=ROOT/f'experiments/results/diagnostics/{a.tag}.json';assert not out.exists();out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(cursor=cursor,preclip_gradient_norm=float(norm),query_clamped_fraction=saturated,concerns=concerns)),flush=True)
if __name__=='__main__':main()
