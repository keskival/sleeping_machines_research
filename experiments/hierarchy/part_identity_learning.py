"""One integrated G-HIER acquisition:complete native parts + identity teaching.

No ablation/control acquisition. Reuse strongest acquired native parameters;
whole-label inference remains native. TEST is never read by the actor.
"""
import argparse,json,math,resource,sys,time,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from torch.nn import functional as F
from sleeping_machines.part_identity_composition import PartIdentityComposition,fit_shape_vocabulary
from sleeping_machines.convex_native_readout import binary_logits
from experiments.hierarchy.grounded_composition_learning import scene_forward,endpoint,PARENT
from experiments.hierarchy.frame_learning import atomic_save
from experiments.hierarchy.observed_training_pairs import training_pairs
from experiments.hierarchy.part_combination_task import corpus
from experiments.hierarchy.native_composition_data import sha,source_hashes
AUDIT='experiments/results/hierarchy/aws_g_hier_ownership_support_20261016_v1.json'
SOURCES=['experiments/hierarchy/part_identity_learning.py','sleeping_machines/part_identity_composition.py','sleeping_machines/observed_ownership_composition.py','sleeping_machines/bound_native_composition.py','experiments/hierarchy/grounded_composition_learning.py','experiments/hierarchy/observed_training_pairs.py','experiments/hierarchy/part_combination_task.py','sleeping_machines/convex_native_readout.py','tests/test_part_identity_composition.py']

def score(model,rows):
 tape=torch.get_rng_state();features=[];zs=[];shape_correct=shape_count=0;work={};model.eval()
 hook=model.head.register_forward_pre_hook(lambda module,args:features.append(args[0].detach().clone()))
 try:
  with torch.no_grad():
   for row in rows:
    torch.manual_seed(113);z,state,events=scene_forward(model,row);_,local=model.identity_loss(state,events)
    zs.append(z);shape_correct+=local['correct'];shape_count+=local['parts']
    for k,v in model.work(state).items():
     if isinstance(v,int):work[k]=work.get(k,0)+v
 finally:hook.remove();torch.set_rng_state(tape)
 return torch.stack(zs),torch.tensor([r.label for r in rows]),torch.stack(features),dict(correct=shape_correct,parts=shape_count,accuracy=shape_correct/shape_count),work

def backward_pair(model,examples):
 tape=torch.get_rng_state();next_tape=None;losses=[];identities=[];counts={}
 for j,row in enumerate(examples):
  if j:torch.set_rng_state(tape)
  z,state,events=scene_forward(model,row);whole=F.cross_entropy(z[None],torch.tensor([row.label]));identity,_=model.identity_loss(state,events)
  loss=whole+.25*identity;(loss/2).backward();losses.append(float(loss.detach()));identities.append(float(identity.detach()))
  for k,v in model.work(state).items():
   if isinstance(v,int):counts[k]=counts.get(k,0)+v
  if not j:next_tape=torch.get_rng_state()
 torch.set_rng_state(next_tape)
 return sum(losses)/2,sum(identities)/2,counts

def main():
 p=argparse.ArgumentParser();p.add_argument('--tag',required=True);p.add_argument('--mode',choices=['gate','fit'],required=True);p.add_argument('--steps',type=int,default=64);a=p.parse_args()
 torch.set_num_threads(1);began=time.monotonic();rows=corpus();train=[r for r in rows if r.split=='train'];cal=[r for r in train if r.nuisance==0];dev=[r for r in rows if r.split=='dev']
 threshold=json.loads((ROOT/AUDIT).read_text())['threshold'];vocab=fit_shape_vocabulary([r.model_input()[0] for r in train],threshold)
 assert len(vocab)==8,'record changed observable target vocabulary before acquisition'
 torch.manual_seed(1451);model=PartIdentityComposition(classes=2,ownership_threshold=threshold,shape_vocabulary=vocab)
 model.load_acquired(torch.load(PARENT,map_location='cpu',weights_only=True));opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01)
 pairs=training_pairs([r.model_input() for r in train]);order=torch.Generator().manual_seed(1453)
 prefix=ROOT/f'experiments/results/hierarchy/{a.tag}';checkpoint=prefix.with_suffix('.progress.pt');initial_features=prefix.with_suffix('.initial_features.npz')
 assert not prefix.with_suffix('.json').exists()
 sources=source_hashes();sources.update({n:sha(ROOT/n) for n in SOURCES});inputs={AUDIT:sha(ROOT/AUDIT),str(PARENT.relative_to(ROOT)):sha(PARENT)}
 vocab_hash=hashlib.sha256(repr(sorted(vocab.items())).encode()).hexdigest();settings=dict(steps=a.steps,threshold=threshold,vocab_sha256=vocab_hash,identity_weight=.25)
 if a.mode=='gate':
  model.train();opt.zero_grad(set_to_none=True);start=time.monotonic();loss,identity,work=backward_pair(model,[train[i] for i in pairs[0]])
  finite=math.isfinite(loss) and all(torch.isfinite(x.grad).all() for x in model.parameters() if x.grad is not None)
  gradients=[float(obj.core.content.weight.grad.norm()) for obj in model.objects];assert finite and all(g>0 for g in gradients)
  torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step();elapsed=time.monotonic()-start;rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
  passed=rss*1.25<3000000 and elapsed*64*1.5<1500
  r=dict(status='completed' if passed else 'failed',battle='G-HIER',stage='part_identity_gate',gate='PASS' if passed else 'STOP',paired_update_s=elapsed,max_rss_kb=rss,loss=loss,identity_nll=identity,content_gradient_norms=gradients,work=work,source_sha256=sources,input_sha256=inputs,settings=settings,scope='One actual native paired optimizer update,TRAIN only;resource/credit gate,no quality selection or ablation.')
  with prefix.with_suffix('.json').open('x') as f:json.dump(r,f,indent=2);f.write('\n')
  assert passed;print(json.dumps(dict(gate=r['gate'],paired_update_s=elapsed,max_rss_kb=rss)),flush=True);return
 start=0;history=[];work_train={};initial=None;past=0.
 if checkpoint.exists():
  c=torch.load(checkpoint,map_location='cpu',weights_only=False);assert c['settings']==settings and c['source_sha256']==sources and c['input_sha256']==inputs
  assert sha(initial_features)==c['initial_sha256'];model.load_state_dict(c['model']);opt.load_state_dict(c['optimizer']);torch.set_rng_state(c['rng']);order.set_state(c['order_rng'])
  start=c['step'];history=c['history'];work_train=c['work_training'];initial=c['initial'];past=c['wall_s']
 def save(step):
  atomic_save(checkpoint,dict(settings=settings,source_sha256=sources,input_sha256=inputs,initial_sha256=sha(initial_features),model=model.state_dict(),optimizer=opt.state_dict(),rng=torch.get_rng_state(),order_rng=order.get_state(),step=step,history=history,work_training=work_train,initial=initial,wall_s=past+time.monotonic()-began))
 if initial is None:
  z,y,x,sh,wt=score(model,cal);zd,yd,xd,shd,wd=score(model,dev);initial,_=endpoint(z,y,x,zd,yd,xd);initial.update(shape_train=sh,shape_dev=shd,work_train_score=wt,work_dev_score=wd)
  np.savez_compressed(initial_features,train=x.numpy(),dev=xd.numpy());save(0)
 for step in range(start,a.steps):
  i,j=pairs[int(torch.randint(len(pairs),(),generator=order))];model.train();opt.zero_grad(set_to_none=True)
  loss,identity,work=backward_pair(model,[train[i],train[j]]);assert math.isfinite(loss) and all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
  torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step();history.append(dict(loss=loss,identity_nll=identity))
  for k,v in work.items():work_train[k]=work_train.get(k,0)+v
  if (step+1)%8==0 or step+1==a.steps:save(step+1);print(json.dumps(dict(step=step+1,last8_identity_nll=sum(x['identity_nll'] for x in history[-8:])/len(history[-8:]))),flush=True)
 z,y,x,sh,wt=score(model,cal);zd,yd,xd,shd,wd=score(model,dev);final,fitted=endpoint(z,y,x,zd,yd,xd);final.update(shape_train=sh,shape_dev=shd)
 raw=prefix.with_suffix('.raw.pt');atomic_save(raw,model.state_dict())
 with torch.no_grad():
  model.head.weight.zero_();model.head.bias.zero_();model.head.weight[1].copy_(fitted['weight']);model.head.bias[1].copy_(fitted['bias'])
 final['materialized_dev_nll']=float(F.cross_entropy(F.linear(xd,model.head.weight,model.head.bias),yd))
 saved=prefix.with_suffix('.final.pt');atomic_save(saved,model.state_dict());features=prefix.with_name(prefix.name+'_features.npz')
 np.savez_compressed(features,final_train=x.numpy(),final_dev=xd.numpy(),train_labels=y.numpy(),dev_labels=yd.numpy())
 r=dict(status='completed',battle='G-HIER',stage='part_identity_learning',settings=settings,steps=a.steps,presentations=2*a.steps,parameters=sum(p.numel() for p in model.parameters()),initial=initial,final=final,history=history,work_training=work_train,work_final_train=wt,work_final_dev=wd,source_sha256=sources,input_sha256=inputs,final_weights=str(saved.relative_to(ROOT)),raw_weights=str(raw.relative_to(ROOT)),feature_packet=dict(path=str(features.relative_to(ROOT)),sha256=sha(features)),process_wall_s=past+time.monotonic()-began,max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
 scope='One integrated candidate,no ablation/control acquisition. Complete observed native parts,whole-label CE plus0.25TRAIN-observable intrinsic shape CE. Shape ids are targets only,not inference inputs or hidden generator ids.64paired updates/128presentations,TRAIN-only ridge.01 root head,reused64scene DEV,TEST excluded. Native actor retains temporal races/counterfactual learning/indexed state;ownership/query policy explicit. No online/persistent-cross-scene or semantic benchmark claim.',decision='Does learned part identity plus complete ownership improve held-out whole-label development quality over its initial actor and the saved acquired0.686609 reference? This is variant development,not a family verdict.')
 with prefix.with_suffix('.json').open('x') as f:json.dump(r,f,indent=2);f.write('\n')
 print(json.dumps(dict(dev_nll=final['materialized_dev_nll'],shape_dev_accuracy=shd['accuracy'])),flush=True)
if __name__=='__main__':main()
