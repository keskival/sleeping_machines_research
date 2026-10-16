"""Fixed-architecture data/training scale test,continued from the acquired learner.

No ablation arm,no model redesign,no hyperparameter search. Original protocol
and broader TRAIN-only readout are reported separately. TEST is never scored.
"""
import argparse,copy,io,json,math,resource,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from torch.nn import functional as F
from experiments.hierarchy.identity_role_learning import score,backward_pair,SOURCES,AUDIT
from experiments.hierarchy.composition_scaling_data import training_corpus,fresh_dev_corpus,broad_calibration,audit_corpus
from experiments.hierarchy.part_combination_task import corpus
from experiments.hierarchy.frame_learning import atomic_save
from experiments.hierarchy.grounded_composition_learning import endpoint
from experiments.hierarchy.observed_training_pairs import training_pairs
from experiments.hierarchy.native_composition_data import sha,source_hashes
from sleeping_machines.part_identity_composition import fit_shape_vocabulary
from sleeping_machines.identity_role_composition import IdentityRoleComposition
PARENT=ROOT/'experiments/results/hierarchy/aws_g_hier_identity_role_20261016_v1.progress.pt'
PARENT_RESULT=ROOT/'experiments/results/hierarchy/aws_g_hier_identity_role_20261016_v1.json'
PARENT_FEATURES=ROOT/'experiments/results/hierarchy/aws_g_hier_identity_role_20261016_v1_features.npz'
MILESTONES=(128,512,2048)


class PairSchedule:
 def __init__(self,count,seed=6101):
  if count<1:raise ValueError('positive pair count required')
  self.count=count;self.generator=torch.Generator().manual_seed(seed);self.order=[];self.cursor=0;self.epochs=0
 def next(self):
  if self.cursor==len(self.order):
   self.order=torch.randperm(self.count,generator=self.generator).tolist();self.cursor=0;self.epochs+=1
  answer=self.order[self.cursor];self.cursor+=1;return answer
 def state_dict(self):return dict(count=self.count,generator=self.generator.get_state(),order=self.order.copy(),cursor=self.cursor,epochs=self.epochs)
 def load_state_dict(self,s):
  assert s['count']==self.count and sorted(s['order'])==list(range(self.count)) and 0<=s['cursor']<=self.count
  self.generator.set_state(s['generator']);self.order=s['order'].copy();self.cursor=s['cursor'];self.epochs=s['epochs']


def update(model,opt,pair):
 model.train();opt.zero_grad(set_to_none=True);loss,identity,work=backward_pair(model,pair)
 assert math.isfinite(loss) and all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
 gradients=[float(obj.core.content.weight.grad.norm()) for obj in model.objects]
 torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
 return dict(loss=loss,identity_nll=identity,content_gradient_norms=gradients),work


def main():
 p=argparse.ArgumentParser();p.add_argument('--tag',required=True);p.add_argument('--mode',choices=['gate','fit'],required=True);a=p.parse_args()
 torch.set_num_threads(1);began=time.monotonic();rows=training_corpus();audit=audit_corpus(rows);pairs=training_pairs([r.model_input() for r in rows]);assert len(pairs)==1024
 old=corpus();cal=[r for r in old if r.split=='train' and r.nuisance==0];dev=[r for r in old if r.split=='dev'];broad=broad_calibration(rows)
 threshold=json.loads((ROOT/AUDIT).read_text())['threshold'];vocab=fit_shape_vocabulary([r.events for r in old if r.split=='train'],threshold)
 assert fit_shape_vocabulary([r.events for r in rows],threshold)==vocab
 parent=torch.load(PARENT,map_location='cpu',weights_only=False);assert parent['step']==64
 assert all(sha(ROOT/n)==h for n,h in parent['source_sha256'].items()),'parent source changed;do not silently migrate'
 torch.manual_seed(1451);model=IdentityRoleComposition(classes=2,ownership_threshold=threshold,shape_vocabulary=vocab);model.load_state_dict(parent['model'])
 opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01);opt.load_state_dict(parent['optimizer']);torch.set_rng_state(parent['rng']);schedule=PairSchedule(len(pairs))
 prefix=ROOT/f'experiments/results/hierarchy/{a.tag}';progress=prefix.with_suffix('.progress.pt');assert not prefix.with_suffix('.json').exists()
 sources=source_hashes();sources.update({n:sha(ROOT/n) for n in SOURCES+['experiments/hierarchy/composition_scale_learning.py','experiments/hierarchy/composition_scaling_data.py','tests/test_composition_scaling_data.py','tests/test_composition_scale_schedule.py']})
 inputs={str(p.relative_to(ROOT)):sha(p) for p in [PARENT,PARENT_RESULT,PARENT_FEATURES,ROOT/AUDIT]}
 settings=dict(additional_updates=2048,additional_presentations=4096,epochs=2,pairs_per_epoch=1024,identity_weight=.25,lr=.001,weight_decay=.01,milestones=list(MILESTONES),data=audit)
 if a.mode=='gate':
  pair=[rows[i] for i in pairs[0]];start=time.monotonic();record,work=update(model,opt,pair);elapsed=time.monotonic()-start
  assert all(g>0 and math.isfinite(g) for g in record['content_gradient_norms'])
  snapshot=copy.deepcopy(dict(model=model.state_dict(),optimizer=opt.state_dict(),rng=torch.get_rng_state()))
  update(model,opt,pair);expected={k:v.clone() for k,v in model.state_dict().items()}
  model.load_state_dict(snapshot['model']);opt.load_state_dict(snapshot['optimizer']);torch.set_rng_state(snapshot['rng']);update(model,opt,pair)
  exact=all(torch.equal(v,model.state_dict()[k]) for k,v in expected.items());assert exact
  rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss;estimate=elapsed*2048*1.35+1800;passed=rss*1.3<3000000 and estimate<22000
  result=dict(status='completed' if passed else 'failed',battle='G-HIER',stage='composition_scale_gate',gate='PASS' if passed else 'STOP',paired_update_s=elapsed,projected_fit_s=estimate,max_rss_kb=rss,exact_optimizer_resume=exact,data=audit,record=record,work=work,source_sha256=sources,input_sha256=inputs,settings=settings,scope='TRAIN-only integrated optimizer/resource/recovery gate;not quality selection.')
  prefix.with_suffix('.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:result[k] for k in ['gate','paired_update_s','projected_fit_s','max_rss_kb','exact_optimizer_resume']}),flush=True);assert passed;return
 start=0;history=[];milestones=[];work_train={};work_eval={};seen=set();past=0.;initial_broad=None
 baseline=json.loads(PARENT_RESULT.read_text())['final']
 with np.load(PARENT_FEATURES) as f:
  initial_x=torch.as_tensor(f['final_train'].copy());initial_xd=torch.as_tensor(f['final_dev'].copy());labels=torch.as_tensor(f['train_labels'].copy());dev_labels=torch.as_tensor(f['dev_labels'].copy())
 if progress.exists():
  saved=torch.load(progress,map_location='cpu',weights_only=False)
  assert saved['settings']==settings and saved['source_sha256']==sources and saved['input_sha256']==inputs
  model.load_state_dict(saved['model']);opt.load_state_dict(saved['optimizer']);torch.set_rng_state(saved['rng']);schedule.load_state_dict(saved['schedule'])
  start=saved['step'];history=saved['history'];milestones=saved['milestones'];work_train=saved['work_training'];work_eval=saved['work_evaluation'];seen=set(saved['seen_pairs']);past=saved['wall_s'];initial_broad=saved['initial_broad']
 def add_work(destination,counts):
  for k,v in counts.items():
   if isinstance(v,int):destination[k]=destination.get(k,0)+v
 def save(step):
  atomic_save(progress,dict(step=step,model=model.state_dict(),optimizer=opt.state_dict(),rng=torch.get_rng_state(),schedule=schedule.state_dict(),settings=settings,source_sha256=sources,input_sha256=inputs,history=history,milestones=milestones,work_training=work_train,work_evaluation=work_eval,seen_pairs=sorted(seen),initial_broad=initial_broad,wall_s=past+time.monotonic()-began))
 if initial_broad is None:
  z,y,x,sh,work=score(model,broad);add_work(work_eval,work)
  initial_broad,_=endpoint(torch.zeros(len(cal)+len(broad),2),torch.cat((labels,y)),torch.cat((initial_x,x)),torch.zeros(len(dev),2),dev_labels,initial_xd)
  initial_broad=dict(train=initial_broad['primary_train'],dev=initial_broad['primary_dev'],shape=sh);save(0)
 for step in range(start,2048):
  k=schedule.next();seen.add(k);record,counts=update(model,opt,[rows[i] for i in pairs[k]]);history.append({k:v for k,v in record.items() if k!='content_gradient_norms'});add_work(work_train,counts)
  if (step+1)%8==0:save(step+1)
  if (step+1)%32==0:print(json.dumps(dict(step=step+1,of=2048,unique_pairs=len(seen),epoch=schedule.epochs,whole_plus_identity_last32=sum(r['loss'] for r in history[-32:])/32,identity_last32=sum(r['identity_nll'] for r in history[-32:])/32,elapsed_s=past+time.monotonic()-began)),flush=True)
  if step+1 in MILESTONES:
   z,y,x,sh,work=score(model,cal);add_work(work_eval,work);zd,yd,xd,shd,work=score(model,dev);add_work(work_eval,work);zb,yb,xb,shb,work=score(model,broad);add_work(work_eval,work)
   legacy,fit=endpoint(z,y,x,zd,yd,xd);expanded,bfit=endpoint(torch.cat((z,zb)),torch.cat((y,yb)),torch.cat((x,xb)),zd,yd,xd)
   row=dict(step=step+1,unique_training_examples=2*len(seen),legacy=legacy,broad_train_readout=expanded,shape_cal=sh,shape_dev=shd,shape_broad=shb)
   milestones.append(row);atomic_save(prefix.with_name(prefix.name+f'.step{step+1}.pt'),dict(model=model.state_dict(),optimizer=opt.state_dict(),metrics=row))
   prefix.with_suffix('.partial.json').write_text(json.dumps(dict(status='running',stage='composition_scaling',settings=settings,milestones=milestones,baseline=baseline,initial_broad=initial_broad,source_sha256=sources,input_sha256=inputs),indent=2)+'\n');save(step+1)
   print(json.dumps(dict(milestone=step+1,legacy_dev_nll=legacy['primary_dev']['nll'],broad_dev_nll=expanded['primary_dev']['nll'],shape_dev_accuracy=shd['accuracy'])),flush=True)
 # Only the preregistered final checkpoint accesses new DEV tapes. No TEST rows.
 zf,yf,xf,shf,work=score(model,fresh_dev_corpus());add_work(work_eval,work)
 fresh_legacy,_=endpoint(z,y,x,zf,yf,xf);fresh_broad,_=endpoint(torch.cat((z,zb)),torch.cat((y,yb)),torch.cat((x,xb)),zf,yf,xf)
 raw=prefix.with_suffix('.raw.pt');atomic_save(raw,model.state_dict())
 with torch.no_grad():
  model.head.weight.zero_();model.head.bias.zero_();model.head.weight[1].copy_(fit['weight']);model.head.bias[1].copy_(fit['bias'])
 final=prefix.with_suffix('.final.pt');atomic_save(final,model.state_dict())
 nll=float(F.cross_entropy(F.linear(xd,model.head.weight,model.head.bias),yd).detach())
 result=dict(status='completed',battle='G-HIER',stage='composition_scaling',settings=settings,parent_updates=64,additional_updates=2048,total_presentations=4096,parameters=sum(p.numel() for p in model.parameters()),baseline=baseline,initial_broad=initial_broad,milestones=milestones,final=dict(materialized_dev_nll=nll,legacy=legacy,broad_train_readout=expanded,shape_dev=shd,fresh_dev_legacy=fresh_legacy,fresh_dev_broad=fresh_broad,fresh_shape=shf),work_training=work_train,work_evaluation=work_eval,process_wall_s=past+time.monotonic()-began,max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,source_sha256=sources,input_sha256=inputs,raw_weights=str(raw.relative_to(ROOT)),final_weights=str(final.relative_to(ROOT)),scope='One fixed architecture/optimizer/objective continuation from exact64update learner.2048additional paired updates,4096presentations,2048unique broader TRAIN scenes,two complete pair epochs.Original64TRAIN topology ceiling unchanged;32fresh event/metric-layout tapes.Original64scene DEV reused;new64scene DEV tapes accessed only at final checkpoint;TEST excluded.Legacy TRAIN64 ridge.01 readout comparable to saved0.686609;broader TRAIN128 readout reported separately.No ablation,hyperparameter sweep,public benchmark,iso-compute or hardware claim.',decision='Final legacy DEV loss versus0.686609;report both whole-object learning curves and local identity. Strong composition milestone additionally requires at least80% accuracy and NLL below0.5 on both old and fresh DEV. No automatic architecture change or further acquisition.')
 prefix.with_suffix('.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(status='completed',dev_nll=nll,fresh_dev_nll=fresh_legacy['primary_dev']['nll'])),flush=True)
if __name__=='__main__':main()
