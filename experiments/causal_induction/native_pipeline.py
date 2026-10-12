"""B5/R1 causal sensor integration gate and private development fit; run_safe only."""
import argparse,copy,hashlib,json,math,resource,time
from pathlib import Path
import numpy as np
import torch
from native_sensor import Sensor,ROOT
from public_stream import records

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
      optimizer.zero_grad();torch.stack(losses).mean().backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step();losses=[];model.detach(st);updates+=1
  assert not st['issued']
  if losses:
   optimizer.zero_grad();torch.stack(losses).mean().backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step();model.detach(st);updates+=1
 return dict(nll=float(np.mean(scores)),targets=len(scores),wall_s=time.monotonic()-begin,optimizer_updates=updates,peak_occupied_nodes=st['peak_nodes'],**st['counts'])

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--tag',required=True);ap.add_argument('--mode',choices=['contract','fit'],required=True);a=ap.parse_args();torch.set_num_threads(1);out=ROOT/f'experiments/results/causal_induction/{a.tag}.json';assert not out.exists();parent=ROOT/'experiments/results/causal_induction/curie_causal_induction_20261012_v1_prepare.json';data=json.loads(parent.read_text());begin=time.monotonic();rows=[];checks={}
 def path(item):
  p=ROOT/item['path'];assert hashlib.sha256(p.read_bytes()).hexdigest()==item['sha256'];return p
 if a.mode=='contract':
  base=next(x for x in data['records'] if x['split']=='heldout_graph' and x['variant']=='base');rename=next(x for x in data['records'] if x['split']=='heldout_graph' and x['variant']=='relabel' and Path(x['path']).stem.split('_')[-2]==Path(base['path']).stem.split('_')[-2]);torch.manual_seed(91);m=Sensor();r0=run(m,path(base),'routed');r1=run(m,path(rename),'routed');assert abs(r0['nll']-r1['nll'])<1e-10,(r0,r1)
  opt=torch.optim.Adam(m.parameters(),lr=.001);before=m.core.units[0][0][0][0].input.weight.detach().clone();train=run(m,path(base),'routed',opt);change=float((before-m.core.units[0][0][0][0].input.weight).abs().max());assert change>0 and math.isfinite(train['nll']);checks=dict(address_relabel_equal_nll=True,query_before_feedback=True,causal_optimizer_boundaries=True,temporal_unit_parameter_change=change,training=train)
 else:
  train=[x for x in data['records'] if x['split']=='train'][:8];dev=[x for x in data['records'] if x['split']=='heldout_graph' and x['variant']=='base'][:4]
  for arm in ['self','routed']:
   torch.manual_seed(191);m=Sensor();opt=torch.optim.Adam(m.parameters(),lr=.001);initial=[run(m,path(x),arm,seed=900+i) for i,x in enumerate(dev)];acq=[]
   for epoch in range(2):
    for i,item in enumerate(train):acq.append(dict(epoch=epoch,**run(m,path(item),arm,opt,seed=191+epoch*100+i)))
   final=[run(m,path(x),arm,seed=900+i) for i,x in enumerate(dev)];ckpt=out.with_name(out.stem+'_'+arm+'.pt');torch.save(dict(model=m.state_dict(),optimizer=opt.state_dict()),ckpt);rows.append(dict(arm=arm,initial=initial,train=acq,dev=final,mean_initial_nll=float(np.mean([x['nll'] for x in initial])),mean_dev_nll=float(np.mean([x['nll'] for x in final])),whole_fit_wall_s=sum(x['wall_s'] for x in acq),checkpoint=str(ckpt.relative_to(ROOT)),checkpoint_sha256=hashlib.sha256(ckpt.read_bytes()).hexdigest()))
 sources=[Path(__file__),ROOT/'experiments/causal_induction/native_sensor.py',ROOT/'experiments/causal_induction/public_stream.py',ROOT/'sleeping_machines/addressed_event_heads.py',ROOT/'sleeping_machines/paired_route_credit.py',ROOT/'sleeping_machines/parallel_head_race_language.py',ROOT/'sleeping_machines/sparse_race_language.py'];result=dict(status='completed',battle='B5/R1',tag=a.tag,mode=a.mode,checks=checks,rows=rows,whole_pipeline_wall_s=time.monotonic()-begin,peak_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,parent_sha256=hashlib.sha256(parent.read_bytes()).hexdigest(),source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},scope='Integrated sensor interface: shared two-layer/two-head/two-receiver temporal core, addressed persistent node states; all receiver proposals during training, winner-only inference; up to8 recent source keys per query and one paid nonwinner sensor decode for conditional route teaching. Sources are selected only at sensor reads; inter-node event delivery and connected learned critic not yet implemented. No hidden graph/gate/oracle access. Eight TRAIN graphs/two passes, four separate development graphs. No depth advantage, credit replacement, public TEST or Transformer claim. Counts are executed operations, not full FLOP/energy estimates. Computational source contexts read at max physical/computational arrival without any future observed data.');out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(tag=a.tag,checks=checks,rows=[{k:x[k] for k in ['arm','mean_initial_nll','mean_dev_nll','whole_fit_wall_s']} for x in rows])),flush=True)
if __name__=='__main__':main()
