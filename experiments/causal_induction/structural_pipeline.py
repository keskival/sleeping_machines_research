"""Diagnosed native improvement: acquired persistence basis and relationship state."""
import argparse,hashlib,json,resource,time
from pathlib import Path
import numpy as np
import torch
from structural_sensor import StructuralSensor
from native_sensor import ROOT
from native_pipeline import run
from public_stream import persistence

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--tag',required=True);ap.add_argument('--mode',choices=['smoke','fit'],required=True);a=ap.parse_args();torch.set_num_threads(1);out=ROOT/f'experiments/results/causal_induction/{a.tag}.json';assert not out.exists();parent=ROOT/'experiments/results/causal_induction/curie_causal_induction_20261012_v1_prepare.json';data=json.loads(parent.read_text());begin=time.monotonic();rows=[]
 def path(x):
  p=ROOT/x['path'];assert hashlib.sha256(p.read_bytes()).hexdigest()==x['sha256'];return p
 train=[x for x in data['records'] if x['split']=='train'][:8];dev=[x for x in data['records'] if x['split']=='heldout_graph' and x['variant']=='base'][:4];prior=(1.,2.);priorbegin=time.monotonic();priorrows=[]
 for x in train:
  metric,prior=persistence(path(x),prior,True);metric.pop('issued_predictions');priorrows.append(metric)
 confidence=prior[0]/prior[1];priorwall=time.monotonic()-priorbegin;controls=[]
 for x in dev:
  metric,_=persistence(path(x),prior,False);metric.pop('issued_predictions');controls.append(metric)
 if a.mode=='smoke':
  torch.manual_seed(192);m=StructuralSensor(confidence,True);before=m.sensor.base.detach().clone();result=run(m,path(train[0]),'routed',torch.optim.Adam(m.parameters(),lr=.001));assert result['relationship_candidate_updates']>0 and float((m.sensor.base-before).abs())>0;rows=[dict(arm='structural_smoke',train=[result])]
 else:
  for arm,structural,route in [('residual_self',False,'self'),('residual_routed',False,'routed'),('structural_routed',True,'routed')]:
   torch.manual_seed(192);m=StructuralSensor(confidence,structural);opt=torch.optim.Adam(m.parameters(),lr=.001);acq=[]
   for epoch in range(2):
    for i,x in enumerate(train):acq.append(dict(epoch=epoch,**run(m,path(x),route,opt,seed=192+epoch*100+i)))
   final=[run(m,path(x),route,seed=900+i) for i,x in enumerate(dev)];ckpt=out.with_name(out.stem+'_'+arm+'.pt');torch.save(dict(model=m.state_dict(),optimizer=opt.state_dict(),confidence=confidence),ckpt);rows.append(dict(arm=arm,train=acq,dev=final,mean_dev_nll=float(np.mean([x['nll'] for x in final])),whole_fit_wall_s=priorwall+sum(x['wall_s'] for x in acq),checkpoint=str(ckpt.relative_to(ROOT)),checkpoint_sha256=hashlib.sha256(ckpt.read_bytes()).hexdigest()))
 sources=[Path(__file__),ROOT/'experiments/causal_induction/structural_sensor.py',ROOT/'experiments/causal_induction/native_sensor.py',ROOT/'experiments/causal_induction/native_pipeline.py',ROOT/'experiments/causal_induction/public_stream.py',ROOT/'sleeping_machines/addressed_event_heads.py',ROOT/'sleeping_machines/paired_route_credit.py',ROOT/'sleeping_machines/parallel_head_race_language.py',ROOT/'sleeping_machines/sparse_race_language.py'];out.write_text(json.dumps(dict(status='completed',battle='B5/R1',tag=a.tag,mode=a.mode,rows=rows,prior_acquisition=dict(confidence=confidence,targets=sum(x['targets'] for x in priorrows),wall_s=priorwall,rows=priorrows),persistence_dev=controls,mean_persistence_dev_nll=float(np.mean([x['nll'] for x in controls])),whole_pipeline_wall_s=time.monotonic()-begin,peak_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,parent_sha256=hashlib.sha256(parent.read_bytes()).hexdigest(),source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},scope='Three native arms diagnose persistence-based sensor output versus additional publicly acquired sparse temporal co-transition evidence. Same8TRAIN graphs/two passes plus charged512-target calibration acquisition; same4DEV graphs. Shared core/actual hard source routes and exact paid nonwinner credit retained. Edge statistics are generic causal evidence, not proven parent edges; their scoring/utility learned, statistic update kernel prescribed. No learned connected credit claim, depth/Transformer/public TEST/efficiency claim.'),indent=2)+'\n');print(json.dumps(dict(tag=a.tag,rows=[{k:x.get(k) for k in ['arm','mean_dev_nll','whole_fit_wall_s']} for x in rows])),flush=True)
if __name__=='__main__':main()
