"""Attribute native credit quality to learning/memory, not changed audit noise."""
import argparse,hashlib,json,resource,time
from pathlib import Path
import numpy as np
import torch
from native_sensor import ROOT
from credit_run import run
from race_credit_sensor import RaceCreditSensor
from public_stream import persistence

class ZeroAudit(RaceCreditSensor):
 def credit_event(self,pair,t,features,kind,loss,audited,st):return features.new_zeros(())
class ResetMemory(RaceCreditSensor):
 def issue(self,r,st,generator,arm='credit',learning=False):
  st['credit_nodes']={}
  return super().issue(r,st,generator,arm,learning)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--tag',required=True);a=ap.parse_args();torch.set_num_threads(1);out=ROOT/f'experiments/results/causal_induction/{a.tag}.json';assert not out.exists();parent=ROOT/'experiments/results/causal_induction/curie_causal_induction_20261012_v1_prepare.json';reference=ROOT/'experiments/results/causal_induction/curie_causal_race_credit_20261012_v1_fit.json';data=json.loads(parent.read_text());ref=json.loads(reference.read_text());assert ref['status']=='completed';begin=time.monotonic();rows=[]
 for name,digest in ref['source_sha256'].items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest
 def path(x):
  p=ROOT/x['path'];assert hashlib.sha256(p.read_bytes()).hexdigest()==x['sha256'];return p
 train=[x for x in data['records'] if x['split']=='train'][:8];dev=[x for x in data['records'] if x['split']=='heldout_graph' and x['variant']=='base'][:4];prior=(1.,2.);start=time.monotonic()
 for x in train:_,prior=persistence(path(x),prior,True)
 priorwall=time.monotonic()-start;confidence=prior[0]/prior[1];torch.manual_seed(194);initial=RaceCreditSensor(confidence).state_dict()
 for label,cls in [('zero25',ZeroAudit),('reset_memory',ResetMemory)]:
  m=cls(confidence);m.load_state_dict(initial);opt=torch.optim.Adam(m.parameters(),lr=.001);acq=[]
  for epoch in range(2):
   for i,x in enumerate(train):acq.append(dict(epoch=epoch,**run(m,path(x),'credit',opt,seed=194+epoch*100+i)))
  final=[run(m,path(x),'credit',seed=900+i) for i,x in enumerate(dev)];ckpt=out.with_name(out.stem+'_'+label+'.pt');torch.save(dict(model=m.state_dict(),optimizer=opt.state_dict(),confidence=confidence),ckpt);rows.append(dict(arm=label,train=acq,dev=final,mean_dev_nll=float(np.mean([x['nll'] for x in final])),whole_fit_wall_s=priorwall+sum(x['wall_s'] for x in acq),checkpoint=str(ckpt.relative_to(ROOT)),checkpoint_sha256=hashlib.sha256(ckpt.read_bytes()).hexdigest()))
 learned=next(x for x in ref['rows'] if x['arm']=='credit');comparison={x['arm']:dict(learned_quality_gain=x['mean_dev_nll']-learned['mean_dev_nll'],learned_time_ratio=learned['whole_fit_wall_s']/x['whole_fit_wall_s']) for x in rows};sources=[Path(__file__),ROOT/'experiments/causal_induction/race_credit_sensor.py',ROOT/'experiments/causal_induction/structural_sensor.py',ROOT/'experiments/causal_induction/native_sensor.py',ROOT/'experiments/causal_induction/credit_run.py',ROOT/'experiments/causal_induction/public_stream.py',ROOT/'sleeping_machines/addressed_event_heads.py',ROOT/'sleeping_machines/paired_route_credit.py',ROOT/'sleeping_machines/parallel_head_race_language.py',ROOT/'sleeping_machines/sparse_race_language.py'];out.write_text(json.dumps(dict(status='completed',battle='B5/R1',tag=a.tag,rows=rows,comparisons=comparison,prior_acquisition_targets=512,prior_acquisition_wall_s=priorwall,whole_pipeline_wall_s=time.monotonic()-begin,peak_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,reference_sha256=hashlib.sha256(reference.read_bytes()).hexdigest(),parent_sha256=hashlib.sha256(parent.read_bytes()).hexdigest(),source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},scope='Exact same seed194 actor/credit initialization, TRAIN exposure, native learning/clipping, factual race and teaching RNG,25% audit schedule. zero25 uses zero contrast prediction plus unbiased exact residual without native credit event computation; reset_memory still learns native credit parameters but erases numerical credit history before each request. These distinguish audit-rate/noise and persistent-memory effects from the saved credit arm. Four private DEV graphs, one actor seed; no independent confirmation or efficiency claim.'),indent=2)+'\n');print(json.dumps(dict(rows=[{k:x[k] for k in ['arm','mean_dev_nll','whole_fit_wall_s']} for x in rows],comparisons=comparison)),flush=True)
if __name__=='__main__':main()
