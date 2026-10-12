"""Unchanged native-credit confirmation with fresh actor seeds/new graphs."""
import argparse,hashlib,json,resource,time
import numpy as np
import torch
from pathlib import Path
from native_sensor import ROOT
from credit_run import run
from race_credit_sensor import RaceCreditSensor
from race_credit_controls import ZeroAudit,ResetMemory
from public_stream import persistence

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--tag',required=True);ap.add_argument('--seed',type=int,required=True,choices=[194,195,196]);a=ap.parse_args();torch.set_num_threads(1);out=ROOT/f'experiments/results/causal_induction/{a.tag}.json';assert not out.exists();parent=ROOT/'experiments/results/causal_induction/curie_causal_induction_20261012_v1_prepare.json';data=json.loads(parent.read_text());begin=time.monotonic();rows=[]
 def path(x):
  p=ROOT/x['path'];assert hashlib.sha256(p.read_bytes()).hexdigest()==x['sha256'];return p
 train=[x for x in data['records'] if x['split']=='train'][:8];dev=[x for x in data['records'] if x['split']=='heldout_graph' and x['variant']=='base'][4:8];prior=(1.,2.);start=time.monotonic()
 for x in train:_,prior=persistence(path(x),prior,True)
 priorwall=time.monotonic()-start;confidence=prior[0]/prior[1];torch.manual_seed(a.seed);initial=RaceCreditSensor(confidence).state_dict()
 for label,cls,arm in [('exact',RaceCreditSensor,'exact'),('credit',RaceCreditSensor,'credit'),('zero25',ZeroAudit,'credit'),('reset_memory',ResetMemory,'credit')]:
  m=cls(confidence);acq=[];priorfit=priorwall
  if a.seed==194:
   tag='curie_causal_race_credit_20261012_v1_fit' if label in ['exact','credit'] else 'curie_causal_race_credit_20261012_v1_controls';refpath=ROOT/f'experiments/results/causal_induction/{tag}.json';ref=json.loads(refpath.read_text());r=next(x for x in ref['rows'] if x['arm']==label);cp=ROOT/r['checkpoint'];assert hashlib.sha256(cp.read_bytes()).hexdigest()==r['checkpoint_sha256'];checkpoint=torch.load(cp,weights_only=False,map_location='cpu');m.load_state_dict(checkpoint['model']);fitwall=r['whole_fit_wall_s'];presentations=1536;source_checkpoint_sha256=r['checkpoint_sha256']
  else:
   m.load_state_dict(initial);opt=torch.optim.Adam(m.parameters(),lr=.001)
   for epoch in range(2):
    for i,x in enumerate(train):acq.append(dict(epoch=epoch,**run(m,path(x),arm,opt,seed=a.seed+epoch*100+i)))
   fitwall=priorwall+sum(x['wall_s'] for x in acq);presentations=512+sum(x['targets'] for x in acq);cp=out.with_name(out.stem+'_'+label+'.pt');torch.save(dict(model=m.state_dict(),optimizer=opt.state_dict(),confidence=confidence),cp);source_checkpoint_sha256=hashlib.sha256(cp.read_bytes()).hexdigest()
  final=[dict(path=x['path'],**run(m,path(x),arm,seed=900+i)) for i,x in enumerate(dev)];rows.append(dict(arm=label,train=acq,dev=final,mean_dev_nll=float(np.mean([x['nll'] for x in final])),whole_fit_wall_s=fitwall,target_presentations=presentations,checkpoint=str(cp.relative_to(ROOT)),checkpoint_sha256=source_checkpoint_sha256))
 learned=next(x for x in rows if x['arm']=='credit');gains={x['arm']:x['mean_dev_nll']-learned['mean_dev_nll'] for x in rows if x['arm']!='credit'};sources=[Path(__file__),ROOT/'experiments/causal_induction/race_credit_controls.py',ROOT/'experiments/causal_induction/race_credit_sensor.py',ROOT/'experiments/causal_induction/structural_sensor.py',ROOT/'experiments/causal_induction/native_sensor.py',ROOT/'experiments/causal_induction/credit_run.py',ROOT/'experiments/causal_induction/public_stream.py',ROOT/'sleeping_machines/addressed_event_heads.py',ROOT/'sleeping_machines/paired_route_credit.py',ROOT/'sleeping_machines/parallel_head_race_language.py',ROOT/'sleeping_machines/sparse_race_language.py'];protocol=ROOT/'experiments/causal_induction/CONFIRMATION.md';out.write_text(json.dumps(dict(status='completed',battle='B5/R1',tag=a.tag,seed=a.seed,rows=rows,quality_gains=gains,all_quality_positive=all(x>0 for x in gains.values()),whole_pipeline_wall_s=time.monotonic()-begin,peak_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,parent_sha256=hashlib.sha256(parent.read_bytes()).hexdigest(),protocol_sha256=hashlib.sha256(protocol.read_bytes()).hexdigest(),source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},scope='Registered unchanged native credit with newDEV graphs2004–2007 and actorseeds194/195/196; seed194 reuses completed immutable checkpoints,195/196 refit identical recipe. Each arm8TRAIN graphs/two neural passes plus512-target confidence acquisition. Four source-input/actor/noise/audit/memory controls. All fitting CPU charged; DEV fixedparameters with public causal numerical memory. Private affine diagnostic, no publicTEST/Transformer/deep-boundary/efficiency claim.'),indent=2)+'\n');print(json.dumps(dict(seed=a.seed,rows=[{k:x[k] for k in ['arm','mean_dev_nll','whole_fit_wall_s']} for x in rows],gains=gains)),flush=True)
if __name__=='__main__':main()
