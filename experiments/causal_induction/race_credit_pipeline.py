"""Native credit-token/race-memory integration and causal quality comparison."""
import argparse,hashlib,json,resource,time
from pathlib import Path
import numpy as np
import torch
from native_sensor import ROOT
from credit_run import run
from race_credit_sensor import RaceCreditSensor
from public_stream import persistence

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--tag',required=True);ap.add_argument('--mode',choices=['contract','fit'],required=True);a=ap.parse_args();torch.set_num_threads(1);out=ROOT/f'experiments/results/causal_induction/{a.tag}.json';assert not out.exists();parent=ROOT/'experiments/results/causal_induction/curie_causal_induction_20261012_v1_prepare.json';data=json.loads(parent.read_text());begin=time.monotonic();rows=[];checks={}
 def path(x):
  p=ROOT/x['path'];assert hashlib.sha256(p.read_bytes()).hexdigest()==x['sha256'];return p
 train=[x for x in data['records'] if x['split']=='train'][:8];dev=[x for x in data['records'] if x['split']=='heldout_graph' and x['variant']=='base'][:4];prior=(1.,2.);start=time.monotonic()
 for x in train:_,prior=persistence(path(x),prior,True)
 priorwall=time.monotonic()-start;confidence=prior[0]/prior[1];torch.manual_seed(194);initial=RaceCreditSensor(confidence).state_dict()
 if a.mode=='contract':
  m=RaceCreditSensor(confidence);m.load_state_dict(initial);exact=run(m,path(train[0]),'exact');m.load_state_dict(initial);joint=run(m,path(train[0]),'joint');assert abs(exact['nll']-joint['nll'])<1e-10;before=m.critic.units[0][0][0][0].input.weight.detach().clone();fit=run(m,path(train[0]),'joint',torch.optim.Adam(m.parameters(),lr=.001));change=float((before-m.critic.units[0][0][0][0].input.weight).abs().max());assert change>0;assert fit['credit_selected_updates']>0 and fit['credit_counterfactual_values']>0;delta=torch.tensor([.4,-.7]);estimate=torch.tensor([.2,.1]);p=.25;assert torch.allclose((1-p)*estimate+p*(estimate+(delta-estimate)/p),delta);checks=dict(initial_policy_preserved=True,native_credit_unit_parameter_change=change,audit_mean_identity=True,causal_fit=fit)
 else:
  for arm in ['exact','credit','joint']:
   m=RaceCreditSensor(confidence);m.load_state_dict(initial);opt=torch.optim.Adam(m.parameters(),lr=.001);acq=[]
   for epoch in range(2):
    for i,x in enumerate(train):acq.append(dict(epoch=epoch,**run(m,path(x),arm,opt,seed=194+epoch*100+i)))
   final=[run(m,path(x),arm,seed=900+i) for i,x in enumerate(dev)];ckpt=out.with_name(out.stem+'_'+arm+'.pt');torch.save(dict(model=m.state_dict(),optimizer=opt.state_dict(),confidence=confidence),ckpt);rows.append(dict(arm=arm,train=acq,dev=final,mean_dev_nll=float(np.mean([x['nll'] for x in final])),whole_fit_wall_s=priorwall+sum(x['wall_s'] for x in acq),checkpoint=str(ckpt.relative_to(ROOT)),checkpoint_sha256=hashlib.sha256(ckpt.read_bytes()).hexdigest()))
 sources=[Path(__file__),ROOT/'experiments/causal_induction/race_credit_sensor.py',ROOT/'experiments/causal_induction/structural_sensor.py',ROOT/'experiments/causal_induction/native_sensor.py',ROOT/'experiments/causal_induction/credit_run.py',ROOT/'experiments/causal_induction/public_stream.py',ROOT/'sleeping_machines/addressed_event_heads.py',ROOT/'sleeping_machines/paired_route_credit.py',ROOT/'sleeping_machines/parallel_head_race_language.py',ROOT/'sleeping_machines/sparse_race_language.py'];out.write_text(json.dumps(dict(status='completed',battle='B5/R1',tag=a.tag,mode=a.mode,checks=checks,rows=rows,prior_acquisition_targets=512,prior_acquisition_wall_s=priorwall,whole_pipeline_wall_s=time.monotonic()-begin,peak_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,parent_sha256=hashlib.sha256(parent.read_bytes()).hexdigest(),source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},scope='Credit predictor itself uses shared two-layer/two-head hard temporal race units with relationship-addressed persistent state; learned credit-token projection, keys/clocks/gates/decay/rotation/message maps and linear future-loss readout. Query tokens only selected/sampled relationships; feedback updates only observed/audited relationships. No dense GRU/MLP credit controller. Exact versus25%-audit native credit and joint cached-credit route prior. Own credit noise and separate actor/credit clipping; numerical memory carries through truncation. Credit predicts future proper forecast loss, not backprop gradients. Private affine diagnostic, no depth/public benchmark/efficiency claim; decoder counterfactuals cheap here, so no presumed economy.'),indent=2)+'\n');print(json.dumps(dict(tag=a.tag,checks=checks,rows=[{k:x[k] for k in ['arm','mean_dev_nll','whole_fit_wall_s']} for x in rows])),flush=True)
if __name__=='__main__':main()
