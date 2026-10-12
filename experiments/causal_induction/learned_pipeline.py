"""B5/R1 actual learned future-loss route-credit fit; no gradient imitation."""
import argparse,copy,hashlib,json,resource,time
from pathlib import Path
import numpy as np
import torch
from native_pipeline import run
from native_sensor import ROOT
from learned_sensor import LearnedSensor

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--tag',required=True);ap.add_argument('--mode',choices=['contract','fit'],required=True);a=ap.parse_args();torch.set_num_threads(1);out=ROOT/f'experiments/results/causal_induction/{a.tag}.json';assert not out.exists();parent=ROOT/'experiments/results/causal_induction/curie_causal_induction_20261012_v1_prepare.json';data=json.loads(parent.read_text());begin=time.monotonic();rows=[];checks={}
 def path(x):
  p=ROOT/x['path'];assert hashlib.sha256(p.read_bytes()).hexdigest()==x['sha256'];return p
 torch.manual_seed(191);initial=LearnedSensor().state_dict()
 if a.mode=='contract':
  p=path(next(x for x in data['records'] if x['split']=='train'));m=LearnedSensor();m.load_state_dict(initial);exact=run(m,p,'exact');m.load_state_dict(initial);joint=run(m,p,'joint');assert abs(exact['nll']-joint['nll'])<1e-10
  delta=torch.tensor([.4,-.7]);estimate=torch.tensor([.2,.1]);prob=.25;mean=(1-prob)*estimate+prob*(estimate+(delta-estimate)/prob);assert torch.allclose(mean,delta)
  before=m.critic[-1].weight.detach().clone();result=run(m,p,'joint',torch.optim.Adam(m.parameters(),lr=.001));change=float((before-m.critic[-1].weight).abs().max());assert change>0;assert 0<result['exact_alternative_audits']<result['queries'];checks=dict(zero_critic_preserves_initial_policy=True,residual_expected_credit_identity=True,critic_parameter_change=change,causal_fit=result)
 else:
  train=[x for x in data['records'] if x['split']=='train'][:8];dev=[x for x in data['records'] if x['split']=='heldout_graph' and x['variant']=='base'][:4]
  for arm in ['exact','credit','joint']:
   m=LearnedSensor();m.load_state_dict(initial);opt=torch.optim.Adam(m.parameters(),lr=.001);acq=[]
   for epoch in range(2):
    for i,x in enumerate(train):acq.append(dict(epoch=epoch,**run(m,path(x),arm,opt,seed=191+epoch*100+i)))
   final=[run(m,path(x),arm,seed=900+i) for i,x in enumerate(dev)];ckpt=out.with_name(out.stem+'_'+arm+'.pt');torch.save(dict(model=m.state_dict(),optimizer=opt.state_dict()),ckpt);rows.append(dict(arm=arm,train=acq,dev=final,mean_dev_nll=float(np.mean([x['nll'] for x in final])),whole_fit_wall_s=sum(x['wall_s'] for x in acq),checkpoint=str(ckpt.relative_to(ROOT)),checkpoint_sha256=hashlib.sha256(ckpt.read_bytes()).hexdigest()))
 sources=[Path(__file__),ROOT/'experiments/causal_induction/learned_sensor.py',ROOT/'experiments/causal_induction/native_sensor.py',ROOT/'experiments/causal_induction/native_pipeline.py',ROOT/'experiments/causal_induction/public_stream.py',ROOT/'sleeping_machines/addressed_event_heads.py',ROOT/'sleeping_machines/paired_route_credit.py',ROOT/'sleeping_machines/parallel_head_race_language.py',ROOT/'sleeping_machines/sparse_race_language.py'];out.write_text(json.dumps(dict(status='completed',battle='B5/R1',tag=a.tag,mode=a.mode,checks=checks,rows=rows,whole_pipeline_wall_s=time.monotonic()-begin,peak_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,parent_sha256=hashlib.sha256(parent.read_bytes()).hexdigest(),source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},scope='Same integrated actor: exact paid nonwinner route credit versus connected forward-loss critic with25% alternative audits; joint additionally uses learned loss predictions to shape source-selection probabilities. Critic trained on revealed factual/audited counterfactual BCE, not backprop targets. Counterfactual route gradient anchored in expectation, local temporal receiver surrogate unchanged. Private affine4-root development; no acquired graph/depth/Transformer/efficiency conclusion. Training cost includes critic and optimizer; source-discovery context reads explicitly charged. No hidden labels. Prior dataset/actor acquisition not amortized away.'),indent=2)+'\n');print(json.dumps(dict(tag=a.tag,checks=checks,rows=[{k:x[k] for k in ['arm','mean_dev_nll','whole_fit_wall_s']} for x in rows])),flush=True)
if __name__=='__main__':main()
