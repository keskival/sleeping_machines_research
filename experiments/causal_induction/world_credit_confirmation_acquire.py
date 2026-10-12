"""Unchanged source acquisition, admitted by prior confirmation-stage pass."""
import argparse
import json
import math
import resource
import time
import numpy as np
import torch
from public_stream import persistence
from world_credit_sensor import WorldCreditSensor
from world_credit_run import run
from world_credit_confirmation_verify import verify_result
from world_credit_pipeline import ROOT,RESULTS,PARENT,sha,fingerprint

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--tag',required=True);ap.add_argument('--seed',type=int,choices=[195,196],required=True);a=ap.parse_args()
 torch.set_num_threads(1);begin=time.monotonic();out=RESULTS/(a.tag+'.json');assert not out.exists()
 previous=f'curie_world_credit_confirm_20261012_v1_s{a.seed-1}_decision';decision=verify_result(previous);assert decision['confirmation_gate']
 contract=verify_result('curie_world_credit_20261012_v1_contract');assert contract['peak_rss_kb']<800000
 source_data=json.loads(PARENT.read_text());train=[r for r in source_data['records'] if r['split']=='train' and r['variant']=='base'];assert len(train)==16
 def path(row):
  p=ROOT/row['path'];assert sha(p)==row['sha256'];return p
 start=time.monotonic();prior=(1.,2.)
 for row in train:_,prior=persistence(path(row),prior,True)
 confidence_wall=time.monotonic()-start;confidence=prior[0]/prior[1]
 torch.manual_seed(a.seed);model=WorldCreditSensor(confidence);optimizer=torch.optim.Adam(model.parameters(),lr=.001);rows=[]
 for i,row in enumerate(train):
  r=run(model,path(row),'exact',optimizer,seed=a.seed+i);rows.append(dict(path=row['path'],**r))
  print(json.dumps(dict(seed=a.seed,episode=i,nll=r['nll'],world_nll=r['world_nll'],wall_s=r['wall_s'])),flush=True)
 checkpoint=out.with_suffix('.pt');saved=dict(model=model.state_dict(),optimizer=optimizer.state_dict(),confidence=confidence)
 incoming=fingerprint(dict(model=saved['model'],optimizer=saved['optimizer']));torch.save(saved,checkpoint)
 world=float(np.mean([r['world_nll'] for r in rows[8:]]))
 sources=dict(contract['source_sha256'])
 for n in ['experiments/causal_induction/world_credit_confirmation_acquire.py','experiments/causal_induction/world_credit_confirmation_verify.py','experiments/causal_induction/WORLD_CREDIT_CONFIRMATION_PROTOCOL.md']:sources[n]=sha(ROOT/n)
 result=dict(status='completed',battle='B5/R1',tag=a.tag,seed=a.seed,rows=rows,targets=2048,confidence_acquisition_targets=2048,
  confidence_acquisition_wall_s=confidence_wall,whole_learning_wall_s=confidence_wall+sum(r['wall_s'] for r in rows),
  checks=dict(acquisition_gate=world<math.log(2),last_eight_world_nll=world,learner_fingerprint=incoming,checkpoint=str(checkpoint.relative_to(ROOT)),checkpoint_sha256=sha(checkpoint)),
  source_sha256=sources,parent_sha256={str(PARENT.relative_to(ROOT)):sha(PARENT),str((RESULTS/(previous+'.json')).relative_to(ROOT)):sha(RESULTS/(previous+'.json'))},
  data_records=train,whole_pipeline_wall_s=time.monotonic()-begin,peak_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
  scope='Registered unchanged actor/world/Adam source acquisition with a fresh initialization seed.16original TRAIN systems,2048unique targets/4096charged confidence-plus-neural presentations. Original source gate retained; no acquired-head/moment substitution, target tuning or public scoring.')
 out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(tag=a.tag,checks=result['checks'])),flush=True)
if __name__=='__main__':main()
