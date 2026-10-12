"""B5/R1 diagnosed teacher separation, own native shared-bank model only."""
import argparse
import hashlib
import json
import resource
import time
from pathlib import Path
import numpy as np
import torch
from native_sensor import ROOT
from task_taught_shared_bank import TaskTaughtSharedBank
from task_taught_run import run
from public_stream import persistence
import shared_bank_pipeline_v2 as witness_module


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
 ap=argparse.ArgumentParser();ap.add_argument('--tag',required=True);ap.add_argument('--mode',choices=['contract','fit'],required=True);a=ap.parse_args()
 torch.set_num_threads(1);begin=time.monotonic();out=ROOT/f'experiments/results/causal_induction/{a.tag}.json';assert not out.exists()
 diagnosis=ROOT/'experiments/results/causal_induction/curie_shared_bank_20261012_v1_future_diagnosis.json'
 d=json.loads(diagnosis.read_text());assert d['status']=='completed'
 for name,digest in d['source_sha256'].items():assert sha(ROOT/name)==digest
 parent=ROOT/'experiments/results/causal_induction/curie_causal_induction_20261012_v1_prepare.json';data=json.loads(parent.read_text())
 def path(r):
  p=ROOT/r['path'];assert sha(p)==r['sha256'];return p
 train=[r for r in data['records'] if r['split']=='train'][:8]
 dev=[r for r in data['records'] if r['split']=='heldout_graph' and r['variant']=='base'][:4]
 prior=(1.,2.);start=time.monotonic()
 for r in train:_,prior=persistence(path(r),prior,True)
 priorwall=time.monotonic()-start;confidence=prior[0]/prior[1]
 torch.manual_seed(194);initial=TaskTaughtSharedBank(confidence).state_dict();checks={};rows=[]
 if a.mode=='contract':
  m=TaskTaughtSharedBank(confidence);m.load_state_dict(initial);m.train();st=m.fresh()
  token=torch.linspace(-1,1,m.critic.content_dim-3)
  prediction=m.credit_event((0,1),0.,token,0.,0.,0.,st)
  (prediction-.4).square().backward()
  assert float(m.critic.head.weight.grad.abs().max())>0
  assert all(p.grad is None or torch.count_nonzero(p.grad)==0 for n,p in m.critic.named_parameters() if not n.startswith('head.'))
  torch.manual_seed(801);active={k:v.clone() for k,v in initial.items()}
  active['sensor.net.2.weight']=torch.randn_like(active['sensor.net.2.weight'])*.1
  original=witness_module.SharedBankSensor;witness_module.SharedBankSensor=TaskTaughtSharedBank
  try:
   taught=witness_module.witness('reciprocal',active)
   untaught=witness_module.witness('untaught',active)
  finally:witness_module.SharedBankSensor=original
  assert taught['logit']==untaught['logit']
  assert taught['critic_write_gradient']>0 and taught['critic_read_gradient']>0
  assert untaught['critic_write_gradient']==untaught['critic_read_gradient']==0
  assert untaught['actor_read_gradient']>0
  # Verify teaching-path change preserves initial executable forward function.
  old=original(confidence,'reciprocal');old.load_state_dict(initial)
  new=TaskTaughtSharedBank(confidence,'reciprocal');new.load_state_dict(initial)
  old_result=run(old,path(train[0]),'credit',seed=194)
  new_result=run(new,path(train[0]),'credit',seed=194)
  assert old_result['nll']==new_result['nll']
  smoke=run(new,path(train[0]),'credit',torch.optim.Adam(new.parameters(),lr=.001),seed=194)
  checks=dict(auxiliary_teacher_updates_only_loss_readout=True,
              task_teacher_retains_native_memory_access_gradients=taught,
              matched_untaught=untaught,initial_forward_parity=True,smoke=smoke)
 else:
  contract=ROOT/'experiments/results/causal_induction/curie_task_taught_bank_20261012_v1_contract.json'
  c=json.loads(contract.read_text());assert c['status']=='completed'
  for name,digest in c['source_sha256'].items():assert sha(ROOT/name)==digest
  for mode in ['untaught','reciprocal']:
   m=TaskTaughtSharedBank(confidence,mode);m.load_state_dict(initial)
   opt=torch.optim.Adam(m.parameters(),lr=.001);acq=[]
   for epoch in range(2):
    for i,r in enumerate(train):acq.append(dict(epoch=epoch,**run(m,path(r),'credit',opt,seed=194+epoch*100+i)))
   final=[run(m,path(r),'credit',seed=900+i) for i,r in enumerate(dev)]
   ckpt=out.with_name(out.stem+'_'+mode+'.pt');torch.save(dict(model=m.state_dict(),optimizer=opt.state_dict(),confidence=confidence),ckpt)
   row=dict(arm=mode,train=acq,dev=final,mean_dev_nll=float(np.mean([r['nll'] for r in final])),whole_fit_wall_s=priorwall+sum(r['wall_s'] for r in acq),checkpoint=str(ckpt.relative_to(ROOT)),checkpoint_sha256=sha(ckpt));rows.append(row)
   print(json.dumps({k:row[k] for k in ['arm','mean_dev_nll','whole_fit_wall_s']}),flush=True)
 sources=[Path(__file__),ROOT/'experiments/causal_induction/task_taught_shared_bank.py',ROOT/'experiments/causal_induction/task_taught_run.py']
 sources+=[ROOT/name for name in d['source_sha256'] if ROOT/name not in sources]
 result=dict(status='completed',battle='B5/R1',tag=a.tag,mode=a.mode,checks=checks,rows=rows,
             parent_sha256=sha(parent),diagnosis_sha256=sha(diagnosis),prior_acquisition_targets=512,
             whole_pipeline_wall_s=time.monotonic()-begin,peak_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
             source_sha256={str(p.relative_to(ROOT)):sha(p) for p in sources},
             scope='Diagnosed native shared-bank teacher separation. Forward task loss trains critic memory/read policy; auxiliary revealed-loss regression trains the identical linear readout on detached native features. Separate actor/native-memory/readout clipping, identical forward construction/parameters/initialization/audit schedule and shared-bank bounds. Two matched untaught/reciprocal arms, same8TRAINgraphs/twopasses/512prior plus1024neural presentations and originalfour selectionDEV. Numeric memory persists, task gradients use eight-feedback windows. No external dense controller, global destination learning, full optimizer meta-gradient, publicTEST, confirmation or efficiency claim.')
 out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(tag=a.tag,checks=checks)),flush=True)


if __name__=='__main__':main()
