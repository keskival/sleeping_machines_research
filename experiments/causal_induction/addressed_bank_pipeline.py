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
from addressed_shared_bank import AddressedSharedBank
from task_taught_run import run
from public_stream import persistence
import shared_bank_pipeline_v2 as witness_module


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
 ap=argparse.ArgumentParser();ap.add_argument('--tag',required=True);ap.add_argument('--mode',choices=['contract','fit'],required=True);a=ap.parse_args()
 torch.set_num_threads(1);begin=time.monotonic();out=ROOT/f'experiments/results/causal_induction/{a.tag}.json';assert not out.exists()
 diagnosis=ROOT/'experiments/results/causal_induction/curie_task_taught_bank_20261012_v1_fit.json'
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
 torch.manual_seed(194);initial=AddressedSharedBank(confidence).state_dict();checks={};rows=[]
 if a.mode=='contract':
  m=AddressedSharedBank(confidence);m.load_state_dict(initial);m.train();st=m.fresh()
  token=torch.linspace(-1,1,m.critic.content_dim-3)
  prediction=m.credit_event((0,1),0.,token,0.,0.,0.,st)
  (prediction-.4).square().backward()
  assert float(m.critic.head.weight.grad.abs().max())>0
  assert all(p.grad is None or torch.count_nonzero(p.grad)==0 for n,p in m.critic.named_parameters() if not n.startswith('head.'))
  torch.manual_seed(801);active={k:v.clone() for k,v in initial.items()}
  active['sensor.net.2.weight']=torch.randn_like(active['sensor.net.2.weight'])*.1
  original=witness_module.SharedBankSensor;created=[]
  def factory(*args,**kwargs):
   model=AddressedSharedBank(*args,**kwargs);created.append(model);return model
  witness_module.SharedBankSensor=factory
  try:
   taught=witness_module.witness('reciprocal',active)
   relation_gradient=float(created[-1].critic.bank_relation.weight.grad.abs().max())
   assert relation_gradient>0
   untaught=witness_module.witness('untaught',active)
  finally:witness_module.SharedBankSensor=original
  assert taught['logit']==untaught['logit']
  assert taught['critic_write_gradient']>0 and taught['critic_read_gradient']>0
  assert untaught['critic_write_gradient']==untaught['critic_read_gradient']==0
  assert untaught['actor_read_gradient']>0
  # Public addressed discovery retrieves evidence outside global recency.
  indexed=AddressedSharedBank(confidence);local=indexed.fresh();indexed.bank_limit=4
  indexed.write_record(('C',(0,1)),'C',torch.ones(16),0.,local)
  for n in range(10,20):indexed.write_record(('F',n),'F',torch.zeros(16),0.,local)
  indexed.read_bank('F',torch.zeros(16),1.,local,reader=(0,))
  assert ('C',(0,1)) in local['bank_last_candidates']['F']
  # Activate output and address keys so relabeling parity is meaningful.
  model=AddressedSharedBank(confidence);model.load_state_dict(active)
  with torch.no_grad():
   model.bank_relation.weight.copy_(torch.tensor([[.2,.3,-.1,.4,.1]]))
   model.critic.bank_relation.weight.copy_(torch.tensor([[-.1,.4,.2,.1,.3]]))
  base=dev[0]; relabel=next(r for r in data['records'] if r['split']=='heldout_graph' and r['variant']=='relabel')
  first=run(model,path(base),'credit',seed=991)
  second=run(model,path(relabel),'credit',seed=991)
  assert first['nll']==second['nll'],(first['nll'],second['nll'])
  future=model.fresh();model.eval()
  model.write_record(('F',0),'F',torch.zeros(16),0.,future)
  torch.manual_seed(998);before=model.context(0,2.,future).detach()
  model.write_record(('C',(0,1)),'C',torch.ones(16)*999,99.,future)
  torch.manual_seed(998);after=model.context(0,2.,future).detach()
  assert torch.equal(before,after)
  new=AddressedSharedBank(confidence);new.load_state_dict(initial)
  smoke=run(new,path(train[0]),'credit',torch.optim.Adam(new.parameters(),lr=.001),seed=194)
  checks=dict(auxiliary_teacher_updates_only_loss_readout=True,
              task_teacher_retains_native_memory_access_gradients=taught,
              matched_untaught=untaught,relative_address_gradient=relation_gradient,
              addressed_discovery=True,relabel_forecast_parity=True,
              future_record_invisible=True,smoke=smoke)
 else:
  contract=ROOT/'experiments/results/causal_induction/curie_addressed_bank_20261012_v1_contract.json'
  c=json.loads(contract.read_text());assert c['status']=='completed'
  for name,digest in c['source_sha256'].items():assert sha(ROOT/name)==digest
  for mode in ['untaught','reciprocal']:
   m=AddressedSharedBank(confidence,mode);m.load_state_dict(initial)
   opt=torch.optim.Adam(m.parameters(),lr=.001);acq=[]
   for epoch in range(2):
    for i,r in enumerate(train):acq.append(dict(epoch=epoch,**run(m,path(r),'credit',opt,seed=194+epoch*100+i)))
   final=[run(m,path(r),'credit',seed=900+i) for i,r in enumerate(dev)]
   ckpt=out.with_name(out.stem+'_'+mode+'.pt');torch.save(dict(model=m.state_dict(),optimizer=opt.state_dict(),confidence=confidence),ckpt)
   row=dict(arm=mode,train=acq,dev=final,mean_dev_nll=float(np.mean([r['nll'] for r in final])),whole_fit_wall_s=priorwall+sum(r['wall_s'] for r in acq),checkpoint=str(ckpt.relative_to(ROOT)),checkpoint_sha256=sha(ckpt));rows.append(row)
   print(json.dumps({k:row[k] for k in ['arm','mean_dev_nll','whole_fit_wall_s']}),flush=True)
 sources=[Path(__file__),ROOT/'experiments/causal_induction/addressed_shared_bank.py',ROOT/'experiments/causal_induction/task_taught_shared_bank.py',ROOT/'experiments/causal_induction/task_taught_run.py']
 sources+=[ROOT/name for name in d['source_sha256'] if ROOT/name not in sources]
 result=dict(status='completed',battle='B5/R1',tag=a.tag,mode=a.mode,checks=checks,rows=rows,
             parent_sha256=sha(parent),diagnosis_sha256=sha(diagnosis),prior_acquisition_targets=512,
             whole_pipeline_wall_s=time.monotonic()-begin,peak_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
             source_sha256={str(p.relative_to(ROOT)):sha(p) for p in sources},
             scope='Public relative-address native sharing: per-node/relationship indexed discovery, learned equality/role keys, no identity parameter table or hidden edges. Diagnosed native shared-bank teacher separation. Forward task loss trains critic memory/read policy; auxiliary revealed-loss regression trains the identical linear readout on detached native features. Separate actor/native-memory/readout clipping, matched native architecture/initialization/audit schedule and shared-bank bounds. Two matched untaught/reciprocal arms, same8TRAINgraphs/twopasses/512prior plus1024neural presentations and originalfour selectionDEV. Numeric memory persists, task gradients use eight-feedback windows. No external dense controller, global destination learning, full optimizer meta-gradient, publicTEST, confirmation or efficiency claim.')
 out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(tag=a.tag,checks=checks)),flush=True)


if __name__=='__main__':main()
