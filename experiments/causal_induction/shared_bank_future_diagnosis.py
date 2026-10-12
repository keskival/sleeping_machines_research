"""Paid actual future forecasts after isolated native-critic Adam updates.
TRAIN-only diagnosis; source actor and its optimizer slots stay unchanged.
"""
import argparse
import copy
import hashlib
import json
import math
import resource
import time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from native_sensor import ROOT
from public_stream import records
from shared_bank_sensor_v2 import SharedBankSensor


def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
 ap=argparse.ArgumentParser();ap.add_argument('--tag',required=True);a=ap.parse_args()
 torch.set_num_threads(1);start=time.monotonic()
 out=ROOT/f'experiments/results/causal_induction/{a.tag}.json';assert not out.exists()
 parent=ROOT/'experiments/results/causal_induction/curie_shared_bank_20261012_v2_fit.json'
 fit=json.loads(parent.read_text());assert fit['status']=='completed'
 for name,h in fit['source_sha256'].items():assert digest(ROOT/name)==h,name
 source=next(r for r in fit['rows'] if r['arm']=='reciprocal')
 assert any(source['mean_dev_nll']>=r['mean_dev_nll'] for r in fit['rows'] if r['arm']!='reciprocal'), 'Failure diagnosis is conditional on the registered four-arm gate failing'
 checkpoint=ROOT/source['checkpoint'];assert digest(checkpoint)==source['checkpoint_sha256']
 saved=torch.load(checkpoint,weights_only=False,map_location='cpu')
 manifest=ROOT/'experiments/results/causal_induction/curie_causal_induction_20261012_v1_prepare.json'
 data=json.loads(manifest.read_text());train=[r for r in data['records'] if r['split']=='train'][:2]
 rows=[]
 for episode,row in enumerate(train):
  path=ROOT/row['path'];assert digest(path)==row['sha256'];stream=list(records(path))
  m=SharedBankSensor(saved['confidence'],'reciprocal');m.load_state_dict(saved['model']);m.train()
  opt=torch.optim.Adam(m.parameters(),lr=.001);opt.load_state_dict(copy.deepcopy(saved['optimizer']))
  st=m.fresh();g=torch.Generator().manual_seed(431+episode);torch.manual_seed(431+episode)
  task=[];aux=[];prefix_start=time.monotonic();cursor=None
  for i,r in enumerate(stream):
   if r['kind']=='event':m.observe(r,st)
   elif r['kind']=='query':m.issue(r,st,g,'credit',True)
   else:
    packet=st['issued'][r['query_id']];y=packet['logit'].new_tensor(float(r['target']))
    factual=F.binary_cross_entropy_with_logits(packet['logit'],y)
    predicted=packet['predicted'];w=packet['winner'];j=packet['alternative']
    regression=F.mse_loss(predicted[w],factual.detach())
    if j is not None and packet['audit']:
     target=F.binary_cross_entropy_with_logits(packet['alt'],y).detach()
     regression=regression+F.mse_loss(predicted[j],target)
    m.feedback(r,st,True);task.append(factual);aux.append(regression)
    if len(task)>=16 and not st['issued']:cursor=i+1;break
  assert cursor is not None
  named=[(n,p) for n,p in m.named_parameters() if n.startswith('critic.')]
  params=[p for _,p in named]
  gt=torch.autograd.grad(torch.stack(task).mean(),params,retain_graph=True,allow_unused=True)
  ga=torch.autograd.grad(torch.stack(aux).mean(),params,allow_unused=True)
  gt=[torch.zeros_like(p) if v is None else v.detach() for p,v in zip(params,gt)]
  ga=[torch.zeros_like(p) if v is None else v.detach() for p,v in zip(params,ga)]
  flat=lambda xs:torch.cat([x.reshape(-1) for x in xs])
  tvec,avec=flat(gt),flat(ga)
  cosine=float(tvec@avec/(tvec.norm()*avec.norm()).clamp_min(1e-30))
  m.detach(st)
  # No live differentiable forecast survives the boundary. Prefix evidence and
  # all RNG states are common to every intervention, not independently rerun.
  numeric=copy.deepcopy(st);weights=copy.deepcopy(m.state_dict())
  moments=copy.deepcopy(opt.state_dict());actor_rng=torch.get_rng_state();route_rng=g.get_state()
  prefix_wall=time.monotonic()-prefix_start;variants=[]
  for label,direction in [('none',None),('coast',[torch.zeros_like(x) for x in gt]),('task',gt),('regression',ga),('combined',[x+y for x,y in zip(gt,ga)])]:
   begin=time.monotonic();branch=SharedBankSensor(saved['confidence'],'reciprocal');branch.load_state_dict(weights)
   optimizer=torch.optim.Adam(branch.parameters(),lr=.001);optimizer.load_state_dict(copy.deepcopy(moments))
   optimizer.zero_grad(set_to_none=True)
   if direction is not None:
    for (_,p),v in zip([(n,p) for n,p in branch.named_parameters() if n.startswith('critic.')],direction):p.grad=v.clone()
    torch.nn.utils.clip_grad_norm_(branch.critic.parameters(),1.);optimizer.step()
   assert all(torch.equal(p,weights[n]) for n,p in branch.named_parameters() if not n.startswith('critic.'))
   branch.eval();state=copy.deepcopy(numeric);bg=torch.Generator();bg.set_state(route_rng);torch.set_rng_state(actor_rng)
   scores=[];before=dict(state['counts'])
   with torch.no_grad():
    for r in stream[cursor:]:
     if r['kind']=='event':branch.observe(r,state)
     elif r['kind']=='query':branch.issue(r,state,bg,'credit',False)
     else:
      _,p=branch.feedback(r,state,False);scores.append(-math.log(max(1e-12,p if r['target'] else 1-p)))
      if len(scores)>=16 and not state['issued']:break
   variants.append(dict(update=label,future_nll=float(np.mean(scores)),targets=len(scores),
                        wall_s=time.monotonic()-begin,
                        work={k:state['counts'][k]-before[k] for k in before if k!='bank_peak_records'}))
  base=variants[0]['future_nll']
  coast=variants[1]['future_nll']
  for v in variants:
   v['gain_vs_no_update']=base-v['future_nll']
   v['gain_vs_momentum_coast']=coast-v['future_nll']
  rows.append(dict(graph=row['graph_seed'] if 'graph_seed' in row else row['path'],
                   prefix_targets=len(task),prefix_wall_s=prefix_wall,
                   task_gradient_norm=float(tvec.norm()),regression_gradient_norm=float(avec.norm()),
                   gradient_cosine=cosine,variants=variants))
 sources=[Path(__file__),ROOT/'experiments/causal_induction/shared_bank_sensor_v2.py',ROOT/'experiments/causal_induction/public_stream.py']
 sources+=[ROOT/name for name in fit['source_sha256'] if (ROOT/name) not in sources]
 result=dict(status='completed',battle='B5/R1',tag=a.tag,rows=rows,
             parent_sha256=digest(parent),checkpoint_sha256=digest(checkpoint),
             source_acquisition_wall_s=source['whole_fit_wall_s'],
             whole_pipeline_wall_s=time.monotonic()-start,
             peak_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
             source_sha256={str(p.relative_to(ROOT)):digest(p) for p in sources},
             scope='TRAIN-only post-fit diagnosis, first two source graphs. Same16-feedback acquired state/RNG, five actual critic-only Adam interventions using copied acquired moments; actor weights/moments unchanged. Zero-gradient coast separates momentum carry from new-gradient effects. Next16feedback forecasts scored before labels, numeric shared memory continues, future weights fixed. Task-memory gradient versus loss-regression gradient and their sum, one existing0.001step, no tuning. No full optimizer meta-gradient, actor-learning continuation, heldout quality or efficiency claim. All prefix/branch work measured and source acquisition separately charged.')
 out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)


if __name__=='__main__':main()
