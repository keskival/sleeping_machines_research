"""Bounded native32-stage FIT learning gate; indexed pooling and generated readout."""
import argparse,json,os,resource,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from torch.nn import functional as F
from sleeping_machines.native_memory_composition import NativeMemoryComposition
from experiments.hierarchy.native_composition_data import execute,source_hashes,sha

def score(model,events,labels):
 saved=torch.get_rng_state();mode=model.training;model.eval();rows=[];work=[]
 prior=torch.bincount(labels,minlength=11).float()/len(labels)
 try:
  with torch.no_grad():
   for seed in [113,127]:
    predictions=[]
    for event in events:
     # Same exogenous tape across episodes; no per-item noise identity.
     torch.manual_seed(seed);logits,_,counts=execute(model,event);predictions.append(logits);work.append(counts)
    logp=torch.stack(predictions).log_softmax(-1)
    nll=float(F.nll_loss(logp,labels));independent=float(-(logp*prior[None]).sum(-1).mean())
    rows.append(dict(seed=seed,nll=nll,accuracy=float((logp.argmax(-1)==labels).float().mean()),
         label_association_gain=independent-nll,logit_std=float(logp.std(0).mean())))
 finally:torch.set_rng_state(saved);model.train(mode)
 return dict(rows=rows,nll=sum(r['nll'] for r in rows)/len(rows),accuracy=sum(r['accuracy'] for r in rows)/len(rows),label_association_gain=sum(r['label_association_gain'] for r in rows)/len(rows)),work

def main():
 p=argparse.ArgumentParser();p.add_argument('--tag',required=True);p.add_argument('--preflight',required=True);p.add_argument('--steps',type=int,default=64);a=p.parse_args()
 torch.set_num_threads(1);torch.manual_seed(101)
 pre_path=ROOT/a.preflight;pre=json.loads(pre_path.read_text())
 if not pre.get('eligible_for_learning'):raise ValueError('native preflight did not admit fit')
 for n,digest in pre['source_sha256'].items():
  if sha(ROOT/n)!=digest:raise ValueError('changed native preflight source: '+n)
 data=ROOT/pre['data']['path']
 if sha(data)!=pre['data']['sha256']:raise ValueError('changed raw prefixes')
 with np.load(data) as packed:events=packed['events'].copy();labels=torch.tensor(packed['labels'],dtype=torch.long)
 model=NativeMemoryComposition();opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01)
 order=torch.Generator().manual_seed(131);outdir=ROOT/'experiments/results/hierarchy';checkpoint=outdir/(a.tag+'.progress.pt')
 sources=source_hashes();sources['experiments/hierarchy/native_composition_learning.py']=sha(ROOT/'experiments/hierarchy/native_composition_learning.py')
 rows=[];first=0;paid=0.;initial_work=[]
 if checkpoint.exists():
  saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
  if saved['source_sha256']!=sources:raise ValueError('changed checkpoint source')
  model.load_state_dict(saved['model']);opt.load_state_dict(saved['optimizer']);torch.set_rng_state(saved['torch_rng']);order.set_state(saved['order_rng'])
  rows=saved['rows'];first=saved['step'];paid=saved['paid_wall_s'];initial=saved['initial'];initial_work=saved['initial_work']
 else:
  t=time.monotonic();initial,initial_work=score(model,events,labels);paid=time.monotonic()-t
 start=time.monotonic()
 for step in range(first,a.steps):
  index=int(torch.randint(len(labels),(1,),generator=order));model.train();opt.zero_grad(set_to_none=True)
  logits,_,work=execute(model,events[index]);loss=F.cross_entropy(logits[None],labels[index:index+1]);loss.backward()
  if not torch.isfinite(loss) or any(not torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None):raise ValueError('nonfinite native learning; exact last checkpoint retained')
  opt.step();rows.append(dict(step=step+1,episode=index,nll=float(loss.detach()),work=work))
  if (step+1)%4==0 or step+1==a.steps:
   saved=dict(model=model.state_dict(),optimizer=opt.state_dict(),torch_rng=torch.get_rng_state(),order_rng=order.get_state(),step=step+1,rows=rows,initial=initial,initial_work=initial_work,paid_wall_s=paid+time.monotonic()-start,source_sha256=sources)
   tmp=checkpoint.with_suffix('.tmp');torch.save(saved,tmp);os.replace(tmp,checkpoint)
   print(json.dumps(dict(step=step+1,nll=rows[-1]['nll'],elapsed_s=paid+time.monotonic()-start)),flush=True)
 final,final_work=score(model,events,labels);weights=outdir/(a.tag+'.final.pt');torch.save(model.state_dict(),weights)
 result=dict(status='completed',battle='G-HIER',stage='native_composition_learning',initial=initial,final=final,
  source_sha256=sources,input_sha256={str(pre_path.relative_to(ROOT)):sha(pre_path),str(data.relative_to(ROOT)):sha(data)},
  steps=a.steps,training_presentations=len(rows),observed_events_per_presentation=events.shape[1],unique_episodes=len(labels),parameters=sum(p.numel() for p in model.parameters()),history=rows,
  initial_scoring_work=initial_work,final_scoring_work=final_work,constant_predictor_nll=pre['constant_predictor_nll'],majority_accuracy=pre['majority_accuracy'],
  final_weights=str(weights.relative_to(ROOT)),fitting_and_scoring_wall_s=paid+time.monotonic()-start,max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
  scope='One FIT recording,first32 raw events/episode,64 batch1 updates. Two fixed shared noise tapes, same stochastic train/eval semantics. Memorization/feature-use diagnostic, not held-out gesture quality or semantic hierarchy.',
  state_policy='Independent episodes reset; all native addressed memories persist through event processing, lower-memory queries and generated readout. No online-stream verdict.',
  accounting='Operation counters charge native key/value work and queries separately; input projection, normalization, readout, optimizer, queue and graph work still require full arithmetic/traffic audit.',
  next_decision='Judge input use by association gain and constant-label floor, not loss alone. If native learning succeeds, test source/temporal contributions and held-out composition; otherwise diagnose clock/content credit before scale.')
 with (outdir/(a.tag+'.json')).open('x') as f:json.dump(result,f,indent=2);f.write('\n')
 print(json.dumps(dict(initial=initial,final=final,constant_predictor_nll=result['constant_predictor_nll'])))
if __name__=='__main__':main()
