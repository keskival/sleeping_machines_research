"""Bounded own-model memorization gate, not a generalization benchmark."""
import argparse,hashlib,json,os,resource,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from torch.nn import functional as F
from sleeping_machines.cached_parent_hierarchy import EventCachedParentHierarchy
from experiments.hierarchy.cached_preflight_v2 import FILES,sha

def main():
 p=argparse.ArgumentParser();p.add_argument('--tag',required=True);p.add_argument('--preflight',required=True);p.add_argument('--steps',type=int,default=64);a=p.parse_args()
 torch.set_num_threads(1);torch.manual_seed(31)
 pre=json.loads((ROOT/a.preflight).read_text())
 if not pre.get('eligible_for_probe'):raise ValueError('preflight did not admit this fit')
 for n,digest in pre['source_sha256'].items():
  if sha(ROOT/n)!=digest:raise ValueError('changed preflight source: '+n)
 data=ROOT/pre['data']['path']
 if sha(data)!=pre['data']['sha256']:raise ValueError('changed data')
 with np.load(data) as packed:
  counts=torch.from_numpy(packed['counts'].copy());labels=torch.from_numpy(packed['labels'].copy())
 model=EventCachedParentHierarchy(width=32,levels=8,refinements=4,roles=4)
 opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=.01)
 rng=torch.Generator().manual_seed(53);outdir=ROOT/'experiments/results/hierarchy'
 checkpoint=outdir/(a.tag+'.progress.pt');rows=[];first_step=0;paid_wall=0.
 sources={n:sha(ROOT/n) for n in [*FILES,'experiments/hierarchy/cached_learning_probe_v2.py']}
 def score(blocked=()):
  # Only this FIT recording; no selection, DEV or TEST access.
  total=0.;correct=0
  with torch.no_grad():
   for i in range(len(labels)):
    output=model(counts[i:i+1].float(),seed=61,sample=False,blocked_reads=blocked)
    total+=float(F.cross_entropy(output['logits'],labels[i:i+1]))
    correct+=int(output['logits'].argmax(-1)==labels[i])
  return dict(nll=total/len(labels),accuracy=correct/len(labels),episodes=len(labels))
 if checkpoint.exists():
  saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
  if saved['source_sha256']!=sources:raise ValueError('changed checkpoint source')
  model.load_state_dict(saved['model']);opt.load_state_dict(saved['optimizer'])
  rng.set_state(saved['audit_rng']);torch.set_rng_state(saved['torch_rng'])
  rows=saved['rows'];first_step=saved['step'];initial=saved['initial'];paid_wall=saved['paid_wall_s']
 else:
  initial_start=time.monotonic();initial=score();paid_wall=time.monotonic()-initial_start
 start=time.monotonic()
 for step in range(first_step,a.steps):
  indices=torch.randint(len(labels),(2,),generator=rng)
  events=counts[indices].float();targets=labels[indices]
  site,q=model.sample_audit_site(events,rng)
  seed=int(torch.randint(2**30,(1,),generator=rng))
  opt.zero_grad(set_to_none=True)
  loss,info=model.route_objective(events,targets,site,seed=seed,site_probability=q)
  loss.backward()
  info.pop('output')
  if not torch.isfinite(loss) or any(not torch.isfinite(v.grad).all() for v in model.parameters() if v.grad is not None):
   raise ValueError('nonfinite learning; exact last checkpoint retained')
  # No importance clipping: retain the declared categorical estimator.
  opt.step()
  rows.append(dict(step=step+1,nll=float(info['factual']),cell=info['audited_cell'],replays=info['replays'],
       suffix_cells=info['suffix_cell_evaluations'],full_replay_cells=info['full_replay_cell_evaluations']))
  if (step+1)%8==0 or step+1==a.steps:
   saved=dict(model=model.state_dict(),optimizer=opt.state_dict(),step=step+1,rows=rows,initial=initial,
        audit_rng=rng.get_state(),torch_rng=torch.get_rng_state(),source_sha256=sources,
        paid_wall_s=paid_wall+time.monotonic()-start)
   temporary=checkpoint.with_suffix('.tmp');torch.save(saved,temporary);os.replace(temporary,checkpoint)
   print(json.dumps(rows[-1]),flush=True)
 final=score();no_raw_top=score(blocked=((7,0),))
 final_path=outdir/(a.tag+'.final.pt');torch.save(model.state_dict(),final_path)
 record=dict(status='completed',battle='G-HIER',stage='bounded_learning_probe',
  scope='64 updates on one official FIT recording; memorization/learnability diagnostic, not held-out quality or semantic hierarchy.',
  initial=initial,final=final,final_without_top_raw_read=no_raw_top,history=rows,source_sha256=sources,
  data_sha256=pre['data']['sha256'],preflight_sha256=sha(ROOT/a.preflight),
  fitting_wall_s=paid_wall+time.monotonic()-start,max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
  final_weights=str(final_path.relative_to(ROOT)),state_policy='Independent gesture episodes reset state; persistent all-level states within refinement. No non-stationary stream verdict.',
  next_decision='If learning succeeds, design held-out composition and same-input source-contribution tests. If it fails, diagnose update variance and content transport before a larger fit.')
 with (outdir/(a.tag+'.json')).open('x') as f:json.dump(record,f,indent=2);f.write('\n')
 print(json.dumps(dict(initial=initial,final=final,final_without_top_raw_read=no_raw_top)))
if __name__=='__main__':main()
