"""Replay failed B10 Transformer from exact saved state to identify nonfinite credit."""
import argparse,copy,json,resource,sys,time,traceback
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'experiments/market'))
from b10_matched_data import prepare
from b10_matched_models import MatchedRace
from b10_matched_pipeline import sha
from b10_matched_update import update

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--tag',required=True);ap.add_argument('--family',choices=['lstm','transformer'],required=True);a=ap.parse_args();torch.set_num_threads(1);torch.set_default_dtype(torch.float64)
 source=ROOT/f'experiments/results/market/aws_b10_matched_{a.family}_s0_20261014_v2.progress.pt';saved=torch.load(source,weights_only=False);data,_=prepare()
 gaps=np.concatenate([np.diff(t) for t,m in data['train']]);gaps=gaps[gaps>0];torch.manual_seed(0);model=MatchedRace(gaps,a.family);opt=torch.optim.AdamW(model.parameters(),lr=saved['lr'],weight_decay=1e-4,foreach=False);model.load_state_dict(saved['model']);opt.load_state_dict(saved['optimizer']);torch.set_rng_state(saved['rng']);rng=np.random.default_rng();rng.bit_generator.state=saved['numpy_rng'];s=saved['state'];order=s['order'];cursor=s['cursor'];rows=[];start=time.monotonic();failure=None
 for attempt in range(52):
  if cursor>=len(order):order=rng.permutation(len(data['train'])).tolist();cursor=0
  indices=order[cursor:cursor+16];batch=[data['train'][i] for i in indices];t=torch.tensor(np.stack([x[0] for x in batch]));m=torch.tensor(np.stack([x[1] for x in batch]));model.train();opt.zero_grad(set_to_none=True)
  try:
   # Parameter hooks name the first nonfinite delivered cotangent; anomaly
   # tracing identifies the actual backward operation that produced it.
   delivered=[];hooks=[]
   for name,p in model.named_parameters():
    if p.requires_grad:
     def inspect(g,name=name):
      if not torch.isfinite(g).all():delivered.append(name)
     hooks.append(p.register_hook(inspect))
   before_model=copy.deepcopy(model.state_dict());before_optimizer=copy.deepcopy(opt.state_dict());before_rng=torch.get_rng_state()
   loss=update(model,opt,t,m)
   rows.append(dict(attempt=attempt,cursor=cursor,windows=indices,loss=loss));cursor+=len(indices)
  except Exception as error:
   failure=dict(attempt=attempt,cursor=cursor,windows=indices,nonfinite_parameter_credit=delivered,error=str(error),traceback=traceback.format_exc(),gap_min=float(torch.diff(t).min()),gap_max=float(torch.diff(t).max()))
   model.load_state_dict(before_model);opt.load_state_dict(before_optimizer);torch.set_rng_state(before_rng)
   try:
    with torch.autograd.detect_anomaly(check_nan=True):update(model,opt,t,m)
   except Exception as traced:
    failure['anomaly_error']=str(traced);failure['anomaly_traceback']=traceback.format_exc()
   break
  finally:
   for h in hooks:h.remove()
 names=['experiments/market/b10_failure_audit_v2.py','experiments/market/b10_matched_models.py','experiments/market/b10_matched_data.py','experiments/market/b10_matched_update.py','experiments/tpp/race_tpp_v5.py']
 record=dict(status='completed',battle='B10',stage='numerical_failure_replay',family=a.family,reproduced=failure is not None,failure=failure,updates=rows,source_sha256={n:sha(ROOT/n) for n in names},input_sha256={str(source.relative_to(ROOT)):sha(source)},wall_s=time.monotonic()-start,max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,scope='Exact failedreference checkpoint replay; diagnostic updates charged separately, no quality result, optimizer change or numerical repair inferred without failure trace.')
 out=ROOT/f'experiments/results/market/{a.tag}.json'
 with out.open('x') as f:json.dump(record,f,indent=2)
 print(json.dumps(record),flush=True)
if __name__=='__main__':main()
