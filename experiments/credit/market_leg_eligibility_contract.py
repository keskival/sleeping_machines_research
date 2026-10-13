"""B10 integrated unchanged-forward and exact addressed-memory own-gradient gate."""
import argparse,hashlib,json,resource,sys,time
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'experiments/credit'))
from market_native_test_time import restored,load,warm
from real_boundary_stable import control
from experiments.market.local_eligibility_state import MarketLegEligibility
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--tag',required=True);a=ap.parse_args();torch.set_num_threads(1);torch.set_default_dtype(torch.float64);started=time.monotonic()
 model,inputs=restored();t,m,path=load('2026-08-24',256);inputs[str(path.relative_to(ROOT))]=sha(path);base=warm(model,t,m,128);params=tuple(model.parameters());names=[n for n,_ in model.named_parameters()];rows=[]
 for group in [1,8,32,128]:
  state=control.detach_state(base);losses=[]
  for k in range(127,127+group):
   forecast,state=control.market_consume(model,t[k:k+1],m[k:k+1],state)
   losses.append(-control.revealed_log_probability(model,forecast,t[k+1:k+2],m[k+1:k+2]).mean())
  reference_score=np.array([float(x.detach()) for x in losses]);reference=torch.autograd.grad(torch.stack(losses).mean(),params,allow_unused=True)
  state=control.detach_state(base);observer=MarketLegEligibility(model,state,control.market_consume);total=[torch.zeros_like(p) for p in params];scores=[]
  for k in range(127,127+group):
   forecast,end=observer.consume(t[k:k+1],m[k:k+1],state);loss=-control.revealed_log_probability(model,forecast,t[k+1:k+2],m[k+1:k+2]).mean();scores.append(float(loss.detach()))
   grads=torch.autograd.grad(loss/group,params+tuple(control.tensors(end)),allow_unused=True)
   remote=observer.historical_gradient(grads[len(params):])
   for index,(name,p,g) in enumerate(zip(names,params,grads[:len(params)])):
    if g is not None:total[index]+=g.detach()
    if name in remote:total[index]+=torch.as_tensor(remote[name],dtype=p.dtype)
   state=control.detach_state(end)
  errors={name:float((x-(torch.zeros_like(p) if y is None else y)).abs().max()) for name,p,x,y in zip(names,params,total,reference)}
  exact_names=[n for n in names if n.startswith(('marks.','clock.','mark_ctx.')) or n=='mark_slot']
  own_error=max(errors[n] for n in exact_names);score_error=float(np.max(np.abs(reference_score-np.array(scores))))
  assert own_error<1e-10 and score_error<1e-12 and observer.physical_error<1e-10
  if group==1:assert max(errors.values())<1e-10
  rows.append(dict(events=group,score_max_error=score_error,own_memory_readout_gradient_max_error=own_error,full_gradient_max_difference=max(errors.values()),gradient_error_by_parameter=errors,exact_parameter_names=exact_names,observer_physical_max_error=observer.physical_error,storage=observer.storage()))
 sources=['experiments/credit/market_leg_eligibility_contract.py','experiments/market/local_eligibility_state.py','experiments/credit/local_clock_eligibility.py','experiments/credit/market_native_test_time.py','experiments/credit/real_boundary_stable.py','experiments/market/episode_state.py','experiments/tpp/race_tpp_v5.py']
 r=dict(status='completed',battle='B10',rows=rows,source_sha256={p:sha(ROOT/p) for p in sources},input_sha256=inputs,whole_wall_s=time.monotonic()-started,peak_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,scope='Restored full18164parameter market actor on128TRAIN forecasts; unchanged physical forward and normalized recording-cell scores. Streaming one-event local backward plus own eligibility has exact addressed mark-memory/readout gradients and all gradients at1event. Cross-leg/embedding/layer remote histories deliberately omitted and measured, not whole-model exactness or learned-credit quality/efficiency. NumPy serial observer is correctness prototype; every graph is released after its observed event.')
 with (ROOT/f'experiments/results/market/{a.tag}.json').open('x') as f:json.dump(r,f,indent=2);f.write('\n')
 print(json.dumps([{k:v for k,v in x.items() if k not in ['gradient_error_by_parameter','exact_parameter_names']} for x in rows]),flush=True)
if __name__=='__main__':main()
