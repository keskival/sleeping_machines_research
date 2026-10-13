"""B10: exact own-clock gradients without retaining a temporal backward graph."""
import argparse,hashlib,json,resource,time
from pathlib import Path
import numpy as np
import torch
from local_clock_eligibility import LocalClockEligibility
ROOT=Path(__file__).resolve().parents[2]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--tag',required=True);a=ap.parse_args();torch.set_num_threads(1);torch.set_default_dtype(torch.float64);started=time.monotonic();rows=[]
 for length in [8,32,128]:
  rng=np.random.default_rng(99731);n,d=4,3
  raw=dict(wr=rng.normal(size=(n,d))*.1,wi=rng.normal(size=(n,d))*.1,gate=rng.normal(size=(n,d))*.1,log_rate=rng.normal(size=n)-1,frequency=rng.normal(size=n))
  params={k:torch.tensor(v,requires_grad=True) for k,v in raw.items()};xs=rng.normal(size=(length,d));gaps=rng.uniform(0,.5,size=length);gaps[::5]=0
  readout=torch.tensor(rng.normal(size=(2*n,3)));state=torch.zeros(n,2);losses=[]
  for x,dt in zip(xs,gaps):
   x=torch.tensor(x);decay=torch.exp(-params['log_rate'].exp()*dt);angle=params['frequency']*dt;c=decay*angle.cos();s=decay*angle.sin();r,i=state.unbind(-1);g=(params['gate']@x).sigmoid()
   state=torch.stack((c*r-s*i+g*(params['wr']@x),s*r+c*i+g*(params['wi']@x)),-1)
   losses.append((state.flatten()@readout).tanh().square().mean())
  exact=torch.autograd.grad(torch.stack(losses).mean(),tuple(params.values()))
  learner=LocalClockEligibility(**raw);total={k:np.zeros_like(v) for k,v in raw.items()};max_state_error=0.;states=[]
  for x,dt in zip(xs,gaps):
   z=learner.consume(x,float(dt));leaf=torch.tensor(z,requires_grad=True);loss=(leaf.flatten()@readout).tanh().square().mean();cot=torch.autograd.grad(loss,leaf)[0].numpy()/length
   gradient=learner.project(cot)
   for k in total:total[k]+=gradient[k]
  errors={k:float(np.max(np.abs(total[k]-v.detach().numpy()))) for k,v in zip(params,exact)}
  state_error=float(np.max(np.abs(learner.state-state.detach().numpy())));assert max(errors.values())<1e-11 and state_error<1e-11
  rows.append(dict(events=length,own_parameters=sum(v.size for v in raw.values()),trace_floats=learner.trace.size,state_floats=learner.state.size,gradient_max_error_by_parameter=errors,final_state_max_error=state_error,zero_gap_events=int(np.count_nonzero(gaps==0))))
 sources=['experiments/credit/local_clock_eligibility.py','experiments/credit/local_clock_eligibility_contract.py']
 result=dict(status='completed',battle='B10',rows=rows,source_sha256={p:sha(ROOT/p) for p in sources},whole_wall_s=time.monotonic()-started,peak_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,scope='Primitive fixed-parameter, fixed-input temporal-leg own-gradient equality against full unrolled BPTT. Trace size independent of event count; local nonlinear readout cotangents stay exact. No cross-leg/input-producer credit, changing-parameter trace transport, native hard-routing integration, market fit or performance claim.')
 with (ROOT/f'experiments/results/market/{a.tag}.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
 print(json.dumps(rows),flush=True)
if __name__=='__main__':main()
