"""B10 information diagnosis: head updates invisible to state affect predictive clocks."""
import argparse,hashlib,json,resource,sys,time
from pathlib import Path
import torch
from torch.func import functional_call
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'experiments/credit'))
from market_native_test_time import restored,load,warm
from real_boundary_stable import control
class Responses(torch.nn.Module):
 def __init__(self,model):super().__init__();self.model=model
 def forward(self,t,m,state):
  forecast,end=control.market_consume(self.model,t,m,state,parameter_version=0)
  return torch.cat([x.flatten() for x in control.tensors(end)]),torch.cat([x.flatten() for x in forecast['clocks']])
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--tag',required=True);a=ap.parse_args();torch.set_num_threads(1);torch.set_default_dtype(torch.float64);begin=time.monotonic()
 model,inputs=restored();t,m,path=load('2026-08-24',129);inputs[str(path.relative_to(ROOT))]=sha(path);state=warm(model,t,m,128)
 module=Responses(model);mapping={'model.'+n:p.detach().clone() for n,p in model.named_parameters()}
 with torch.no_grad():
  factual=module(t[128:129],m[128:129],control.detach_state(state));perturbed=dict(mapping)
  name='model.clock.bias';assert name in perturbed;perturbed[name]=perturbed[name]+.01
  changed=functional_call(module,perturbed,(t[128:129],m[128:129],control.detach_state(state)))
 state_error=float((factual[0]-changed[0]).abs().max());forecast_change=float((factual[1]-changed[1]).abs().max())
 assert state_error==0 and forecast_change>1e-8
 sources=['experiments/credit/market_forecast_response_contract.py','experiments/credit/market_native_test_time.py','experiments/credit/real_boundary_stable.py','experiments/market/episode_state.py','experiments/tpp/race_tpp_v5.py']
 r=dict(status='completed',battle='B10',checks=dict(state_response_max=state_error,forecast_response_max=forecast_change,state_width=factual[0].numel(),forecast_width=factual[1].numel(),perturbed_parameter=name,perturbation=.01,observed_events=129),input_sha256=inputs,source_sha256={p:sha(ROOT/p) for p in sources},whole_wall_s=time.monotonic()-begin,peak_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,scope='Causal TRAIN-only information diagnosis on restored full actor. Clock-head perturbation changes predictive law while persistent-state response is zero. No critic training, quality improvement or proof that all existing critic inputs are insufficient.')
 with (ROOT/f'experiments/results/market/{a.tag}.json').open('x') as f:json.dump(r,f,indent=2);f.write('\n')
 print(json.dumps(r['checks']),flush=True)
if __name__=='__main__':main()
