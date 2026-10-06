"""Integrated numeric-state, route-RNG and factual credit witnesses."""
import argparse,hashlib,json,sys
from pathlib import Path
from types import SimpleNamespace
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'experiments'))
from split_horizon_token_language_engine import Model
from sleeping_machines.sparse_counterfactual_episodes import initial_state

def main():
 p=argparse.ArgumentParser();p.add_argument('--tag',required=True);a=p.parse_args()
 output=ROOT/f'experiments/results/diagnostics/{a.tag}.json';assert not output.exists()
 torch.set_num_threads(1);torch.manual_seed(6)
 settings=SimpleNamespace(seed=6,payload=4,depth=2,heads=2,pool=3,input_init='balanced',tie_pools=False,readout='adaptive',readout_init='frequency',minimum_tail_width=0,state_mode='carry',route_credit='sampled',future_site='uniform',free_bias=0.,temperature=1.,compiled=False,credit_window=32,future_credit_window=32)
 model=Model(settings,torch.arange(50257),torch.ones(50257,dtype=torch.long));ids=torch.arange(64).reshape(2,32)
 base=initial_state(model.core,2)
 base['mem']=[torch.randn_like(v) for v in base['mem']];base['seen']=[torch.ones_like(v) for v in base['seen']]
 rows=[];reference=None
 for factual,alternative in ((8,8),(8,32),(32,8),(32,32)):
  settings.credit_window=factual;settings.future_credit_window=alternative
  state={k:[v.clone().requires_grad_() if k=='mem' else v.clone() for v in value] if isinstance(value,list) else value.clone() for k,value in base.items()}
  gen=torch.Generator().manual_seed(7)
  x,s,_=model(ids,state,gen,settings)
  grad=torch.autograd.grad(x[:,-1].square().sum(),state['mem'][0],allow_unused=True)[0]
  size=0. if grad is None else float(grad.abs().sum())
  assert (size==0) if factual==8 else (size>0)
  numeric=(x.detach(),s,gen.get_state())
  if reference is None:reference=numeric
  else:
   assert torch.equal(reference[0],numeric[0]) and torch.equal(reference[2],numeric[2])
   for key in ('mem','arr','seen'):
    assert all(torch.equal(v,w) for v,w in zip(reference[1][key],s[key]))
   for key in ('position','ctx_vals','ctx_arr','has_ctx','last_writes','race_winners'):assert torch.equal(reference[1][key],s[key]),key
  rows.append(dict(factual=factual,alternative=alternative,initial_memory_gradient_l1=size))
 result=dict(status='completed',rows=rows,numeric_state_parity=True,feature_parity=True,route_rng_parity=True,factual_gradient_contract=True,source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),ROOT/'experiments/split_horizon_token_language_engine.py']},scope='Integrated two-depth state/gradient witness. Alternative utility learning is covered by separate actual-driver contracts; no quality claim.')
 output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
if __name__=='__main__':main()
