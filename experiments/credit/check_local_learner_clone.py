"""R1 local-law continuation contract, for an integrated repair pipeline only."""
import argparse,json,sys
from pathlib import Path
from types import SimpleNamespace
import torch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'experiments'))
import local_clip_receiver_token_language as driver
from sleeping_machines.learning.receiver_bank_adamw import ReceiverBankAdamW
from sleeping_machines.learning.clone_local_receiver_learner import clone_local_receiver_learner

def update(model,opt,a):
 opt.zero_grad();ids=torch.tensor([[1,2,3,4]]);x,_,_=model(ids,None,torch.Generator().manual_seed(9),a,deterministic=True)
 model.readout.nll(x,torch.tensor([[2,3,4,5]])).mean().backward();opt.step()

def main():
 p=argparse.ArgumentParser();p.add_argument('--tag',required=True);tag=p.parse_args().tag;torch.set_num_threads(1)
 a=SimpleNamespace(**json.loads((ROOT/'experiments/results/token_language/aws_native_deep_d8_8m_20261013_s6_v1.json').read_text())['args']);a.payload=8;a.credit_window=4
 counts=torch.ones(50257,dtype=torch.long);torch.manual_seed(6);model=driver.Model(a,torch.arange(50257),counts);opt=ReceiverBankAdamW(model,local_clip=.25,weight_decay=.01)
 update(model,opt,a);clone,other=clone_local_receiver_learner(model,opt);assert other.receiver.local.local_clip==.25
 update(model,opt,a);update(clone,other,a)
 error=max(float((p-clone.state_dict()[n]).abs().max()) for n,p in model.named_parameters());assert error==0
 assert opt.state_dict()['versions']==other.state_dict()['versions']
 for bank,original in opt.receiver.local.optimizers.items():
  candidate=other.receiver.local.optimizers[bank]
  assert len(original.state)==len(candidate.state)
  for lhs,rhs in zip(original.state.values(),candidate.state.values()):
   for k,v in lhs.items():
    if isinstance(v,torch.Tensor):assert torch.equal(v,rhs[k])
 from sleeping_machines.route_credit_horizon import credit_endpoint
 assert credit_endpoint(3,64,16,1)==16
 assert credit_endpoint(3,64,16,4)==64
 assert credit_endpoint(63,64,16,4)==64
 names=['experiments/credit/check_local_learner_clone.py','sleeping_machines/learning/clone_local_receiver_learner.py','experiments/local_clip_receiver_token_language.py','sleeping_machines/route_credit_horizon.py']
 with (ROOT/f'experiments/results/credit/{tag}.json').open('x') as f:json.dump(dict(status='completed',battle='R1',gate='PASS',parameter_error=error,local_clip=.25,source_sha256={n:driver.sha(ROOT/n) for n in names},scope='Native D8 factual local-clipped AdamW continuation; no horizoncritic fit or speed claim.'),f,indent=2)
if __name__=='__main__':main()
