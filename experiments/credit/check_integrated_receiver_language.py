"""R1 actual native D8 global clip/AdamW weight-decay and receiver-bank parity."""
import argparse,copy,hashlib,json,resource,sys,time
from pathlib import Path
from types import SimpleNamespace
import torch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'experiments'))
import bernoulli_teacher_token_language as native
from sleeping_machines.learning.receiver_bank_adamw import ReceiverBankAdamW

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--tag',required=True);a=ap.parse_args();start=time.monotonic();torch.set_num_threads(1)
 out=ROOT/f'experiments/results/credit/{a.tag}.json'
 if out.exists():raise FileExistsError(out)
 args=SimpleNamespace(seed=6,payload=8,depth=8,heads=2,pool=4,input_init='balanced',tie_pools=False,readout='adaptive',readout_init='frequency',minimum_tail_width=0,credit_window=4,state_mode='carry',route_credit='sampled',future_site='uniform',free_bias=0.,temperature=1.,compiled=False)
 torch.manual_seed(6);ref=native.Model(args,torch.arange(50257),torch.ones(50257,dtype=torch.long));new=copy.deepcopy(ref)
 exact=torch.optim.AdamW(ref.parameters(),lr=.003,weight_decay=.01);bank=ReceiverBankAdamW(new,lr=.003,weight_decay=.01)
 states=[None,None];gens=[torch.Generator().manual_seed(91) for _ in range(2)];rows=[]
 for step in range(4):
  tape=torch.randint(0,31,(1,9),generator=torch.Generator().manual_seed(98+step))
  for i,(model,opt) in enumerate(((ref,exact),(new,bank))):
   opt.zero_grad();x,states[i],_=model(tape[:,:-1],states[i],gens[i],args);model.readout.nll(x,tape[:,1:]).mean().backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step();states[i]=native.detach(states[i])
  error=max(float((p-q).detach().abs().max()) for p,q in zip(ref.parameters(),new.parameters()));assert error==0,error
  assert native.learner_fingerprint(states[0])==native.learner_fingerprint(states[1])
  assert torch.equal(gens[0].get_state(),gens[1].get_state())
  rows.append(dict(update=step,parameter_error=error))
  if step==1:
   saved=copy.deepcopy(bank.state_dict());replacement=ReceiverBankAdamW(new,lr=.003,weight_decay=.01);replacement.load_state_dict(saved);bank=replacement
 sources={}
 for mod in list(sys.modules.values()):
  f=getattr(mod,'__file__',None)
  if f:
   p=Path(f).resolve()
   if p.is_file() and p.is_relative_to(ROOT) and p.suffix=='.py' and p.relative_to(ROOT).parts[0] in ('experiments','sleeping_machines'):sources[str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest()
 result=dict(status='completed',battle='R1',tag=a.tag,gate='PASS',rows=rows,storage=bank.receiver.storage(),wall_s=time.monotonic()-start,max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,source_sha256=sources,scope='Native D8 four factual updates, global norm1 clip, AdamW0.01decay, exact parameters/state/RNG and receiver-bank optimizer resume. Gradient generation is still factual BP; no asynchronous hardware timing/energy claim.')
 out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
if __name__=='__main__':main()
