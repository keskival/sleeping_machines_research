"""R1 acquired bank moment/ownership migration gate; no fitting or updates."""
import argparse,json,sys,time,resource
from pathlib import Path
from types import SimpleNamespace
import torch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'experiments'))
import integrated_receiver_token_language as driver
from sleeping_machines.learning.receiver_bank_adamw import ReceiverBankAdamW
from sleeping_machines.learning.optional_depth_learner import migrate_optional_depth_learner

def main():
 p=argparse.ArgumentParser();p.add_argument('--tag',required=True);a=p.parse_args();torch.set_num_threads(1);start=time.monotonic();base=ROOT/'experiments/results/token_language/aws_integrated_receiver_d8_8m_20261014_s6_v1';r=json.loads(base.with_suffix('.json').read_text());args=SimpleNamespace(**r['args']);tr=driver.load_tokens(args.train_file);counts=driver.LaneTokens(tr,0,args.train_tokens,args.lanes).counts();order=torch.argsort(counts,descending=True,stable=True);torch.manual_seed(args.seed);parent=driver.Model(args,order,counts);ck=torch.load(base.with_suffix('.pt'),weights_only=False);parent.load_state_dict(ck['model']);opt=ReceiverBankAdamW(parent,lr=args.lr,weight_decay=args.weight_decay);opt.load_state_dict(ck['optimizer']);opt.zero_grad()
 grown_args=SimpleNamespace(**vars(args));grown_args.depth=args.depth+2
 with torch.random.fork_rng():torch.manual_seed(501);fresh=driver.Model(grown_args,order,counts)
 rng=torch.get_rng_state();grown,rebuilt,state=migrate_optional_depth_learner(parent,opt,fresh.core,ck['state']);assert torch.equal(rng,torch.get_rng_state())
 assert rebuilt.update==opt.update and rebuilt.deliveries==opt.deliveries
 old_banks=set(opt.receiver.local.parameters);added=set(rebuilt.receiver.local.parameters)-old_banks
 for bank in old_banks:
  old=opt.receiver.local.parameters[bank];new=rebuilt.receiver.local.parameters[bank]
  assert old.data_ptr()!=new.data_ptr();prefix=new[:len(old)] if old.shape!=new.shape else new
  assert torch.equal(old,prefix),bank
  assert rebuilt.receiver.local.versions[bank]==opt.receiver.local.versions[bank]
  old_s=opt.receiver.local.optimizers[bank].state[old];new_s=rebuilt.receiver.local.optimizers[bank].state[new]
  for key,t in old_s.items():
   if torch.is_tensor(t):
    candidate=new_s[key];candidate=candidate[:len(t)] if candidate.shape!=t.shape else candidate
    assert torch.equal(t,candidate),(bank,key)
   else:assert t==new_s[key]
 for bank in added:
  assert rebuilt.receiver.local.versions[bank]==0 and not rebuilt.receiver.local.optimizers[bank].state
 for k in ck['state']:
  if isinstance(state[k],list):assert all(torch.equal(u,v) and u.data_ptr()!=v.data_ptr() for u,v in zip(state[k],ck['state'][k]))
  else:assert torch.equal(state[k],ck['state'][k])
 names=['experiments/credit/native_optional_depth_learner_contract.py','sleeping_machines/learning/optional_depth_learner.py','sleeping_machines/optional_temporal_depth.py','sleeping_machines/learning/receiver_bank_adamw.py','sleeping_machines/learning/calibrated_receiver_adamw.py','experiments/integrated_receiver_token_language.py']
 record=dict(status='completed',battle='R1',old_banks=len(old_banks),added_banks=len(added),old_parameters_moments_versions_ownership_preserved=True,numerical_state_rng_preserved=True,source_sha256={n:driver.sha(ROOT/n) for n in names},input_sha256={str(p.relative_to(ROOT)):driver.sha(p) for p in [base.with_suffix('.json'),base.with_suffix('.pt')]},wall_s=time.monotonic()-start,max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,scope='Actual acquired global receiverD8 learner →optionalD10; no optimizer update/fit, newbankcalibration and activationselection pending; shared expandedtransport arrays inheritoldstep withzeroaddedmoments.')
 out=ROOT/f'experiments/results/diagnostics/{a.tag}.json'
 with out.open('x') as f:json.dump(record,f,indent=2)
 print(json.dumps(record),flush=True)
if __name__=='__main__':main()
