"""Actual strongest acquired learner: preserve old caps, moments and next update."""
import argparse,copy,json,sys,time
from pathlib import Path
from types import SimpleNamespace
import torch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'experiments'));sys.path.insert(0,str(Path(__file__).parent))
import protected_depth_local_gate as gate
from sleeping_machines.learning.protected_growth_banks import ProtectedGrowthAdamW,migrate_parent_banks

def main():
 p=argparse.ArgumentParser();p.add_argument('--tag',required=True);tag=p.parse_args().tag;torch.set_num_threads(1);began=time.monotonic();parent,_,args,train,_,_=gate.restore_arm('parent');base=ROOT/'experiments/results/token_language/aws_protected_depth_parent_2m_20261014_s6_v1';ck=torch.load(base.with_suffix('.pt'),weights_only=False);parent.load_state_dict(ck['model'])
 for name,rng in ck['owned_rng'].items():getattr(parent,name).set_state(rng)
 opt=gate.CalibratedReceiverBankAdamW(parent,ck['optimizer']['bank_caps'],lr=args.lr,weight_decay=args.weight_decay);opt.load_state_dict(ck['optimizer']);config=SimpleNamespace(**vars(args));config.depth+=2;counts=gate.pilot.driver.LaneTokens(train,0,args.train_tokens,args.lanes).counts();order=torch.argsort(counts,descending=True,stable=True)
 with torch.random.fork_rng():torch.manual_seed(args.seed+17);fresh=gate.pilot.driver.Model(config,order,counts)
 grown=gate.pilot.grow_protected_language(parent,fresh);probe=ProtectedGrowthAdamW(grown,args.depth,lr=args.lr,weight_decay=args.weight_decay);old_names={b.replace('readout.','readout.base.',1) if b.startswith('readout.') else b for b in opt.receiver.mapping};new_caps={b:1. for b in set(probe.receiver.mapping)-old_names};go,mapping=migrate_parent_banks(opt,grown,new_caps)
 for old,new in mapping.items():
  assert go.receiver.local.bank_caps[new]==opt.receiver.local.bank_caps[old];assert gate.equal_tree(opt.receiver.local.optimizers[old].state_dict(),go.receiver.local.optimizers[new].state_dict());assert go.receiver.local.versions[new]==opt.receiver.local.versions[old]
 tokens=gate.pilot.driver.BlockLaneTokens(train,args.train_tokens,args.lanes,args.exposure,args.steps+64+4096+4,args.chunk);tokens.block=ck['training_block'];cursor=ck['cursor'];state=gate.pilot.clone_state(ck['state'])
 if cursor>=tokens.shape[1]-1:tokens.advance();cursor=0;state=None
 n=min(args.chunk,tokens.shape[1]-1-cursor);buf=tokens[:,cursor:cursor+n+1];results=[]
 for model,optimizer,config in [(parent,opt,args),(grown,go,config)]:
  model.train();optimizer.zero_grad();gen=torch.Generator().set_state(ck['generator']);teacher=torch.Generator().set_state(ck['teacher_generator']);loss,st,audits=gate.pilot.raw_step(model,gate.pilot.clone_state(state),gen,teacher,config,buf,ck['step']+1,with_new=False);optimizer.step();optimizer.zero_grad();results.append((loss,st,audits,gen.get_state(),teacher.get_state()))
 assert torch.equal(results[0][0],results[1][0]);assert gate.equal_tree(results[0][2:],results[1][2:]);names=dict(parent.named_parameters());new=dict(grown.named_parameters());errors={}
 for name,value in names.items():
  target=name.replace('readout.','readout.base.',1) if name.startswith('readout.') else name;other=new[target]
  if other.shape!=value.shape:other=other[:value.shape[0]]
  errors[name]=float((value-other).detach().abs().max())
 maximum=max(errors.values());assert maximum<1e-7,sorted(errors.items(),key=lambda x:-x[1])[:5]
 for old,new in mapping.items():assert go.receiver.local.versions[new]==opt.receiver.local.versions[old]
 sources=['experiments/credit/protected_growth_bank_contract.py','sleeping_machines/learning/protected_growth_banks.py','sleeping_machines/protected_depth_language.py','experiments/credit/protected_depth_learning_pilot.py'];record=dict(status='completed',battle='R1',old_bank_count=len(mapping),new_bank_count=len(new_caps),old_caps_moments_versions_preserved=True,old_next_update_parameter_error=maximum,paired_old_supported_teacher_audits=results[0][2],wall_s=time.monotonic()-began,source_sha256={n:gate.pilot.driver.sha(ROOT/n) for n in sources},input_sha256={str(p.relative_to(ROOT)):gate.pilot.driver.sha(p) for p in [base.with_suffix('.json'),base.with_suffix('.pt')]},scope='Strongest acquired6.06556D2 receiver learner grows to protectedD4 at zero correction. Actual next update includes original sampled supportedteacher; oldcaps/moments/versions and transport-prefix bank preserved. Newcaps1 are explicit numerical-contract placeholders, NOT production calibration. Added branch learning/conditionalemission and frozen old-credit ownership still need integration; no fit/quality or hardware claim.')
 with (ROOT/f'experiments/results/credit/{tag}.json').open('x') as f:json.dump(record,f,indent=2)
 print(json.dumps({k:v for k,v in record.items() if k not in ['source_sha256','input_sha256']}),flush=True)
if __name__=='__main__':main()
