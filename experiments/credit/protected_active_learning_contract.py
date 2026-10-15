"""Keep acquired learning trajectory intact while a nonzero new correction learns."""
import argparse,copy,json,sys,time
from pathlib import Path
from types import SimpleNamespace
import torch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'experiments'));sys.path.insert(0,str(Path(__file__).parent))
import protected_depth_local_gate as gate
from sleeping_machines.learning.protected_growth_banks import ProtectedGrowthAdamW,migrate_parent_banks

def main():
 p=argparse.ArgumentParser();p.add_argument('--tag',required=True);tag=p.parse_args().tag;torch.set_num_threads(1);began=time.monotonic();parent,_,args,train,_,_=gate.restore_arm('parent');base=ROOT/'experiments/results/token_language/aws_protected_depth_parent_2m_20261014_s6_v1';ck=torch.load(base.with_suffix('.pt'),weights_only=False);parent.load_state_dict(ck['model']);opt=gate.CalibratedReceiverBankAdamW(parent,ck['optimizer']['bank_caps'],lr=args.lr,weight_decay=args.weight_decay);opt.load_state_dict(copy.deepcopy(ck['optimizer']));config=SimpleNamespace(**vars(args));config.depth+=2;counts=gate.pilot.driver.LaneTokens(train,0,args.train_tokens,args.lanes).counts();order=torch.argsort(counts,descending=True,stable=True)
 with torch.random.fork_rng():torch.manual_seed(61);fresh=gate.pilot.driver.Model(config,order,counts)
 grown=gate.pilot.grow_protected_language(parent,fresh);probe=ProtectedGrowthAdamW(grown,args.depth,lr=args.lr,weight_decay=args.weight_decay);oldnames={b.replace('readout.','readout.base.',1) if b.startswith('readout.') else b for b in opt.receiver.mapping};go,mapping=migrate_parent_banks(opt,grown,{b:1. for b in set(probe.receiver.mapping)-oldnames})
 with torch.no_grad():grown.readout.correction.weight.normal_(0,.01,generator=torch.Generator().manual_seed(73))
 initial=grown.readout.correction.weight.detach().clone();sp=sg=None;gp=torch.Generator().manual_seed(69);gg=torch.Generator().manual_seed(69);tokens=gate.pilot.driver.LaneTokens(train,12000000,2048,args.lanes);rows=[]
 for step in range(3):
  start=step*args.chunk;buf=tokens[:,start:start+args.chunk+1];opt.zero_grad();go.zero_grad();xp,sp,_=parent(buf[:,:-1],sp,gp,args);xg,sg,_=grown(buf[:,:-1],sg,gg,config);lp=parent.readout.nll(xp,buf[:,1:]).mean();base_loss=grown.readout.base.nll(xg[...,:grown.readout.parent_width],buf[:,1:]).mean();aug=grown.readout.nll(xg,buf[:,1:]).mean();assert torch.equal(lp,base_loss);lp.backward();opt.step();go.step_separated(base_loss,aug,set(mapping.values()));old=dict(parent.named_parameters());new=dict(grown.named_parameters());maximum=0.
  for name,p in old.items():
   target=name.replace('readout.','readout.base.',1) if name.startswith('readout.') else name;q=new[target]
   if p.shape!=q.shape:q=q[:p.shape[0]]
   maximum=max(maximum,float((p-q).detach().abs().max()))
  assert maximum<1e-7,maximum;assert torch.equal(gp.get_state(),gg.get_state())
  for name,target in mapping.items():assert go.receiver.local.versions[target]==opt.receiver.local.versions[name]
  rows.append(dict(update=step+1,parent_parameter_error=maximum,parent_loss=float(lp.detach()),augmented_loss=float(aug.detach())));sp=gate.pilot.detach(sp);sg=gate.pilot.detach(sg)
 movement=float((initial-grown.readout.correction.weight).norm());assert movement>0
 names=['experiments/credit/protected_active_learning_contract.py','sleeping_machines/learning/protected_growth_banks.py','sleeping_machines/protected_depth_language.py'];record=dict(status='completed',battle='R1',rows=rows,new_correction_movement=movement,source_sha256={n:gate.pilot.driver.sha(ROOT/n) for n in names},wall_s=time.monotonic()-began,scope='Threeactual factual-only localAdam updates on sameTRAIN: nonzero correction learns while old parameter trajectory/caps/versions/RNG match continuedparent. Newcaps1 are numerical placeholders; no productioncalibration or integrated alternative-teacher/emission policy or qualitygain claim.')
 with (ROOT/f'experiments/results/credit/{tag}.json').open('x') as f:json.dump(record,f,indent=2)
 print(json.dumps(record),flush=True)
if __name__=='__main__':main()
