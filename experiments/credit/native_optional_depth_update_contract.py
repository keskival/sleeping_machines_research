"""R1 old next-update containment and actual new-stage update utility witness.

One TRAIN panel, no tuning, no quality fit. New-stage utility isolates their
actual Adam displacement by restoring all acquired coordinates after the
hypothetical grown update. This is a conditional diagnostic, not a production
activation teacher or a convergence result.
"""
import argparse,copy,json,resource,sys,time
from pathlib import Path
from types import SimpleNamespace
import torch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'experiments'))
import integrated_receiver_token_language as driver
from sleeping_machines.learning.receiver_bank_adamw import ReceiverBankAdamW
from sleeping_machines.learning.optional_depth_learner import migrate_optional_depth_learner
from sleeping_machines.optional_temporal_depth import optional_depth_features
from sleeping_machines.ordered_token_lookup import ordered_content
from sleeping_machines.ordered_token_episodes import token_features

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--tag',required=True);a=ap.parse_args();torch.set_num_threads(1);start=time.monotonic()
 base=ROOT/'experiments/results/token_language/aws_integrated_receiver_d8_8m_20261014_s6_v1';r=json.loads(base.with_suffix('.json').read_text());args=SimpleNamespace(**r['args']);tr=driver.load_tokens(args.train_file);counts=driver.LaneTokens(tr,0,args.train_tokens,args.lanes).counts();order=torch.argsort(counts,descending=True,stable=True)
 torch.manual_seed(args.seed);parent=driver.Model(args,order,counts);ck=torch.load(base.with_suffix('.pt'),weights_only=False);parent.load_state_dict(ck['model']);opt=ReceiverBankAdamW(parent,lr=args.lr,weight_decay=args.weight_decay);opt.load_state_dict(ck['optimizer']);opt.zero_grad()
 bigger=SimpleNamespace(**vars(args));bigger.depth=args.depth+2
 with torch.random.fork_rng():torch.manual_seed(501);fresh=driver.Model(bigger,order,counts)
 grown,newopt,_=migrate_optional_depth_learner(parent,opt,fresh.core)
 panel=driver.LaneTokens(tr,262144,136,8)[:,:];inputs=panel[:,:16];targets=panel[:,1:17]
 def loss(model,fn):
  x,_,_=fn(model.core,inputs,None,torch.Generator().manual_seed(92),eos=None,route_credit='none',token_content=ordered_content(model.core.content.weight,inputs));return model.readout.nll(x,targets).mean()
 def step(model,opt,fn):
  opt.zero_grad();l=loss(model,fn);l.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step();opt.zero_grad();return float(l.detach())
 before=step(parent,opt,token_features);neutral=step(grown,newopt,optional_depth_features);assert before==neutral
 errors=[]
 for name,p in parent.named_parameters():
  q=dict(grown.named_parameters())[name];q=q[:p.shape[0]] if p.shape!=q.shape else q
  error=float((p-q).detach().abs().max());errors.append(error);assert torch.allclose(p,q,rtol=1e-6,atol=1e-7),(name,error)
 for bank,version in opt.receiver.local.versions.items():assert newopt.receiver.local.versions[bank]==version
 # Reset to the acquired state, then evaluate an activated alternative.
 opt.zero_grad();parent.load_state_dict(ck['model']);opt.load_state_dict(ck['optimizer']);grown,newopt,_=migrate_optional_depth_learner(parent,opt,fresh.core);grown.core.execution_depth=grown.core.depth
 old_parameters={n:p.detach().clone() for n,p in parent.named_parameters()};new_rng=grown.core.new_depth_generator.get_state();old_loss=float(loss(grown,optional_depth_features).detach());grown.core.new_depth_generator.set_state(new_rng)
 step(grown,newopt,optional_depth_features)
 with torch.no_grad():
  for name,p in grown.named_parameters():
   if name in old_parameters:
    old=old_parameters[name]
    if p.shape==old.shape:p.copy_(old)
    else:p[:old.shape[0]].copy_(old)
 grown.core.new_depth_generator.set_state(new_rng);after=float(loss(grown,optional_depth_features).detach())
 # Verify restoring old coordinates protects the neutral prediction even
 # after the hypothetical new-stage update. This says nothing about deadlines.
 grown.core.execution_depth=args.depth;parent_neutral=float(loss(parent,token_features).detach());grown_neutral=float(loss(grown,optional_depth_features).detach());assert parent_neutral==grown_neutral
 names=['experiments/credit/native_optional_depth_update_contract.py','sleeping_machines/learning/optional_depth_learner.py','sleeping_machines/optional_temporal_depth.py','sleeping_machines/learning/receiver_bank_adamw.py','experiments/integrated_receiver_token_language.py']
 record=dict(status='completed',battle='R1',neutral_next_update_parameter_max_error=max(errors),neutral_next_update_versions_preserved=True,alternative_before=old_loss,alternative_after_new_coordinates_only=after,new_stage_actual_update_gain=old_loss-after,protected_parent_loss=parent_neutral,protected_parent_after_new_update_loss=grown_neutral,useful_new_stage_update=after<old_loss,source_sha256={n:driver.sha(ROOT/n) for n in names},input_sha256={str(p.relative_to(ROOT)):driver.sha(p) for p in [base.with_suffix('.json'),base.with_suffix('.pt')]},wall_s=time.monotonic()-start,max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,scope='AcquiredD8→optionalD10,128TRAINtargets,singlepairedrace tape. Factual-only oldlearning update at inheritedclip1/Adam; oldparameter tolerance1e-6/1e-7. New-stage conditional actualAdam displacement from fullgradient clips, then allacquiredcoordinates restored; newsharedtransport coords inheritoldbank step. No fittedgain, calibration/activationteacher or generalization claim.')
 out=ROOT/f'experiments/results/diagnostics/{a.tag}.json'
 with out.open('x') as f:json.dump(record,f,indent=2)
 print(json.dumps(record),flush=True)
if __name__=='__main__':main()
