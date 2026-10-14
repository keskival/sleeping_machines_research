"""R1 two-policy continuation from an identical completed native learner."""
import argparse,json,math,sys,time,resource,platform
from pathlib import Path
from types import SimpleNamespace
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'experiments/credit'))
import integrated_receiver_token_language as driver
from sleeping_machines.learning.receiver_bank_adamw import ReceiverBankAdamW
from sleeping_machines.learning.calibrated_receiver_adamw import CalibratedReceiverBankAdamW
from sleeping_machines.depth_credit_allocator import DepthCreditAllocator
from receiver_allocated_future_world import ReceiverAllocatedFutureWorld

def verify_cap_law():
 from sleeping_machines.learning.calibrated_receiver_adamw import CalibratedLocalCreditBanks
 from sleeping_machines.learning.local_credit_banks import CreditPacket
 p=torch.nn.Parameter(torch.tensor([.4,-.2]));q=torch.nn.Parameter(p.detach().clone());banks=CalibratedLocalCreditBanks([('b',p)],{'b':.05},lr=.003,weight_decay=.01);reference=torch.optim.AdamW([q],lr=.003,weight_decay=.01)
 for i in range(3):
  g=torch.tensor([2.,-1.])*(i+1);q.grad=g.clone();torch.nn.utils.clip_grad_norm_([q],.05);reference.step();reference.zero_grad(set_to_none=True);banks.apply(CreditPacket('b',i,g,f'u{i}'));assert torch.equal(p,q)
 return 'three local-cap AdamW updates exact versus manual torch clip'

def main():
 cap_gate=verify_cap_law()
 p=argparse.ArgumentParser();p.add_argument('--tag',required=True);p.add_argument('--policy',choices=['global','calibrated'],required=True);p.add_argument('--updates',type=int,required=True);p.add_argument('--eval-every',type=int,default=1024);p.add_argument('--resume');a=p.parse_args();assert a.updates>0;torch.set_num_threads(1);started=time.monotonic()
 out=ROOT/f'experiments/results/token_language/{a.tag}';assert not out.with_suffix('.json').exists()
 if out.with_suffix('.pt').exists() and not a.resume:raise FileExistsError('Use exact resume or a unique tag')
 ref='aws_integrated_receiver_d8_8m_20261014_s6_v1';base=ROOT/f'experiments/results/token_language/{ref}';r=json.loads(base.with_suffix('.json').read_text());assert r['status']=='completed'
 for name,digest in r['source_sha256'].items():assert driver.sha(ROOT/name)==digest,name
 cap_path=ROOT/'experiments/results/credit/aws_same_cut_local_clip_calibration_20261014_v1.json';calibration=json.loads(cap_path.read_text());assert calibration['status']=='completed'
 for name,digest in {**calibration['source_sha256'],**calibration['input_sha256']}.items():assert driver.sha(ROOT/name)==digest,name
 args=SimpleNamespace(**r['args']);args.tag=a.tag;args.steps+=a.updates;args.eval_every=a.eval_every;args.resume=None;args.stop_after_step=None
 tr=driver.load_tokens(args.train_file);dv=driver.load_tokens(args.dev_file);counts=driver.LaneTokens(tr,0,args.train_tokens,args.lanes).counts();torch.manual_seed(args.seed);model=driver.Model(args,torch.argsort(counts,descending=True,stable=True),counts)
 ck=torch.load(base.with_suffix('.pt'),map_location='cpu',weights_only=False);model.load_state_dict(ck['model'])
 if a.policy=='global':opt=ReceiverBankAdamW(model,lr=args.lr,weight_decay=args.weight_decay)
 else:opt=CalibratedReceiverBankAdamW(model,calibration['calibration']['caps'],lr=args.lr,weight_decay=args.weight_decay)
 # Explicit migration of identical old moments/versions into a changed clipping law.
 ReceiverBankAdamW.load_state_dict(opt,ck['optimizer'])
 for name in ('site_generator','position_generator','local_generator','alternative_generator'):getattr(model,name).set_state(ck[name])
 gen=torch.Generator().set_state(ck['generator']);teacher_gen=torch.Generator().set_state(ck['teacher_generator']);torch.set_rng_state(ck['torch_rng']);allocator=DepthCreditAllocator(args.depth,args.heads,args.credit_window,args.allocation_exploration,args.allocation_decay);allocator.load_state_dict(ck['depth_credit_allocator']);state=driver.clone_numeric_state(ck['state'])
 source=(ROOT/'experiments/integrated_receiver_token_language.py').read_text();begin=source.index('        use_future = bool(');end=source.index('        loss.backward()',begin)+len('        loss.backward()');body='\n'.join(line[4:] for line in source[begin:end].splitlines())
 text='def kernel(model,opt,state,gen,teacher_gen,allocator,a,step,buf,policy):\n    n=buf.shape[1]-1\n    opt.zero_grad(set_to_none=True)\n    teacher_calls=teacher_replay_tokens=teacher_scored_targets=0\n'+body+'\n    if policy=="global":torch.nn.utils.clip_grad_norm_(model.parameters(),1.)\n    opt.step()\n    return token_nll.detach(),detach(state),(teacher_calls,teacher_replay_tokens,teacher_scored_targets)\n';ns=dict(driver.__dict__);exec(compile(text,'source-bound-clipping-continuation','exec'),ns);kernel=ns['kernel']
 train=driver.BlockLaneTokens(tr,args.train_tokens,args.lanes,args.exposure,args.steps,args.chunk);train.block=ck['training_block'];cursor=ck['cursor'];dev=driver.interval_tensor(dv,args.dev_offset,args.dev_tokens,args.eval_lanes)
 curve=[dict(step=ck['step'],dev_nll=driver.evaluate(model,dev,args))];assert curve[0]['dev_nll']==r['curve'][-1]['dev_nll'];work=list(ck['teacher_work']);targets=0;training_s=0.;parity=None;start_index=0
 checkpoint_binding=dict(driver_sha256=driver.sha(Path(__file__)),parent_checkpoint_sha256=driver.sha(base.with_suffix('.pt')),caps_sha256=driver.sha(cap_path),policy=a.policy,updates=a.updates,eval_every=a.eval_every,tag=a.tag)
 if a.resume:
  saved=torch.load(a.resume,map_location='cpu',weights_only=False)
  if saved['binding']!=checkpoint_binding:raise ValueError('Continuation recovery changed source/parent/caps/settings')
  model.load_state_dict(saved['model']);opt.load_state_dict(saved['optimizer']);state=driver.clone_numeric_state(saved['state']);cursor=saved['cursor'];train.block=saved['training_block'];allocator.load_state_dict(saved['depth_credit_allocator']);gen.set_state(saved['generator']);teacher_gen.set_state(saved['teacher_generator']);torch.set_rng_state(saved['torch_rng'])
  for name in ('site_generator','position_generator','local_generator','alternative_generator'):getattr(model,name).set_state(saved[name])
  work=saved['teacher_work'];targets=saved['added_targets'];training_s=saved['added_training_s'];curve=saved['curve'];start_index=saved['update_index'];parity=saved['global_first_update_parity']
 for index in range(start_index,a.updates):
  if cursor>=train.shape[1]-1:cursor=0;state=None;train.advance()
  n=min(args.chunk,train.shape[1]-1-cursor);buf=train[:,cursor:cursor+n+1];step=ck['step']+index+1
  witness=None
  if index==0 and a.policy=='global':
   witness=ReceiverAllocatedFutureWorld(model,opt,state,gen,args,teacher_generator=teacher_gen,next_step=step,allocator=allocator);expected=witness.advance(buf[:,:-1],buf[:,1:])
  before=time.monotonic();loss,state,cost=kernel(model,opt,state,gen,teacher_gen,allocator,args,step,buf,a.policy);training_s+=time.monotonic()-before
  assert bool(torch.isfinite(loss).all()) and all(bool(torch.isfinite(p).all()) for p in model.parameters())
  if witness is not None:
   assert torch.equal(loss.reshape_as(expected),expected)
   assert all(torch.equal(p,witness.model.state_dict()[name]) for name,p in model.named_parameters());parity='exact first-update weights/loss against frozen production reference'
  work=[x+y for x,y in zip(work,cost)];cursor+=n;targets+=n*args.lanes
  if (index+1)%a.eval_every==0 or index+1==a.updates:
   row=dict(step=step,dev_nll=driver.evaluate(model,dev,args),train_wall_s=training_s,added_targets=targets);curve.append(row);print(json.dumps(row),flush=True)
   snapshot=dict(binding=checkpoint_binding,added_targets=targets,added_training_s=training_s,curve=curve,update_index=index+1,global_first_update_parity=parity,model=model.state_dict(),optimizer=opt.state_dict(),state=state,cursor=cursor,step=step,training_block=train.block,depth_credit_allocator=allocator.state_dict(),generator=gen.get_state(),teacher_generator=teacher_gen.get_state(),torch_rng=torch.get_rng_state(),teacher_work=work,policy=a.policy,cap_sha256=driver.sha(cap_path),args=vars(args))
   for name in ('site_generator','position_generator','local_generator','alternative_generator'):snapshot[name]=getattr(model,name).get_state()
   tmp=out.with_suffix('.pt.tmp');torch.save(snapshot,tmp);tmp.replace(out.with_suffix('.pt'))
 sources=dict(r['source_sha256'])
 for name in ['experiments/calibrated_clip_continuation.py','sleeping_machines/learning/calibrated_receiver_adamw.py','experiments/credit/receiver_allocated_future_world.py','sleeping_machines/learning/clone_receiver_learner.py']:sources[name]=driver.sha(ROOT/name)
 inputs={str(p.relative_to(ROOT)):driver.sha(p) for p in [base.with_suffix('.json'),base.with_suffix('.pt'),cap_path]}
 result=dict(status='completed',battle='R1',args=vars(args),policy=a.policy,parent=ref,added_updates=a.updates,added_targets=targets,inherited_training_s=r['train_wall_s_total'],added_training_s=training_s,train_wall_s_total=r['train_wall_s_total']+training_s,presentations_total=r['presentations_total']+targets,curve=curve,parameters=r['parameters'],core_parameters=r['core_parameters'],readout_parameters=r['readout_parameters'],teacher_work=dict(admitted_calls=work[0],replayed_tokens=work[1],scored_targets=work[2]),local_cap_gate=cap_gate,global_first_update_parity=parity,calibration_wall_s=calibration['wall_s'] if a.policy=='calibrated' else 0.,calibration_factual_targets=16384 if a.policy=='calibrated' else 0,source_sha256=sources,input_sha256=inputs,host=platform.node(),max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,process_wall_s=time.monotonic()-started,scope='Matched own-model continuation, same initial complete8Mlearner/data/order/updates/teaching; clipping only differs. Calibrated arm also pays inherited47.759sTRAIN calibration and its extra target exposure. Final weights; no publicvalidation or Transformer comparison. Version-bound full checkpoint resume supported; interrupted/resumed numerical gate still required before preemptible placement.')
 with out.with_suffix('.json').open('x') as f:json.dump(result,f,indent=2)
if __name__=='__main__':main()
