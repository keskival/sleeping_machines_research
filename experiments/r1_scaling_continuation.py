"""R1 scaling: continue the strongest native learner on fresh TRAIN tokens with a milestone DEV curve."""
import argparse,copy,json,resource,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'experiments/credit'))
import torch
import protected_depth_local_gate as gate
from public_token_scoring import score
PARENT='experiments/results/token_language/aws_memory_policy_corpus_eos_2m_20261015_v1'
PARENT_UPDATES=4096
MILESTONES=(256,4096,8192,16384,32768,65536,131072)

def dev_score(model,state_rng,dev,args):
 # Score a frozen copy and restore the global RNG so DEV milestones never perturb training.
 rng=torch.get_rng_state();frozen=copy.deepcopy(model)
 for name,value in state_rng.items():getattr(frozen,name).set_state(value)
 out=score(frozen,dev,args,gate.pilot.driver,25165824,8192,1024,8);torch.set_rng_state(rng);return out

def main():
 p=argparse.ArgumentParser();p.add_argument('--tag',required=True);p.add_argument('--updates',type=int,required=True);a=p.parse_args();assert a.updates in MILESTONES;torch.set_num_threads(1);began=time.monotonic()
 model,_,args,train,_,_=gate.restore_arm('parent');base=ROOT/PARENT;parent=torch.load(base.with_suffix('.pt'),weights_only=False);model.load_state_dict(parent['model'])
 assert parent['binding']['policy']=='shared' and parent['binding']['reset_policy']=='corpus_eos'
 opt=gate.CalibratedReceiverBankAdamW(model,parent['optimizer']['bank_caps'],lr=args.lr,weight_decay=args.weight_decay)
 stream_start=parent['binding']['stream_start']+PARENT_UPDATES*args.chunk*args.lanes+args.lanes;length=a.updates*args.chunk*args.lanes+args.lanes
 binding=dict(policy='shared',reset_policy='corpus_eos',parent_sha256=gate.pilot.driver.sha(base.with_suffix('.pt')),stream_start=stream_start,stream_total_updates=a.updates,driver_sha256=gate.pilot.driver.sha(Path(__file__)))
 prefix=ROOT/f'experiments/results/token_language/{a.tag}';progress=prefix.with_suffix('.progress.pt');assert not prefix.with_suffix('.json').exists()
 ck=parent;offset=0;elapsed=0.;work=dict(calls=0,replayed_targets=0,scored_targets=0);curve=[];milestones=[];loss_sum=0.;count=0;activation_observed=activation_clipped=0
 if progress.exists():
  ck=torch.load(progress,weights_only=False);assert ck['binding']==binding;model.load_state_dict(ck['model']);offset=ck['added_updates'];elapsed=ck['training_s'];work=ck['teacher_work'];curve=ck['curve'];milestones=ck['milestones'];loss_sum=ck['loss_sum'];count=ck['targets'];activation_observed,activation_clipped=ck['activation']
 opt.load_state_dict(copy.deepcopy(ck['optimizer']))
 for name,rng in ck['owned_rng'].items():getattr(model,name).set_state(rng)
 gen=torch.Generator().set_state(ck['generator']);teacher=torch.Generator().set_state(ck['teacher_generator']);torch.set_rng_state(ck['torch_rng']);state=gate.pilot.clone_state(ck['state'])
 tokens=gate.pilot.driver.LaneTokens(train,stream_start,length,args.lanes);assert tokens.shape[1]==a.updates*args.chunk+1 and stream_start+length<=len(train)
 dev=gate.pilot.driver.load_tokens(args.dev_file);model.train()
 if offset==0:
  gate_score=dev_score(model,gate.private_state(model),dev,args)['nll'];assert abs(gate_score-5.978387557528713)<1e-9,gate_score;print(json.dumps(dict(parent_replay_dev_nll=gate_score)),flush=True)
 segment=time.monotonic()
 for index in range(offset,a.updates):
  cursor=index*args.chunk;buf=tokens[:,cursor:cursor+args.chunk+1];opt.zero_grad();loss,state,audits=gate.pilot.raw_step(model,state,gen,teacher,args,buf,parent['step']+index+1);loss_sum+=float(loss.double().sum());count+=loss.numel();opt.step();opt.zero_grad()
  for receipt in opt.receiver.local.receipts:activation_observed+=1;activation_clipped+=int(receipt['scale']<1)
  for audit in audits:work['calls']+=1;work['replayed_targets']+=audit['replayed_targets'];work['scored_targets']+=audit['scored_targets']
  done=index+1
  if done%256==0 or done==a.updates:
   training_s=elapsed+time.monotonic()-segment;row=dict(update=done,added_targets=count,prequential_nll=loss_sum/count,training_s=training_s);curve.append(row)
   if done in MILESTONES:
    started=time.monotonic();scored=dev_score(model,gate.private_state(model),dev,args);milestones.append(dict(update=done,added_targets=count,scored=scored,training_s=training_s,scoring_s=time.monotonic()-started));print(json.dumps(dict(milestone=done,dev_nll=scored['nll'])),flush=True)
   saved=dict(model=model.state_dict(),optimizer=opt.state_dict(),state=state,generator=gen.get_state(),teacher_generator=teacher.get_state(),owned_rng=gate.private_state(model),torch_rng=torch.get_rng_state(),step=parent['step']+done,added_updates=done,targets=count,loss_sum=loss_sum,teacher_work=work,training_s=training_s,curve=curve,milestones=milestones,activation=(activation_observed,activation_clipped),binding=binding)
   tmp=prefix.with_suffix('.tmp');torch.save(saved,tmp);tmp.replace(progress);print(json.dumps(row),flush=True)
 final=prefix.with_suffix('.pt');torch.save(saved,final)
 names=['experiments/r1_scaling_continuation.py','experiments/credit/protected_depth_local_gate.py','experiments/credit/protected_depth_learning_pilot.py','experiments/multidepth_readout_token_language.py','experiments/public_token_scoring.py']
 record=dict(status='completed',battle='R1',policy='shared',reset_policy='corpus_eos',parent_dev_nll=5.978387557528713,added_updates=a.updates,added_targets=count,prequential_nll=loss_sum/count,milestones=milestones,scored=milestones[-1]['scored'],curve=curve,training_s=saved['training_s'],teacher_work=work,local_clip_activation_fraction=activation_clipped/max(activation_observed,1),final_weights=str(final.relative_to(ROOT)),source_sha256={n:gate.pilot.driver.sha(ROOT/n) for n in names},input_sha256={str(base.with_suffix('.pt').relative_to(ROOT)):binding['parent_sha256']},binding=binding,max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,process_wall_s=time.monotonic()-began,
  scope='Data-scaling continuation of the strongest native R1 learner (shared lane clocks, corpus-EOS training boundary, calibrated local caps, supported teacher, full capacity/keys/values/writes, persistent numerical state, gradient window16). Eight contiguous fresh FineWeb GPT-2 TRAIN streams after the parent stream; scores each target before its update. Milestone DEV uses a deep-copied frozen learner on the same 8192 reused DEV targets/1024 history/no EOS reset as the parent stage; training state and RNG are untouched. Single seed6; no public validation or external fit.')
 with prefix.with_suffix('.json').open('x') as f:json.dump(record,f,indent=2)
if __name__=='__main__':main()
