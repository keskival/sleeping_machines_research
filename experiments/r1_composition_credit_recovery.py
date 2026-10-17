"""Paired complete-native continuations: conditional content credit, then causal DEV adaptation.
No external architecture, TEST, sweep, capacity restriction or memory reset in online evaluation.
"""
import argparse,ast,copy,shutil,gc,hashlib,io,json,math,os,resource,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'experiments')]
import numpy as np
import torch
from r1_cerl import load_fineweb
from r1_local_native_deep import LocalNativeDeepLM
from experiments.native_query_migration import migrate
from sleeping_machines.producer_replay_native_race import enable_producer_credit
from sleeping_machines.document_boundary_causes import enable_document_relevance
from sleeping_machines.coupled_native_query_credit import enable_coupled_query_credit
from sleeping_machines.compositional_producer_credit import enable_compositional_credit
from sleeping_machines.sampled_compositional_credit import enable_sampled_compositional_credit
from experiments.hierarchy.native_composition_data import source_hashes
from tests.test_native_query_migration import identical
from experiments.credit.persistent_tape_transition import transition_batch


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def atomic(p,value):
 tmp=p.with_suffix('.writing');torch.save(value,tmp);os.replace(tmp,p)
def source_migration(parent):
 changes=[]
 for name,digest in parent['source_sha256'].items():
  if sha(ROOT/name)==digest:continue
  assert name=='experiments/r1_cerl.py',name
  old=subprocess.check_output(['git','show','2f662dd9a:'+name]);assert hashlib.sha256(old).hexdigest()==digest
  def used(s):
   t=ast.parse(s);c=next(n for n in t.body if isinstance(n,ast.ClassDef) and n.name=='CERL')
   return {n.name:ast.dump(n,include_attributes=False) for n in c.body if isinstance(n,ast.FunctionDef) and n.name in ['__init__','unit_score','wake']}
  assert used(old)==used((ROOT/name).read_bytes())
  changes.append(dict(file=name,old=digest,new=sha(ROOT/name),proof='Native indexed head uses unchanged constructor/wake/unit_score and direct parameter laws; changed optional lane/race hooks are not called.'))
 return changes


def evaluate(model,val,windows):
 old=model.training;model.eval();rng=torch.get_rng_state();rows=[];started=time.monotonic()
 try:
  with torch.no_grad():
   for start,length in windows:
    w=torch.from_numpy(np.asarray(val[start:start+length+1]).astype(np.int64));state=model.new_state();g=torch.Generator().manual_seed(113);alt=torch.Generator().manual_seed(114);values=[]
    for b in range(0,length,32):
     nll,_,state,_,_=model.stream_nll(w[b:b+32],w[b+1:b+33],state,g,alt,learn=False,stats=False);values+=nll.detach().double().tolist()
    seen={int(w[0])};novel=[]
    for target in w[1:]:novel.append(int(target) not in seen);seen.add(int(target))
    rows.append(dict(start=start,targets=length,nll=sum(values)/length,loss_sum=sum(values),novel_loss_sum=sum(v for v,m in zip(values,novel) if m),novel_targets=sum(novel)))
 finally:torch.set_rng_state(rng);model.train(old)
 count=sum(r['targets'] for r in rows);n=sum(r['novel_targets'] for r in rows)
 return dict(nll=sum(r['loss_sum'] for r in rows)/count,targets=count,novel_nll=sum(r['novel_loss_sum'] for r in rows)/n if n else None,novel_targets=n,windows=rows,wall_s=time.monotonic()-started,scope='Frozen weights; state persists within prescribed windows and resets only for this separately labelled fixed-window diagnostic.')


def fit_arm(a,arm,anchor,digest,train,val):
 stem=ROOT/f'experiments/results/token_language/{a.tag}_{arm}';out=stem.with_suffix('.json');ckpt=stem.with_suffix('.resume.pt')
 if out.exists():return json.loads(out.read_text())
 began=time.monotonic();parent=torch.load(anchor,map_location='cpu',weights_only=False);migration=source_migration(parent)
 model=LocalNativeDeepLM(50257,parent['model']['causes.class_of'],parent['model']['causes.class_words'],width=64,depth=8,heads=4,pool=16,concept_units=4096,topk=4,explore=2,dk=32,packed=True)
 opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01);model,opt,state,_=migrate(model,opt,parent['state'])
 enable_producer_credit(model,state);enable_document_relevance(model,opt);model.load_state_dict(parent['model']);opt.load_state_dict(parent['optimizer']);state=parent['state'];enable_coupled_query_credit(model,state)
 if arm=='child':
  enable_compositional_credit(model,state);enable_sampled_compositional_credit(model,state)
 g=torch.Generator();g.set_state(parent['g']);alt=torch.Generator();alt.set_state(parent['alt']);torch.set_rng_state(parent['rng'])
 start=parent['cursor'];cursor=start;end=start+a.extra_targets;inherited_wall=parent['wall_s'];online=0;history=[];online_losses=[];initial=None;frozen=None;past=0.;verified=False;audit=0;del parent
 sources=source_hashes()
 for f in ['experiments/r1_composition_credit_recovery.py','experiments/credit/persistent_tape_transition.py','experiments/native_query_migration.py','experiments/r1_local_native_deep.py','experiments/r1_cerl.py','tests/test_native_query_migration.py']:sources[f]=sha(ROOT/f)
 settings=dict(arm=arm,extra_targets=a.extra_targets,online_targets=a.online_targets,batch=32,parent_sha256=digest,child_credit='full-depth inverse-inclusion path' if arm=='child' else 'one-hop joint producer',online_start=30000000,lr=.001,weight_decay=.01)
 counters={k:state[k] for k in state if isinstance(state[k],int)}
 def snapshot():return dict(model=model.state_dict(),opt=opt.state_dict(),state=state,g=g.get_state(),alt=alt.get_state(),rng=torch.get_rng_state(),cursor=cursor,online=online,history=history,online_losses=online_losses,initial=initial,frozen=frozen,wall=past+time.monotonic()-began,verified=verified,audit=audit,settings=settings,source_sha256=sources)
 def restore(s):
  nonlocal state,cursor,online,history,online_losses,initial,frozen,past,verified,audit
  assert s['settings']==settings
  if s['source_sha256']!=sources:
   assert a.recover_tag and cursor==start
   for name,value in s['source_sha256'].items():assert sha(ROOT/name)==value,name
   assert s['cursor']==end and s['online']==0 and s['frozen'] is not None
   assert s['verified'] and s['settings']['parent_sha256']==digest
   print(json.dumps(dict(arm=arm,recovered_completed_training=True,checkpoint_sha256=sha(ckpt))),flush=True)
  model.load_state_dict(s['model']);opt.load_state_dict(s['opt']);state=s['state'];g.set_state(s['g']);alt.set_state(s['alt']);torch.set_rng_state(s['rng'])
  cursor=s['cursor'];online=s['online'];history=s['history'];online_losses=s['online_losses'];initial=s['initial'];frozen=s['frozen'];past=s['wall'];verified=s['verified'];audit=s['audit']
 def update(w):
  model.train();opt.zero_grad(set_to_none=True);nll,obj,_,packets,_=model.stream_nll(w[:-1],w[1:],state,g,alt,stats=False)
  assert bool(torch.isfinite(obj));obj.backward();model.deliver_credit(packets,state)
  assert all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in model.parameters())
  norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step()
  return dict(nll=float(nll.detach().double().mean()),losses=nll.detach().double().tolist(),norm=float(norm))
 def tokens(data,b):return torch.from_numpy(np.asarray(data[b:b+33]).astype(np.int64))
 if ckpt.exists():restore(torch.load(ckpt,map_location='cpu',weights_only=False));began=time.monotonic()
 if initial is None:initial=evaluate(model,val,[(10485760,1024)]);atomic(ckpt,snapshot())
 if not verified:
  saved=torch.load(ckpt,map_location='cpu',weights_only=False);first=update(tokens(train,cursor));expected=io.BytesIO();torch.save((model.state_dict(),opt.state_dict(),state,g.get_state(),alt.get_state(),torch.get_rng_state()),expected)
  charged=past+time.monotonic()-began;restore(saved);past=charged;began=time.monotonic();again=update(tokens(train,cursor));expected.seek(0)
  assert first==again and identical(torch.load(expected,weights_only=False),(model.state_dict(),opt.state_dict(),state,g.get_state(),alt.get_state(),torch.get_rng_state()))
  cursor+=32;audit+=32;verified=True;history.append(dict(cursor=cursor,nll=again['nll'],preclip_norm=again['norm'],clip_multiplier=min(1.,1/(again['norm']+1e-6))));del saved,expected
  assert resource.getrusage(resource.RUSAGE_SELF).ru_maxrss<2700000,'32target recovery requires reduced-concurrency reservation'
  atomic(ckpt,snapshot());print(json.dumps(dict(arm=arm,exact_recovery=True,max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)),flush=True)
 while cursor<end:
  row=update(tokens(train,cursor));cursor+=32;history.append(dict(cursor=cursor,nll=row['nll'],preclip_norm=row['norm'],clip_multiplier=min(1.,1/(row['norm']+1e-6))))
  if cursor%4096==0:atomic(ckpt,snapshot());print(json.dumps(dict(arm=arm,cursor=cursor,end=end,last_nll=row['nll'],wall_s=past+time.monotonic()-began)),flush=True)
 if frozen is None:
  atomic(ckpt,snapshot());frozen=evaluate(model,val,[(q,1024) for q in range(25165824,25165824+8192,1024)]);atomic(stem.with_suffix('.weights.pt'),model.state_dict());atomic(ckpt,snapshot())
 # Primary online DEV: continue numerical TRAIN state; forecast before outcome-driven weight changes.
 while online<a.online_targets:
  row=update(transition_batch(val,30000000+online,32,state,tape_start=30000000));online_losses+=row['losses'];online+=32
  if online%1024==0:atomic(ckpt,snapshot())
 counts={k:state[k]-counters.get(k,0) for k in state if isinstance(state[k],int)}
 online_windows=[sum(online_losses[b:b+1024])/len(online_losses[b:b+1024]) for b in range(0,len(online_losses),1024)]
 result=dict(status='completed',battle='R1',stage='paired acquired native composition-credit continuation',arm=arm,settings=settings,parent_cursor=start,inherited_wall_s=inherited_wall,parent_sha256=digest,source_migration=migration,initial=initial,frozen_flagship=frozen,online=dict(nll=sum(online_losses)/len(online_losses),targets=len(online_losses),window_nll=online_windows,state='TRAIN state carries continuously. Last revealed TRAIN target predicts first new DEV target; subsequent DEV targets continue without memory reset.',weights='AdamW updates after each32 causally scored targets; both arms eligible'),additional_train_targets=a.extra_targets,discarded_recovery_targets=audit,exact_recovery=verified,parameters=sum(p.numel() for p in model.parameters()),capacity=dict(depth=8,receiver_slots=8*4*16,concept_units=4096),history=history,work_counts=counts,numeric_storage=model.core.storage(state),bank_storage=state['cause_bank'].storage(),recorded_process_s=past+time.monotonic()-began,process_scope='Recorded intervals;unsaved interruption tails are not included and no process-cost win is claimed.',max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,source_sha256=sources,scope='One shared acquired checkpoint/seed,paired same-token continuation;pure quality comparison with unequal learning work. FrozenDEV and chronological onlineDEV reported separately;no TEST,public benchmark,iso-FLOP or energy claim. Full arithmetic/special-function/traffic audit incomplete;all measured native work counters and process costs disclosed. Both arms retain legal joint query teacher;only new actual-child content credit differs,so this does not test an exact historical route teacher.')
 atomic(ckpt,snapshot());out.write_text(json.dumps(result,indent=2)+'\n');return result


def main():
 p=argparse.ArgumentParser();p.add_argument('--tag',required=True);p.add_argument('--recover-tag',required=True);p.add_argument('--extra-targets',type=int,default=65536);p.add_argument('--online-targets',type=int,default=8192);a=p.parse_args();assert a.extra_targets%32==a.online_targets%32==0 and a.online_targets>=1024 and a.extra_targets>=32
 torch.set_num_threads(1);anchor=ROOT/f'experiments/results/token_language/{a.tag}.parent.pt'
 if not anchor.exists():
  raw=(ROOT/f'experiments/results/token_language/{a.recover_tag}.parent.pt').read_bytes();tmp=anchor.with_suffix('.writing');tmp.write_bytes(raw);os.replace(tmp,anchor);del raw
 assert sha(anchor)==sha(ROOT/f'experiments/results/token_language/{a.recover_tag}.parent.pt')
 recovered=ROOT/f'experiments/results/token_language/{a.tag}_joint.resume.pt'
 if not recovered.exists():shutil.copyfile(ROOT/f'experiments/results/token_language/{a.recover_tag}_joint.resume.pt',recovered)
 digest=sha(anchor);train=load_fineweb(ROOT/'data/fineweb_gpt2/fineweb_train_000001.bin');val=load_fineweb(ROOT/'data/fineweb_gpt2/fineweb_val_000000.bin');assert len(val)>30000000+a.online_targets
 rows=[]
 for arm in ['joint','child']:rows.append(fit_arm(a,arm,anchor,digest,train,val));gc.collect()
 b,c=rows;assert b['settings']['parent_sha256']==c['settings']['parent_sha256'] and b['initial']['nll']==c['initial']['nll']
 r=dict(status='completed',battle='R1',stage='paired composition-credit empirical decision',parent_sha256=digest,extra_targets_per_arm=a.extra_targets,online_targets_per_arm=a.online_targets,arms=rows,online_nll_gain=b['online']['nll']-c['online']['nll'],frozen_nll_gain=b['frozen_flagship']['nll']-c['frozen_flagship']['nll'],novel_frozen_nll_gain=b['frozen_flagship']['novel_nll']-c['frozen_flagship']['novel_nll'],frozen_window_gains=[x['nll']-y['nll'] for x,y in zip(b['frozen_flagship']['windows'],c['frozen_flagship']['windows'])],scope='Single-parent/seed development comparison;positive gains favor bounded child-producer credit. Online and frozen verdicts separate;not a public or matched-compute claim.',source_sha256={k:v for row in rows for k,v in row['source_sha256'].items()})
 out=ROOT/f'experiments/results/token_language/{a.tag}.json'
 with out.open('x') as f:json.dump(r,f,indent=2);f.write('\n')
 print('RESULT '+json.dumps({k:r[k] for k in ['online_nll_gain','frozen_nll_gain','novel_frozen_nll_gain']}),flush=True)
if __name__=='__main__':main()
