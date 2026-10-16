"""Owned R1 development: deep native causes, persistent state, local credit.

No dense control fit, ablation, public validation or TEST. Fixed final model.
"""
import argparse,copy,hashlib,json,math,os,resource,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'experiments')]
import numpy as np
import torch
from r1_cerl import load_fineweb
from r1_local_native_deep import LocalNativeDeepLM,word_classes
from experiments.hierarchy.native_composition_data import source_hashes


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def atomic(path,value):
 tmp=path.with_suffix('.writing');torch.save(value,tmp);os.replace(tmp,path)
def equal(a,b):
 if isinstance(a,torch.Tensor):return torch.equal(a,b)
 if isinstance(a,dict):return a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
 if isinstance(a,(list,tuple)):return len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
 return a==b


def score(model,val,windows):
 old=model.training;model.eval();rng=torch.get_rng_state();total=0.;count=0;gains=[];depth=[];causes=[];bank_work={}
 try:
  with torch.no_grad():
   for start,targets in windows:
    tokens=torch.from_numpy(np.asarray(val[start:start+targets+1]).astype(np.int64))
    state=model.new_state();g=torch.Generator().manual_seed(113);alt=torch.Generator().manual_seed(114)
    for b in range(0,targets,32):
     e=min(b+32,targets);nll,_,state,_,report=model.stream_nll(tokens[b:e],tokens[b+1:e+1],state,g,alt,learn=False,stats=True)
     total+=float(nll.double().sum());count+=len(nll);gains.append((report['native_cause_gain'],len(nll)))
     depth.append((torch.tensor(report['depth_responsibility']),len(nll)));causes.append((torch.tensor(report['cause_responsibility']),len(nll)))
    for k in ['keys_examined','index_entries_examined','delivered_candidates','writes']:
     bank_work[k]=bank_work.get(k,0)+state['cause_bank'].storage()[k]
 finally:torch.set_rng_state(rng);model.train(old)
 return dict(nll=total/count,targets=count,native_cause_gain=sum(x*n for x,n in gains)/count,
  depth_responsibility=(sum(x*n for x,n in depth)/count).tolist(),cause_responsibility=(sum(x*n for x,n in causes)/count).tolist(),index_work=bank_work,
  protocol=dict(windows=windows,race_seed=113,weights='frozen',state='persistent within each prescribed window;independent windows reset;cause messages mature causally'))


def main():
 p=argparse.ArgumentParser();p.add_argument('--tag',required=True);p.add_argument('--targets',type=int,default=65536);p.add_argument('--batch',type=int,default=32);p.add_argument('--seed',type=int,default=0);a=p.parse_args()
 assert a.targets%a.batch==0 and a.targets>=64
 torch.set_num_threads(1);torch.manual_seed(a.seed);began=time.monotonic()
 train=load_fineweb(ROOT/'data/fineweb_gpt2/fineweb_train_000001.bin');val=load_fineweb(ROOT/'data/fineweb_gpt2/fineweb_val_000000.bin')
 prefix=np.asarray(train[:4194304]).astype(np.int64);counts=np.bincount(prefix,minlength=50257);classes,words=word_classes(counts,256)
 model=LocalNativeDeepLM(50257,classes,words,width=64,depth=8,heads=4,pool=16,concept_units=4096,topk=4,explore=2,dk=32,packed=True)
 opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01);state=model.new_state();g=torch.Generator().manual_seed(a.seed+13);alt=torch.Generator().manual_seed(a.seed+14)
 stem=ROOT/f'experiments/results/token_language/{a.tag}';out=stem.with_suffix('.json');checkpoint=stem.with_suffix('.resume.pt');assert not out.exists()
 sources=source_hashes()
 for f in ['experiments/r1_local_native_deep.py','experiments/r1_cerl.py','experiments/r1_native_local_stream_fit.py']:sources[f]=sha(ROOT/f)
 settings=dict(targets=a.targets,batch=a.batch,seed=a.seed,width=64,depth=8,heads=4,pool=16,concepts=4096,bank_capacity=32768,bank_candidates=128,lr=.001,weight_decay=.01,global_clip=1.)
 cursor=0;history=[];initial=None;past=0.;resume_verified=False;audit_work=0
 def snapshot():return dict(model=model.state_dict(),optimizer=opt.state_dict(),state=state,g=g.get_state(),alt=alt.get_state(),rng=torch.get_rng_state(),cursor=cursor,history=history,initial=initial,wall_s=past+time.monotonic()-began,resume_verified=resume_verified,audit_work=audit_work,settings=settings,source_sha256=sources)
 def restore(saved):
  nonlocal state,cursor,history,initial,past,resume_verified,audit_work
  assert saved['settings']==settings and saved['source_sha256']==sources
  model.load_state_dict(saved['model']);opt.load_state_dict(saved['optimizer']);state=saved['state'];g.set_state(saved['g']);alt.set_state(saved['alt']);torch.set_rng_state(saved['rng'])
  cursor=saved['cursor'];history=saved['history'];initial=saved['initial'];past=saved['wall_s'];resume_verified=saved['resume_verified'];audit_work=saved['audit_work']
 if checkpoint.exists():restore(torch.load(checkpoint,map_location='cpu',weights_only=False));began=time.monotonic()
 def update(start):
  model.train();opt.zero_grad(set_to_none=True);tokens=torch.from_numpy(np.asarray(train[start:start+a.batch+1]).astype(np.int64))
  nll,obj,_,packets,report=model.stream_nll(tokens[:-1],tokens[1:],state,g,alt,stats=True)
  obj.backward();model.deliver_credit(packets,state)
  assert math.isfinite(float(obj.detach())) and all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in model.parameters())
  torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
  return dict(mean_nll=float(nll.detach().mean()),native_cause_gain=report['native_cause_gain'],depth_responsibility=report['depth_responsibility'],cause_responsibility=report['cause_responsibility'],bank_occupied=report['bank']['occupied'])
 if initial is None:initial=score(model,val,[(10485760,1024)]);atomic(checkpoint,snapshot());print(json.dumps(dict(initial_selection=initial)),flush=True)
 if not resume_verified:
  # One real first update,then serialize. The second update is checked from
  # that exact serialized state;its discarded duplicate is explicitly charged.
  if cursor==0:history.append(update(0));cursor=a.batch;atomic(checkpoint,snapshot())
  saved=torch.load(checkpoint,map_location='cpu',weights_only=False);record=update(cursor)
  expected_model=copy.deepcopy(model.state_dict());expected_opt=copy.deepcopy(opt.state_dict());expected_g=g.get_state();expected_alt=alt.get_state();expected_rng=torch.get_rng_state();expected_state=copy.deepcopy(state)
  charged_wall=past+time.monotonic()-began
  restore(saved);past=charged_wall;began=time.monotonic();again=update(cursor)
  assert equal(model.state_dict(),expected_model) and equal(opt.state_dict(),expected_opt) and record==again
  assert torch.equal(g.get_state(),expected_g) and torch.equal(alt.get_state(),expected_alt) and torch.equal(torch.get_rng_state(),expected_rng)
  assert equal(state['layers'],expected_state['layers']) and torch.equal(state['cause_bank'].features,expected_state['cause_bank'].features)
  assert state['cause_bank'].index==expected_state['cause_bank'].index and torch.equal(state['cause_inbox'],expected_state['cause_inbox'])
  history.append(again);cursor+=a.batch;resume_verified=True;audit_work+=a.batch;del expected_model,expected_opt,expected_state,saved
  atomic(checkpoint,snapshot());print(json.dumps(dict(exact_serialized_resume=True,cursor=cursor,audit_targets=audit_work)),flush=True)
 while cursor<a.targets:
  history.append(update(cursor));cursor+=a.batch
  if cursor%1024==0 or cursor==a.targets:
   atomic(checkpoint,snapshot());print(json.dumps(dict(targets=cursor,of=a.targets,last1024_nll=sum(r['mean_nll'] for r in history[-1024//a.batch:])/(1024//a.batch),native_cause_gain=history[-1]['native_cause_gain'],depth=history[-1]['depth_responsibility'],bank_occupied=state['cause_bank'].storage()['occupied'],elapsed_s=past+time.monotonic()-began)),flush=True)
 fit_wall=past+time.monotonic()-began
 final_selection=score(model,val,[(10485760,1024)])
 flagship=score(model,val,[(q,1024) for q in range(25165824,25165824+8192,1024)])
 atomic(stem.with_suffix('.pt'),model.state_dict())
 result=dict(status='completed',battle='R1',stage='native deep persistent local-learning development',settings=settings,parameters=sum(p.numel() for p in model.parameters()),initial_selection=initial,final_selection=final_selection,flagship=flagship,history=history,
  exact_serialized_resume=resume_verified,resume_audit_targets=audit_work,training_counts={k:state[k] for k in ['events','candidate_scores','selected_updates','alternative_values','credit_projections']},bank=state['cause_bank'].storage(),numeric_state=model.core.storage(state),fit_and_initial_eval_wall_s=fit_wall,total_wall_s=past+time.monotonic()-began,max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,source_sha256=sources,
  data_sha256=dict(class_partition_prefix=hashlib.sha256(prefix.tobytes()).hexdigest(),observed_train_tokens=hashlib.sha256(np.asarray(train[:a.targets+1]).tobytes()).hexdigest()),
  scope='One fixed native8layer chronological development acquisition,no recurrentBPTT,densecarrier,ablation or publicTEST.Full receiver/indexedepisodic/inbox state persists across32targetoptimizerboundaries;own-writeeligibilityconditional and changing-parameterhistoryapproximate. GlobalAdamW/clipping still synchronous. TRAINpre-update loss uses a class map fitted on4MTRAINtokens and is not untouched online evidence. Fixedfinalselection1024targets and flagship8192targets/1024windows;no comparable-data/compute victory inferred against larger saved hybrids. Full arithmetic/special-function/traffic accounting pending.',decision='Does the strongest integrated native deep local learner improve held-out prediction after65536targets while retaining long numerical history? Select a larger competitive native acquisition or a diagnosed repair;no automatic long extension.')
 out.write_text(json.dumps(result,indent=2)+'\n');print('RESULT '+json.dumps(dict(selection=final_selection,flagship=flagship)),flush=True)
if __name__=='__main__':main()
