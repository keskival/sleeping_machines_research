"""R1 owned native continuation from acquired weights,Adam and causal history."""
import argparse,copy,hashlib,json,math,os,resource,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'experiments')]
import numpy as np
import torch
from r1_cerl import load_fineweb
from r1_local_native_deep import LocalNativeDeepLM,word_classes
from r1_native_local_stream_fit import score,atomic
from experiments.native_query_migration import migrate
from sleeping_machines.producer_replay_native_race import enable_producer_credit
from sleeping_machines.document_boundary_causes import enable_document_relevance
from sleeping_machines.coupled_native_query_credit import enable_coupled_query_credit
from tests.test_native_query_migration import identical
from experiments.hierarchy.native_composition_data import source_hashes


def main():
 p=argparse.ArgumentParser();p.add_argument('--tag',required=True);p.add_argument('--parent',required=True);p.add_argument('--resume-from',required=True);p.add_argument('--targets',type=int,default=4194304);a=p.parse_args()
 torch.set_num_threads(1);torch.manual_seed(0);began=time.monotonic()
 parent_path=ROOT/f'experiments/results/token_language/{a.parent}.resume.pt'
 parent_result=ROOT/f'experiments/results/token_language/{a.parent}.json'
 parent_report=json.loads(parent_result.read_text());assert parent_report['status']=='completed'
 train=load_fineweb(ROOT/'data/fineweb_gpt2/fineweb_train_000001.bin');val=load_fineweb(ROOT/'data/fineweb_gpt2/fineweb_val_000000.bin')
 counts=np.bincount(np.asarray(train[:4194304]).astype(np.int64),minlength=50257);classes,words=word_classes(counts,256)
 model=LocalNativeDeepLM(50257,classes,words,width=64,depth=8,heads=4,pool=16,concept_units=4096,topk=4,explore=2,dk=32,packed=True)
 opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01)
 stem=ROOT/f'experiments/results/token_language/{a.tag}';out=stem.with_suffix('.json');checkpoint=stem.with_suffix('.resume.pt');assert not out.exists()
 parent=torch.load(parent_path,map_location='cpu',weights_only=False)
 assert parent['cursor']==65536 and parent['resume_verified'] and a.targets>parent['cursor'] and a.targets%32==0
 model.load_state_dict(parent['model']);opt.load_state_dict(parent['optimizer'])
 with torch.random.fork_rng():
  torch.manual_seed(16016);model,opt,state,migration=migrate(model,opt,parent['state'],publication_capacity=16)
 credit_upgrade=enable_producer_credit(model,state)
 joint_upgrade=enable_coupled_query_credit(model,state)
 g=torch.Generator();alt=torch.Generator();g.set_state(parent['g']);alt.set_state(parent['alt']);torch.set_rng_state(parent['rng'])
 cursor=parent['cursor'];parent_cursor=cursor;del parent
 sources=source_hashes()
 for f in ['experiments/r1_joint_native_stream_fit.py','experiments/r1_document_native_stream_fit.py','experiments/r1_producer_credit_stream_fit.py','experiments/r1_queryable_native_stream_fit.py','experiments/native_query_migration.py','experiments/r1_native_local_stream_fit.py','experiments/r1_local_native_deep.py','experiments/r1_cerl.py','tests/test_native_query_migration.py']:
  sources[f]=hashlib.sha256((ROOT/f).read_bytes()).hexdigest()
 settings=dict(targets=a.targets,batch=32,parent=a.parent,parent_cursor=parent_cursor,query_root_capacity=16,query_reads=2,query_init_seed=16016,lr=.001,weight_decay=.01,global_clip=1.,producer_credit_version=1,document_boundary_version=1,joint_query_credit_version=1)
 document_upgrade=None
 history=[];milestones=[];initial=None;past=0.;resume_verified=False;audit_work=0
 def snapshot():return dict(model=model.state_dict(),optimizer=opt.state_dict(),state=state,g=g.get_state(),alt=alt.get_state(),rng=torch.get_rng_state(),cursor=cursor,history=history,milestones=milestones,initial=initial,wall_s=past+time.monotonic()-began,resume_verified=resume_verified,audit_work=audit_work,settings=settings,source_sha256=sources,migration=migration,credit_upgrade=credit_upgrade,document_upgrade=document_upgrade,joint_upgrade=joint_upgrade)
 def restore(saved):
  nonlocal state,cursor,history,milestones,initial,past,resume_verified,audit_work
  assert saved['settings']==settings and saved['source_sha256']==sources
  model.load_state_dict(saved['model']);opt.load_state_dict(saved['optimizer']);state=saved['state'];g.set_state(saved['g']);alt.set_state(saved['alt']);torch.set_rng_state(saved['rng'])
  cursor=saved['cursor'];history=saved['history'];milestones=saved['milestones'];initial=saved['initial'];past=saved['wall_s'];resume_verified=saved['resume_verified'];audit_work=saved['audit_work']
 if checkpoint.exists():
  document_upgrade=enable_document_relevance(model,opt)
  restore(torch.load(checkpoint,map_location='cpu',weights_only=False));began=time.monotonic()
 else:
  document_upgrade=enable_document_relevance(model,opt)
  prior_path=ROOT/f'experiments/results/token_language/{a.resume_from}.resume.pt'
  prior=torch.load(prior_path,map_location='cpu',weights_only=False)
  assert prior['settings']=={k:v for k,v in settings.items() if k!='joint_query_credit_version'}
  for name,digest in prior['source_sha256'].items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,name
  model.load_state_dict(prior['model']);opt.load_state_dict(prior['optimizer']);state=prior['state']
  state.setdefault('producer_records',{});state.setdefault('producer_replay_reads',0)
  g.set_state(prior['g']);alt.set_state(prior['alt']);torch.set_rng_state(prior['rng'])
  state.setdefault('joint_pair_score_entries',0);state.setdefault('coupled_query_audits',0)
  cursor=prior['cursor'];history=prior['history'];milestones=prior['milestones'];initial=prior['initial'];past=prior['wall_s'];audit_work=prior['audit_work']
  resume_verified=False;began=time.monotonic();del prior
  print(json.dumps(dict(credit_upgrade=credit_upgrade,document_upgrade=document_upgrade,upgraded_from=a.resume_from,cursor=cursor)),flush=True)
 def update(start):
  model.train();opt.zero_grad(set_to_none=True);tokens=torch.from_numpy(np.asarray(train[start:start+33]).astype(np.int64))
  nll,obj,_,packets,report=model.stream_nll(tokens[:-1],tokens[1:],state,g,alt,stats=True)
  obj.backward();model.deliver_credit(packets,state)
  assert math.isfinite(float(obj.detach())) and all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in model.parameters())
  torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
  return dict(mean_nll=float(nll.detach().mean()),native_cause_gain=report['native_cause_gain'],depth=report['depth_responsibility'])
 if not checkpoint.exists():atomic(checkpoint,snapshot())
 if initial is None:
  initial=score(model,val,[(10485760,1024)]);atomic(checkpoint,snapshot());print(json.dumps(dict(migration=migration,initial_selection=initial)),flush=True)
 if not resume_verified:
  # Exact acquired-model serialized continuation before committing the long fit.
  saved=torch.load(checkpoint,map_location='cpu',weights_only=False);record=update(cursor)
  expected=(copy.deepcopy(model.state_dict()),copy.deepcopy(opt.state_dict()),copy.deepcopy(state),g.get_state(),alt.get_state(),torch.get_rng_state())
  charged=past+time.monotonic()-began;restore(saved);past=charged;began=time.monotonic();again=update(cursor)
  assert record==again and identical(expected,(model.state_dict(),opt.state_dict(),state,g.get_state(),alt.get_state(),torch.get_rng_state()))
  del expected,saved
  history.append(again);cursor+=32;resume_verified=True;audit_work+=32;atomic(checkpoint,snapshot());print(json.dumps(dict(exact_serialized_resume=True,cursor=cursor,audit_targets=audit_work)),flush=True)
 while cursor<a.targets:
  history.append(update(cursor));cursor+=32
  if cursor%1024==0:
   print(json.dumps(dict(targets=cursor,of=a.targets,last1024_nll=sum(r['mean_nll'] for r in history[-32:])/len(history[-32:]),elapsed_s=past+time.monotonic()-began,query_records=len(state['bindings'].entries),producer_replay_reads=state['producer_replay_reads'],document_bias=model.causes.document_bias.detach().tolist(),coupled_query_audits=state['coupled_query_audits'])),flush=True)
  if cursor in [262144,1048576,a.targets]:
   result=score(model,val,[(10485760,1024)]);milestones.append(dict(targets=cursor,selection=result));print('MILESTONE '+json.dumps(milestones[-1]),flush=True)
  if cursor%4096==0 or cursor==a.targets:atomic(checkpoint,snapshot())
 flagship=score(model,val,[(q,1024) for q in range(25165824,25165824+8192,1024)])
 atomic(stem.with_suffix('.pt'),model.state_dict())
 names=['events','candidate_scores','selected_updates','alternative_values','credit_projections','memory_key_scores','memory_selected_reads','memory_alternative_reads','memory_transport_heads','memory_pair_products','queried_historical_versions','memory_gc_removed','producer_replay_reads','joint_pair_score_entries','coupled_query_audits']
 result=dict(status='completed',battle='R1',stage='joint-query native document/producer continuation',joint_upgrade=joint_upgrade,document_upgrade=document_upgrade,credit_upgrade=credit_upgrade,upgraded_from=a.resume_from,settings=settings,migration=migration,parameters=sum(p.numel() for p in model.parameters()),initial_selection=initial,milestones=milestones,flagship=flagship,history=history,exact_serialized_resume=resume_verified,resume_audit_targets=audit_work,training_counts={k:state[k] for k in names},bank=state['cause_bank'].storage(),numeric_state=model.core.storage(state),total_wall_s=past+time.monotonic()-began,max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,source_sha256=sources,parent_result_sha256=hashlib.sha256(parent_result.read_bytes()).hexdigest(),
  scope='One chronological native continuation,shared64Kacquisition/Adam/state retained;4Mtotaltargets if admitteddefault,discardedresumeaudit targets recorded/charged across source and upgradedcontinuation. Nonzero joint maps change architecture;unequaldata/work vs64Kparent,no architecture-isolating claim. Frozen DEV1024selectionmilestones and final8192/1024flagship,no publicTEST.Legalorderedpairquerycounterfactuals/correctconditionalclock,localjointsurrogate.Causal learned crossdocumentrepeat/continuation log-rate biases,no hardreset.No recurrentBPTT;boundedlocalproducerreplay/own-write credit conditional,changedparametersapproximate,optimizer synchronous;allquery keys/losers/traceprojectionscharged.Full arithmetic/traffic audit pending,no matched-compute or hardwareclaim.')
 out.write_text(json.dumps(result,indent=2)+'\n');print('RESULT '+json.dumps(dict(flagship=flagship)),flush=True)
if __name__=='__main__':main()
