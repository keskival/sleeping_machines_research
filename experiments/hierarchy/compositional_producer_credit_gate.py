"""R1 conditional composition-credit contracts plus acquired native optimizer/recovery gate."""
import argparse,hashlib,io,json,resource,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT),str(ROOT/'experiments')]
import numpy as np
import torch,pytest
from r1_cerl import load_fineweb
from r1_local_native_deep import LocalNativeDeepLM
from experiments.native_query_migration import migrate
from sleeping_machines.producer_replay_native_race import enable_producer_credit
from sleeping_machines.document_boundary_causes import enable_document_relevance
from sleeping_machines.coupled_native_query_credit import enable_coupled_query_credit
from sleeping_machines.compositional_producer_credit import enable_compositional_credit
from experiments.hierarchy.native_composition_data import source_hashes
from tests.test_native_query_migration import identical


def main():
 p=argparse.ArgumentParser();p.add_argument('--tag',required=True);a=p.parse_args();torch.set_num_threads(1);began=time.monotonic()
 assert pytest.main(['-q','tests/test_compositional_producer_credit.py','tests/test_coupled_native_query_credit.py'])==0
 path=ROOT/'experiments/results/token_language/aws_r1_native_document_credit_4m_s0_20261016_v1.resume.pt'
 raw=path.read_bytes();digest=hashlib.sha256(raw).hexdigest();saved=torch.load(io.BytesIO(raw),weights_only=False,map_location='cpu');del raw
 model=LocalNativeDeepLM(50257,saved['model']['causes.class_of'],saved['model']['causes.class_words'],width=64,depth=8,heads=4,pool=16,concept_units=4096,topk=4,explore=2,dk=32,packed=True)
 opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01);model,opt,state,_=migrate(model,opt,saved['state'])
 enable_producer_credit(model,state);enable_document_relevance(model,opt);model.load_state_dict(saved['model']);opt.load_state_dict(saved['optimizer']);state=saved['state']
 enable_coupled_query_credit(model,state);receipt=enable_compositional_credit(model,state)
 g=torch.Generator();g.set_state(saved['g']);alt=torch.Generator();alt.set_state(saved['alt']);torch.set_rng_state(saved['rng']);cursor=saved['cursor'];del saved
 data=load_fineweb(ROOT/'data/fineweb_gpt2/fineweb_train_000001.bin');tokens=torch.from_numpy(np.asarray(data[cursor:cursor+17]).astype(np.int64))
 def update(b):
  opt.zero_grad(set_to_none=True);nll,obj,_,packets,_=model.stream_nll(tokens[b:b+8],tokens[b+1:b+9],state,g,alt,stats=True)
  obj.backward();model.deliver_credit(packets,state)
  assert all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in model.parameters()) and bool(torch.isfinite(obj))
  norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
  return dict(nll=float(nll.detach().mean()),gradient_norm=float(norm),recursive_edges=state['recursive_replay_edges'],recursive_roots=state['recursive_replay_roots'])
 first=update(0);assert state['recursive_replay_edges']>0
 snap=io.BytesIO();torch.save((model.state_dict(),opt.state_dict(),state,g.get_state(),alt.get_state(),torch.get_rng_state()),snap)
 second=update(8);expected=io.BytesIO();torch.save((model.state_dict(),opt.state_dict(),state,g.get_state(),alt.get_state(),torch.get_rng_state()),expected)
 snap.seek(0);weights,moments,state,gs,als,rng=torch.load(snap,weights_only=False);model.load_state_dict(weights);opt.load_state_dict(moments);g.set_state(gs);alt.set_state(als);torch.set_rng_state(rng)
 again=update(8);expected.seek(0);assert second==again and identical(torch.load(expected,weights_only=False),(model.state_dict(),opt.state_dict(),state,g.get_state(),alt.get_state(),torch.get_rng_state()))
 sources=source_hashes()
 for f in ['experiments/hierarchy/compositional_producer_credit_gate.py','tests/test_compositional_producer_credit.py','tests/test_coupled_native_query_credit.py','tests/test_native_query_migration.py','experiments/native_query_migration.py','experiments/r1_local_native_deep.py','experiments/r1_cerl.py']:sources[f]=hashlib.sha256((ROOT/f).read_bytes()).hexdigest()
 result=dict(status='completed',battle='R1',stage='compositional producer credit gate',gate='PASS',checkpoint_cursor=cursor,checkpoint_sha256=digest,migration=receipt,first=first,second=second,exact_serialized_resume=True,unique_train_targets=16,discarded_resume_audit_targets=8,wall_s=time.monotonic()-began,max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,storage=model.core.storage(state),source_sha256=sources,scope='Private acquired native8layer numerical/optimizer/recovery gate;no DEV/TEST or quality claim. Historical routes/times fixed;bounded actual-child content VJP,changed-parameter replay approximate. Source checkpoint and active trainer untouched.')
 with (ROOT/f'experiments/results/diagnostics/{a.tag}.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
 print(json.dumps({k:result[k] for k in ['gate','checkpoint_cursor','wall_s','max_rss_kb']}),flush=True)
if __name__=='__main__':main()
