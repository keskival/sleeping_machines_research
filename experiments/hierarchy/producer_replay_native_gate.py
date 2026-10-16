"""R1 bounded producer credit on an acquired checkpoint,not a new acquisition."""
import argparse,hashlib,io,json,resource,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT),str(ROOT/'experiments')]
import numpy as np
import torch,pytest
from r1_cerl import load_fineweb
from r1_local_native_deep import LocalNativeDeepLM
from experiments.native_query_migration import migrate
from sleeping_machines.producer_replay_native_race import enable_producer_credit
from tests.test_native_query_migration import identical


def main():
 p=argparse.ArgumentParser();p.add_argument('--tag',required=True);a=p.parse_args();torch.set_num_threads(1);began=time.monotonic()
 assert pytest.main(['-q','tests/test_producer_replay_native_race.py'])==0
 path=ROOT/'experiments/results/token_language/aws_r1_queryable_native_4m_s0_20261016_v1.resume.pt'
 raw=path.read_bytes();digest=hashlib.sha256(raw).hexdigest();saved=torch.load(io.BytesIO(raw),weights_only=False,map_location='cpu');del raw
 model=LocalNativeDeepLM(50257,saved['model']['causes.class_of'],saved['model']['causes.class_words'],width=64,depth=8,heads=4,pool=16,concept_units=4096,topk=4,explore=2,dk=32,packed=True)
 opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01);model,opt,state,_=migrate(model,opt,saved['state'])
 model.load_state_dict(saved['model']);opt.load_state_dict(saved['optimizer']);receipt=enable_producer_credit(model,state)
 g=torch.Generator();g.set_state(saved['g']);alt=torch.Generator();alt.set_state(saved['alt']);torch.set_rng_state(saved['rng']);cursor=saved['cursor'];del saved
 data=load_fineweb(ROOT/'data/fineweb_gpt2/fineweb_train_000001.bin');tokens=torch.from_numpy(np.asarray(data[cursor:cursor+65]).astype(np.int64))
 def update(b):
  opt.zero_grad(set_to_none=True);nll,obj,_,packets,report=model.stream_nll(tokens[b:b+32],tokens[b+1:b+33],state,g,alt,stats=True)
  obj.backward();model.deliver_credit(packets,state)
  assert all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in model.parameters())
  norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
  return dict(nll=float(nll.detach().mean()),preclip_norm=float(norm),replay_reads=state['producer_replay_reads'])
 first=update(0);snapshot=io.BytesIO();torch.save((model.state_dict(),opt.state_dict(),state,g.get_state(),alt.get_state(),torch.get_rng_state()),snapshot)
 second=update(32);expected=io.BytesIO();torch.save((model.state_dict(),opt.state_dict(),state,g.get_state(),alt.get_state(),torch.get_rng_state()),expected)
 snapshot.seek(0);weights,moments,state,gs,als,rng=torch.load(snapshot,weights_only=False);model.load_state_dict(weights);opt.load_state_dict(moments);g.set_state(gs);alt.set_state(als);torch.set_rng_state(rng)
 again=update(32);assert second==again
 expected.seek(0);assert identical(torch.load(expected,weights_only=False),(model.state_dict(),opt.state_dict(),state,g.get_state(),alt.get_state(),torch.get_rng_state()))
 assert state['producer_replay_reads']>0
 result=dict(status='completed',battle='R1',stage='producer_replay_native_gate',gate='PASS',checkpoint_sha256=digest,checkpoint_cursor=cursor,migration=receipt,first=first,second=second,exact_serialized_resume=True,storage=model.core.storage(state),wall_s=time.monotonic()-began,max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
  scope='Two32targetacquiredTRAINupdates plus discarded32targetresumeaudit;running checkpoint untouched. No DEV/TEST,ablationorquality claim. Forward unchanged by local producer tangent;fixed-input/route/time local derivative exact under fixed parameters,changedcomposer/mix historical maps approximate. Own-write credit separate,optimizer unchanged.')
 out=ROOT/f'experiments/results/diagnostics/{a.tag}.json';assert not out.exists();out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:result[k] for k in ['gate','checkpoint_cursor','wall_s','max_rss_kb']}),flush=True)
if __name__=='__main__':main()
