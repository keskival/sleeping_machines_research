"""R1 acquired native boundary-law/optimizer/recovery gate;no DEV/TEST."""
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
from tests.test_native_query_migration import identical


def main():
 p=argparse.ArgumentParser();p.add_argument('--tag',required=True);a=p.parse_args();torch.set_num_threads(1);began=time.monotonic()
 assert pytest.main(['-q','tests/test_document_boundary_causes.py'])==0
 path=ROOT/'experiments/results/token_language/aws_r1_native_producer_credit_4m_s0_20261016_v2.resume.pt'
 raw=path.read_bytes();digest=hashlib.sha256(raw).hexdigest();saved=torch.load(io.BytesIO(raw),weights_only=False,map_location='cpu');del raw
 model=LocalNativeDeepLM(50257,saved['model']['causes.class_of'],saved['model']['causes.class_words'],width=64,depth=8,heads=4,pool=16,concept_units=4096,topk=4,explore=2,dk=32,packed=True)
 opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01);model,opt,state,_=migrate(model,opt,saved['state'])
 enable_producer_credit(model,state);model.load_state_dict(saved['model']);opt.load_state_dict(saved['optimizer'])
 before=model.state_dict();receipt=enable_document_relevance(model,opt)
 assert all(torch.equal(v,model.state_dict()[k]) for k,v in before.items())
 g=torch.Generator();g.set_state(saved['g']);alt=torch.Generator();alt.set_state(saved['alt']);torch.set_rng_state(saved['rng']);cursor=saved['cursor'];del saved
 data=load_fineweb(ROOT/'data/fineweb_gpt2/fineweb_train_000001.bin');tokens=torch.from_numpy(np.asarray(data[cursor:cursor+65]).astype(np.int64));gradients=[]
 def update(b):
  opt.zero_grad(set_to_none=True);nll,obj,_,packets,_=model.stream_nll(tokens[b:b+32],tokens[b+1:b+33],state,g,alt,stats=True)
  obj.backward();model.deliver_credit(packets,state);gradients.append(model.causes.document_bias.grad.detach().tolist())
  assert all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in model.parameters())
  norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step();return dict(nll=float(nll.detach().mean()),norm=float(norm),document_bias=model.causes.document_bias.detach().tolist())
 first=update(0);snapshot=io.BytesIO();torch.save((model.state_dict(),opt.state_dict(),state,g.get_state(),alt.get_state(),torch.get_rng_state()),snapshot)
 second=update(32);expected=io.BytesIO();torch.save((model.state_dict(),opt.state_dict(),state,g.get_state(),alt.get_state(),torch.get_rng_state()),expected)
 snapshot.seek(0);weights,moments,state,gs,als,rng=torch.load(snapshot,weights_only=False);model.load_state_dict(weights);opt.load_state_dict(moments);g.set_state(gs);alt.set_state(als);torch.set_rng_state(rng)
 again=update(32);assert second==again;expected.seek(0)
 assert identical(torch.load(expected,weights_only=False),(model.state_dict(),opt.state_dict(),state,g.get_state(),alt.get_state(),torch.get_rng_state()))
 result=dict(status='completed',battle='R1',stage='document_boundary_native_gate',gate='PASS',checkpoint_sha256=digest,checkpoint_cursor=cursor,migration=receipt,first=first,second=second,document_bias_gradients=gradients,exact_serialized_resume=True,wall_s=time.monotonic()-began,max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
  scope='Zero-biasforward/existing-gradientparity,normalizationandfinite-differencecontracts;real64TRAINtargets plus32discardedrecoveryaudit on a privatecheckpoint. No DEV/TEST,ablationorqualityclaim;running source unchanged.')
 out=ROOT/f'experiments/results/diagnostics/{a.tag}.json';assert not out.exists();out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:result[k] for k in ['gate','checkpoint_cursor','wall_s','max_rss_kb']}),flush=True)
if __name__=='__main__':main()
