"""R1 numerical repair gate; run only through guarded queue."""
import argparse,json,sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'experiments'))
import unrestricted_receiver_token_language as new
import local_clip_receiver_token_language as old

def main():
 p=argparse.ArgumentParser();p.add_argument('--tag',required=True);tag=p.parse_args().tag;torch.set_num_threads(1)
 a=SimpleNamespace(**json.loads((ROOT/'experiments/results/token_language/aws_native_deep_d8_8m_20261013_s6_v1.json').read_text())['args']);a.payload=8;a.credit_window=32;a.eos_reset=False
 data=np.arange(200,dtype=np.uint16);lanes=new.ContinuousLaneTokens(data,32,2,'fresh',3,16)
 assert torch.equal(lanes[:,0:49],torch.tensor(np.array(data[:98],dtype=np.int64)).reshape(2,49))
 assert lanes.blocks==1 and torch.equal(lanes.counts(),new.LaneTokens(data,0,32,2).counts())
 counts=torch.ones(50257,dtype=torch.long);order=torch.arange(50257);torch.manual_seed(6);model=new.Model(a,order,counts)
 ids=torch.arange(32).reshape(1,32);ids[0,8]=50256
 model.eval();g=torch.Generator().manual_seed(1)
 x,state,_=model(ids,None,g,a,deterministic=True)
 g=torch.Generator().manual_seed(1);left,st,_=model(ids[:,:16],None,g,a,deterministic=True);right,st,_=model(ids[:,16:],st,g,a,deterministic=True)
 err=float((x-torch.cat([left,right],1)).abs().max());assert err==0
 model.train();model.zero_grad();x,_,_=model(ids,None,torch.Generator().manual_seed(1),a,deterministic=True);model.readout.nll(x,torch.ones_like(ids)).mean().backward()
 assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
 a.eos_reset=True;a.credit_window=16;baseline=old.Model(a,order,counts);baseline.load_state_dict(model.state_dict());model.eval();baseline.eval()
 x,_,_=model(ids,None,torch.Generator().manual_seed(1),a,deterministic=True);y,_,_=baseline(ids,None,torch.Generator().manual_seed(1),a,deterministic=True);compat=float((x-y).abs().max());assert compat==0
 sources={name:new.sha(ROOT/name) for name in ['experiments/check_unrestricted_language.py','experiments/unrestricted_receiver_token_language.py','experiments/local_clip_receiver_token_language.py']}
 out=ROOT/f'experiments/results/credit/{tag}.json';out.parent.mkdir(exist_ok=True)
 with out.open('x') as f:json.dump(dict(status='completed',battle='R1',gate='PASS',continuous_chunk_error=err,legacy_policy_error=compat,finite_long_window_gradients=True,source_sha256=sources,scope='D8 synthetic32-token/EOS witness; contiguous lane indexing and prior contract. No fitted quality or cross-update credit claim.'),f,indent=2)
if __name__=='__main__':main()
