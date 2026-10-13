"""R1 all-depth readout contract: unchanged native state/races and normalized prior."""
import argparse,hashlib,json,sys
from pathlib import Path
from types import SimpleNamespace
import torch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'experiments'))
import bernoulli_teacher_token_language as original
import multidepth_readout_token_language as candidate

def maxerror(x,y):
 if isinstance(x,dict):return max(maxerror(x[k],y[k]) for k in x)
 if isinstance(x,list):return max(maxerror(a,b) for a,b in zip(x,y))
 if x.dtype==torch.bool:return float(not torch.equal(x,y))
 return float((x-y).abs().max())

def main():
 p=argparse.ArgumentParser();p.add_argument('--tag',required=True);a=p.parse_args();torch.set_num_threads(1);rows=[]
 path=ROOT/'experiments/results/token_language/aws_bernoulli_teacher_512k_20261013_s6_v1.json';base=json.loads(path.read_text());args=SimpleNamespace(**base['args'])
 train=original.load_tokens(args.train_file);counts=original.LaneTokens(train,0,args.train_tokens,args.lanes).counts();order=torch.argsort(counts,descending=True,stable=True)
 for dtype in (torch.float32,torch.float64):
  torch.set_default_dtype(dtype);torch.manual_seed(6);old=original.Model(args,order,counts);torch.manual_seed(6);new=candidate.Model(args,order,counts)
  new.core.load_state_dict(old.core.state_dict());ids=torch.tensor([[10,11,12,50256,13,14,15,16],[9,8,9,8,7,6,5,4]],dtype=torch.long)
  left=torch.Generator().manual_seed(7);right=torch.Generator().manual_seed(7)
  old.eval();new.eval();x,st,pis=old(ids,None,left,args);y,nt,qis=new(ids,None,right,args)
  state_error=maxerror(st,nt);feature_error=float((x-y[:,:,-x.shape[-1]:]).abs().max());probability_error=max(float((l[1]-r[1]).abs().max()) for l,r in zip(pis,qis))
  assert state_error==feature_error==probability_error==0 and torch.equal(left.get_state(),right.get_state())
  assert y.shape[-1]==args.depth*x.shape[-1]
  target=torch.tensor([[1,2,3,4,5,6,7,8],[8,7,6,5,4,3,2,1]])
  with torch.no_grad():prior_error=float((old.readout.nll(x,target)-new.readout.nll(y,target)).abs().max())
  assert prior_error<2e-6
  old.zero_grad();new.zero_grad();x.square().sum().backward();y[:,:,-x.shape[-1]:].square().sum().backward()
  errors=[]
  for (ln,lp),(rn,rp) in zip(old.core.named_parameters(),new.core.named_parameters()):
   assert ln==rn
   if lp.grad is None or rp.grad is None:assert lp.grad is None and rp.grad is None
   else:errors.append(float((lp.grad-rp.grad).abs().max()))
  gradient_error=max(errors);assert gradient_error<1e-6
  rows.append(dict(dtype=str(dtype),state_error=state_error,final_leg_feature_error=feature_error,probability_error=probability_error,prior_error=prior_error,final_leg_gradient_error=gradient_error,selected_writes=int(st['last_writes'].sum()),readout_width=y.shape[-1]))
 sources=dict(base['source_sha256'])
 for name in ('experiments/multidepth_readout_token_language.py','sleeping_machines/multidepth_counterfactual_episodes.py','sleeping_machines/multidepth_ordered_episodes.py',str(Path(__file__).relative_to(ROOT))):sources[name]=original.sha(ROOT/name)
 for name,digest in sources.items():assert original.sha(ROOT/name)==digest,name
 d=dict(status='completed',battle='R1',gate='PASS',rows=rows,source_sha256=sources,input_sha256={str(path.relative_to(ROOT)):original.sha(path)},scope='Float32/64 native two-layer8-event contract inclEOS. Same core parameters/RNG: exact state, final-layer features, route probabilities and selected writes; final-leg gradient tolerance1e-6. Initial normalized TRAIN-prior loss unchanged; earlier-layer channels transported to final arrival. Active output payload/readout work increases. No fitted quality or resource advantage claimed.')
 with (ROOT/f'experiments/results/diagnostics/{a.tag}.json').open('x') as f:json.dump(d,f,indent=2);f.write('\n')
 print(json.dumps(rows),flush=True)
if __name__=='__main__':main()
