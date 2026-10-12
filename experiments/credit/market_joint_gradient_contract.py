"""B10: test whether exact cotangent extraction shares the local backward sweep."""
import argparse,hashlib,json,resource,sys,time
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'experiments/credit'))
from market_native_test_time import restored,load,warm
from real_feedback_credit_metric_v2 import packet
from real_boundary_stable import control
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--tag',required=True);a=ap.parse_args();torch.set_num_threads(1);torch.set_default_dtype(torch.float64);begin=time.monotonic()
 model,inputs=restored();t,m,path=load('2026-08-24',160);inputs[str(path.relative_to(ROOT))]=sha(path);state=warm(model,t,m,128);rows=[]
 for group in [8,32]:
  produced,leaf,end,losses,_,_=packet(model,t,m,state,127,group,0);params=tuple(model.parameters());leaves=tuple(control.tensors(leaf));mean=torch.stack(losses).mean()
  local=torch.autograd.grad(mean,params,retain_graph=True,allow_unused=True)
  target=torch.autograd.grad(torch.stack(losses[1:]).sum()/len(losses),leaves,retain_graph=True)
  joint=torch.autograd.grad(mean,params+leaves,retain_graph=True,allow_unused=True)
  pe=max(float(((torch.zeros_like(p) if x is None else x)-(torch.zeros_like(p) if y is None else y)).abs().max()) for p,x,y in zip(params,local,joint[:len(params)]))
  ce=max(float((x-y).abs().max()) for x,y in zip(target,joint[len(params):]));assert max(pe,ce)<1e-12
  separate_times=[];joint_times=[]
  for _ in range(5):
   started=time.monotonic();torch.autograd.grad(mean,params,retain_graph=True,allow_unused=True);torch.autograd.grad(torch.stack(losses[1:]).sum()/len(losses),leaves,retain_graph=True);separate_times.append(time.monotonic()-started)
   started=time.monotonic();torch.autograd.grad(mean,params+leaves,retain_graph=True,allow_unused=True);joint_times.append(time.monotonic()-started)
  rows.append(dict(group=group,parameter_gradient_max_error=pe,boundary_cotangent_max_error=ce,separate_backward_wall_s=separate_times,joint_backward_wall_s=joint_times))
 sources=['experiments/credit/market_joint_gradient_contract.py','experiments/credit/market_native_test_time.py','experiments/credit/real_feedback_credit_metric_v2.py','experiments/credit/real_boundary_stable.py','experiments/market/episode_state.py','experiments/tpp/race_tpp_v5.py']
 result=dict(status='completed',battle='B10',rows=rows,source_sha256={p:sha(ROOT/p) for p in sources},input_sha256=inputs,whole_wall_s=time.monotonic()-begin,peak_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,scope='TRAIN-only reused-graph gradient equality and diagnostic backward timing. No model fit or whole-learning speed claim. Same exact boundary cotangent may be extracted with parameter gradients in one backward pass, tightening the actual credit-replacement opportunity. Reused graphs/warm timing are not independent full fits.')
 with (ROOT/f'experiments/results/market/{a.tag}.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
 print(json.dumps(rows),flush=True)
if __name__=='__main__':main()
