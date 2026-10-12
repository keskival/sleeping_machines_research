"""B10: verify the state-credit blind spot and an actual optimizer response."""
import argparse,hashlib,json,resource,sys,time
from pathlib import Path
import torch
from torch.func import functional_call
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'experiments/credit'))
from market_native_test_time import restored,load,warm
from market_exact_anchor_future import Window,adam_proposal
from real_feedback_credit_metric_v2 import packet
from real_boundary_stable import control

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--tag',required=True);a=ap.parse_args();torch.set_num_threads(1);torch.set_default_dtype(torch.float64);begin=time.monotonic()
 model,inputs=restored();t,m,path=load('2026-08-24',144);inputs[str(path.relative_to(ROOT))]=sha(path);state=warm(model,t,m,128)
 produced,leaf,end,losses,_,_=packet(model,t,m,state,127,8,0);params=tuple(model.parameters());names=[n for n,_ in model.named_parameters()]
 probe=torch.autograd.grad(control.tensors(produced),params,grad_outputs=[torch.ones_like(x) for x in control.tensors(produced)],retain_graph=True,allow_unused=True)
 local=torch.autograd.grad(torch.stack(losses).mean(),params,retain_graph=True,allow_unused=True)
 readout=[i for i,n in enumerate(names) if n.startswith(('clock.','mark_ctx.')) or n=='mark_slot']
 assert readout and all(probe[i] is None or torch.count_nonzero(probe[i])==0 for i in readout)
 readout_nll_grad=sum(float(local[i].square().sum()) for i in readout if local[i] is not None);assert readout_nll_grad>0
 target=torch.autograd.grad(torch.stack(losses[1:]).sum()/len(losses),control.tensors(leaf),retain_graph=True)
 boundary=torch.autograd.grad(control.tensors(produced),params,grad_outputs=target,retain_graph=True,allow_unused=True)
 gradients=[(torch.zeros_like(p) if x is None else x.detach())+(torch.zeros_like(p) if y is None else y.detach()) for p,x,y in zip(params,local,boundary)]
 opt=torch.optim.Adam(model.parameters(),lr=.0003);before={n:p.detach().clone() for n,p in model.named_parameters()};proposed=adam_proposal(model,opt,gradients)
 delta={n:proposed['model.'+n].detach()-before[n] for n in names}
 for p,g in zip(params,gradients):p.grad=g.clone()
 torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step()
 parity=max(float((p.detach()-proposed['model.'+n]).abs().max()) for n,p in model.named_parameters());assert parity<1e-10
 alpha=torch.tensor(1.,requires_grad=True)
 intervention={'model.'+n:before[n]+alpha*delta[n] for n in names}
 future=functional_call(Window(model),intervention,(t,m,control.detach_state(end),135,8,1))
 derivative=torch.autograd.grad(future,alpha,retain_graph=True)[0]
 parameter_teacher=torch.autograd.grad(future,tuple(intervention.values()),retain_graph=True)
 projected=sum((g*delta[n]).sum() for n,g in zip(names,parameter_teacher));identity=float((projected-derivative).abs());assert identity<1e-10
 def evaluate(value):return float(functional_call(Window(model),{'model.'+n:before[n]+value*delta[n] for n in names},(t,m,control.detach_state(end),135,8,1)).detach())
 fd_curve=[]
 for eps in [1e-2,3e-3,1e-3]:
  fd=(evaluate(1+eps)-evaluate(1-eps))/(2*eps);fd_curve.append(dict(epsilon=eps,derivative=fd,error=abs(fd-float(derivative))))
 print(json.dumps(dict(analytic_derivative=float(derivative),finite_difference_curve=fd_curve)),flush=True)
 fd_error=min(x['error'] for x in fd_curve);assert fd_error<1e-6
 sources=['experiments/credit/market_credit_reachability_contract_v2.py','experiments/credit/market_native_test_time.py','experiments/credit/market_exact_anchor_future.py','experiments/credit/real_feedback_credit_metric_v2.py','experiments/market/episode_state.py','experiments/tpp/race_tpp_v5.py']
 result=dict(status='completed',battle='B10',checks=dict(state_credit_zero_for_clock_readout=True,readout_parameters=sum(params[i].numel() for i in readout),actor_parameters=sum(p.numel() for p in params),readout_current_loss_gradient_squared_norm=readout_nll_grad,readout_names=[names[i] for i in readout],actual_adam_proposal_error=parity,optimizer_control_utility_derivative=float(derivative),utility_identity_error=identity,utility_finite_difference_error=fd_error,finite_difference_curve=fd_curve),source_sha256={p:sha(ROOT/p) for p in sources},input_sha256=inputs,whole_wall_s=time.monotonic()-begin,peak_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,scope='Existing full market actor on16revealed TRAIN targets; complete exact current gradients, one actual Adam proposal parity and next-window scalar learning-rate utility derivative. Parameter groups used only for diagnosis. No trained optimizer controller, market-quality win or sparse hardware claim.')
 with (ROOT/f'experiments/results/market/{a.tag}.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
 print(json.dumps(result['checks']),flush=True)
if __name__=='__main__':main()
