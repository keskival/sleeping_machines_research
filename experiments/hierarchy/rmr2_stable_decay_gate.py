"""Long-window standing decay oracle, numerical derivative and finite-gradient admission."""
import argparse,hashlib,json,resource,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'experiments'),str(ROOT)]
import torch
from r1_rmr2 import RMR2,contracts
from r1_rmr2_stable import StableRMR2
p=argparse.ArgumentParser();p.add_argument('--tag',required=True);a=p.parse_args();torch.set_num_threads(1);torch.manual_seed(16016);start=time.monotonic()
checks=contracts();assert checks['status']=='passed'
# Exercise the real method without building a language model. Three standing slots, 8192 possible writes.
class Fixture:
 standing_coeff=StableRMR2.standing_coeff
 def __init__(self):self.st_decay=torch.full((3,),-3.0,requires_grad=True)
f=Fixture();T=8192;ds=torch.randn(1,T,2,requires_grad=True);ws=torch.softmax(torch.randn(1,T,3),-1).requires_grad_();errors=[]
for c0,lim in [(0,'le'),(4096,'le'),(8128,'le'),(4096,4096)]:
 rows=torch.arange(c0,c0+64);aw=torch.softmax(torch.randn(1,64,3),-1).requires_grad_();actual=f.standing_coeff(aw,[(ds,ws,lim)],rows,T)
 tau=torch.arange(T);valid=(tau[None,:]<=rows[:,None]) if lim=='le' else (tau[None,:]<lim).expand(64,T)
 delta=(rows[:,None]-tau[None,:]).clamp_min(0).float();rate=torch.nn.functional.softplus(f.st_decay).clamp(max=.5)
 decay=torch.exp(-delta[:,:,None]*rate);coef=torch.einsum('bru,btu,rtu->brt',aw,ws,decay).masked_fill(~valid,0);expected=coef@ds
 err=float((actual-expected).abs().max());errors.append(err);assert err<1e-4,err
 grads=torch.autograd.grad(actual.square().mean(),[f.st_decay,aw,ws,ds],retain_graph=True);assert all(bool(torch.isfinite(g).all()) for g in grads)
# A formerly masked-out future write must not poison an early read.
legacy=RMR2.standing_coeff(f,torch.ones(1,64,3)/3,[(ds,ws,'le')],torch.arange(64),T)
assert not bool(torch.isfinite(legacy).all()),'witness no longer reproduces; inspect upstream repair'
# Long-window finite differences of decay on the stable implementation.
rows=torch.arange(4096,4160);aw=torch.ones(1,64,3)/3;val=f.standing_coeff(aw,[(ds,ws,'le')],rows,T).double().sum();analytic=torch.autograd.grad(val,f.st_decay)[0];eps=1e-3
with torch.no_grad():
 f.st_decay[0]+=eps;plus=f.standing_coeff(aw,[(ds,ws,'le')],rows,T).double().sum();f.st_decay[0]-=2*eps;minus=f.standing_coeff(aw,[(ds,ws,'le')],rows,T).double().sum();f.st_decay[0]+=eps
error=abs(float((plus-minus)/(2*eps)-analytic[0]));assert error<.05,error
sources=['experiments/hierarchy/rmr2_stable_decay_gate.py','experiments/r1_rmr2_stable.py','experiments/r1_rmr2_stable_stream.py','experiments/r1_rmr2.py','experiments/r1_rmr.py','experiments/r1_cerl.py','experiments/r1_deep_keyed_lm.py','experiments/r1_token_keyed_lm.py']
r=dict(status='completed',battle='R1',stage='RMR2 stable decay admission',gate='PASS',legacy_nonfinite_reproduced=True,long_window=8192,oracle_max_errors=errors,decay_finite_difference_error=error,upstream_contracts=checks,elapsed_s=time.monotonic()-start,max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,source_sha256={s:hashlib.sha256((ROOT/s).read_bytes()).hexdigest() for s in sources})
with (ROOT/f'experiments/results/diagnostics/{a.tag}.json').open('x') as out:json.dump(r,out,indent=2);out.write('\n')
print(json.dumps(r),flush=True)
