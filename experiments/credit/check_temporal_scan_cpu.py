"""R1/B10 exact CPU scan contracts and bounded work measurement; no fitting."""
import argparse
import hashlib
import json
import resource
import sys
import time
from copy import deepcopy
from types import MethodType
from pathlib import Path

import torch
from reciprocal_native_v1 import ROOT, native_pair, batch
sys.path.insert(0,str(ROOT/'experiments/tpp'))
from temporal_scan_cpu import scan, reference, layer_forward, extension

SOURCES=['experiments/credit/check_temporal_scan_cpu.py','experiments/tpp/temporal_scan_cpu.py',
    'experiments/tpp/temporal_scan_cpu.cpp','experiments/credit/reciprocal_native_v1.py',
    'experiments/tpp/recall_tpp_v4.py','experiments/tpp/race_tpp_v5.py']


def tensors(B,L,N):
    angle=torch.randn(B,L,N,dtype=torch.float64)
    values=[.95*angle.cos(),.95*angle.sin(),torch.randn_like(angle)*.1,torch.randn_like(angle)*.1,
            torch.randn(B,2*N,dtype=torch.float64)*.1]
    return [v.detach().requires_grad_(True) for v in values]


def compare(B,L,N):
    x=tensors(B,L,N);y=[v.detach().clone().requires_grad_(True) for v in x]
    original=reference(*x);fast=scan(*y);weight=torch.randn_like(original)
    g=torch.autograd.grad((original*weight).sum(),x)
    h=torch.autograd.grad((fast*weight).sum(),y)
    errors=dict(forward=float((original-fast).abs().max()),
                gradient=max(float((a-b).abs().max()) for a,b in zip(g,h)))
    assert max(errors.values())<1e-10,errors
    # Persistent carry across chunks preserves full gradients unless explicitly detached.
    z=[v.detach().clone().requires_grad_(True) for v in x];split=L//2
    if split:
        left=scan(*(v[:,:split] for v in z[:4]),z[4])
        right=scan(*(v[:,split:] for v in z[:4]),left[:,-1])
        joined=torch.cat((left,right),1);jg=torch.autograd.grad((joined*weight).sum(),z)
        errors['chunked_forward']=float((joined-original).abs().max())
        errors['chunked_gradient']=max(float((a-b).abs().max()) for a,b in zip(g,jg))
        assert max(errors.values())<1e-10,errors
    return dict(batch=B,length=L,modes=N,errors=errors)


def integrated(depth):
    original,_=native_pair(depth,175);fast=deepcopy(original)
    for layer in fast.model.layers:layer.forward=MethodType(layer_forward,layer)
    t,m=batch(175,pairs=3);loss=original(t,m);other=fast(t,m)
    op=list(original.named_parameters());fp=list(fast.named_parameters())
    assert [k for k,_ in op]==[k for k,_ in fp]
    g=torch.autograd.grad(loss,[v for _,v in op],allow_unused=True)
    h=torch.autograd.grad(other,[v for _,v in fp],allow_unused=True)
    error=0.;count=0
    for (_,p),a,b in zip(op,g,h):
        a=torch.zeros_like(p) if a is None else a;b=torch.zeros_like(p) if b is None else b
        error=max(error,float((a-b).abs().max()));count+=p.numel()
    assert error<1e-10 and abs(float(loss-other))<1e-12
    return dict(depth=depth,parameters=count,loss_error=abs(float(loss-other)),maximum_parameter_gradient_error=error)


def contracts():
    torch.manual_seed(175)
    raw=[compare(2,17,3),compare(1,129,16),compare(1,1024,16)]
    native=[integrated(d) for d in (2,4,8)]
    return dict(scan=raw,native=native,first_order_only=True,
        numerical_state_carry=True,chunked_gradient_preserved=True,
        scope='Identical native model and complete first-order gradients. Double backward/meta-Hessians are unsupported; do not substitute this backend in higher-order meta-gradient contracts.')


def smoke():
    torch.manual_seed(175);rows=[]
    for B,L,N in [(1,128,16),(8,1024,16)]:
        timing={}
        for name,function in [('reference',reference),('scan',scan)]:
            x=tensors(B,L,N);start=time.monotonic();out=function(*x)
            torch.autograd.grad(out.square().sum(),x)
            timing[name]=time.monotonic()-start
        # Floating multiply/add arithmetic for this kernel, including carry adjoint.
        rows.append(dict(batch=B,length=L,modes=N,timing_s=timing,
             scalar_forward_flops=8*B*L*N,scalar_backward_flops=14*B*L*N,
             source_coefficient_elements=4*B*L*N,output_state_elements=2*B*L*N,
             runtime_reference_over_scan=timing['reference']/timing['scan']))
    return dict(rows=rows,scope='One bounded forward/backward observation per shape; warm cached compilation, exact recurrence kernel only. Native model projections, clocks, mark memory, optimizer and credit-model work are excluded. Whole training speedup unmeasured.')


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--tag',required=True)
    ap.add_argument('--mode',choices=['contract','smoke'],required=True);a=ap.parse_args()
    out=ROOT/'experiments/results/credit'/(a.tag+'.json')
    if out.exists():raise FileExistsError(out)
    torch.set_num_threads(1);torch.set_default_dtype(torch.float64);start=time.monotonic()
    extension();result=contracts() if a.mode=='contract' else smoke()
    record=dict(status='completed',tag=a.tag,battle='R1/B10',training=False,args=vars(a),metrics=result,
        decision='Can exact compiled recurrence/adjoint preserve native deep learning contracts and reduce execution overhead?',
        wall_s=time.monotonic()-start,peak_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        source_sha256={s:hashlib.sha256((ROOT/s).read_bytes()).hexdigest() for s in SOURCES},
        compiler_threads=1,torch_version=torch.__version__)
    temporary=out.with_suffix('.tmp');temporary.write_text(json.dumps(record,indent=2)+'\n');temporary.replace(out)
    print('RESULT',json.dumps(record),flush=True)


if __name__=='__main__':main()
