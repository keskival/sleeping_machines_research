"""R1/B1 bounded complete native forward/backward/Adam measurement.
No benchmark fit or test set; identical complete factual gradients in both arms.
"""
import argparse,hashlib,json,resource,statistics,time
from copy import deepcopy
from types import MethodType
import torch
from reciprocal_native_v1 import ROOT,native_pair,batch
from temporal_scan_cpu import layer_forward,extension
SOURCES=['experiments/credit/measure_integrated_scan.py','experiments/credit/reciprocal_native_v1.py','experiments/tpp/recall_tpp_v4.py','experiments/tpp/race_tpp_v5.py','experiments/tpp/temporal_scan_cpu.py','experiments/tpp/temporal_scan_cpu.cpp']

def step(model,opt,data):
    start=time.perf_counter();opt.zero_grad(set_to_none=True);loss=model(*data);loss.backward()
    grads=[torch.zeros_like(p) if p.grad is None else p.grad.detach().clone() for p in model.parameters()]
    opt.step();elapsed=time.perf_counter()-start
    return float(loss.detach()),grads,elapsed

def measure(depth,pairs):
    reference,_=native_pair(depth,181);fast=deepcopy(reference)
    for layer in fast.model.layers:layer.forward=MethodType(layer_forward,layer)
    opts=[torch.optim.Adam(m.parameters(),lr=.001) for m in (reference,fast)]
    timings={'reference':[],'scan':[]};maxgrad=maxloss=maxparam=maxstate=0.;losses=[]
    for k in range(4):
        data=batch(18100+k,size=1,pairs=pairs);result={}
        # Alternate order to reduce systematic cache/timing-order effects.
        for i in ((0,1) if k%2==0 else (1,0)):
            name=('reference','scan')[i];result[name]=step((reference,fast)[i],opts[i],data)
            if k:timings[name].append(result[name][2])
        a,b=result['reference'],result['scan'];maxloss=max(maxloss,abs(a[0]-b[0]));maxgrad=max(maxgrad,max(float((x-y).abs().max()) for x,y in zip(a[1],b[1])))
        maxparam=max(maxparam,max(float((x-y).abs().max().detach()) for x,y in zip(reference.parameters(),fast.parameters())))
        for x,y in zip(opts[0].state.values(),opts[1].state.values()):
            for key in x:maxstate=max(maxstate,float((x[key]-y[key]).abs().max()))
        losses.append(a[0]);assert max(maxgrad,maxloss,maxparam,maxstate)<1e-10
    L=int(data[0].shape[1]);median={k:statistics.median(v) for k,v in timings.items()}
    return dict(depth=depth,batch=1,length=L,parameters=sum(p.numel() for p in reference.parameters()),scored_targets_per_step=L-1,steps_per_arm=4,warmup_steps=1,timing_s=timings,median_step_s=median,speedup=median['reference']/median['scan'],maximum_loss_error=maxloss,maximum_gradient_error=maxgrad,maximum_parameter_error=maxparam,maximum_optimizer_state_error=maxstate,reference_losses=losses,scope='Entire native likelihood, backward and Adam plus identical gradient-copy instrumentation; CPU one thread float64. Four updates per arm, three timed after warmup; not work-to-quality or a sparse-learning result.')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--tag',required=True);a=ap.parse_args();out=ROOT/'experiments/results/credit'/f'{a.tag}.json';assert not out.exists()
    torch.set_num_threads(1);torch.set_default_dtype(torch.float64);start=time.monotonic();extension()
    rows=[measure(d,p) for d in (2,4,8) for p in (32,128)]
    result=dict(status='completed',tag=a.tag,battle='R1/B1',training='bounded four-update numerical/throughput diagnostic per arm and shape',metrics=rows,wall_s=time.monotonic()-start,peak_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,source_sha256={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in SOURCES},decision='Do exact recurrence execution savings persist through complete native learning and optimizer updates as depth/horizon grow?',work_scope='All timed complete steps included; no FLOP/energy advantage claimed. Same model arithmetic in both arms, execution backend differs. Persistent mark memory/key scoring remain dense; no higher-order gradients.')
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
if __name__=='__main__':main()
