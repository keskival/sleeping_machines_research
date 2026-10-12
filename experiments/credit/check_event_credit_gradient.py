"""B1 exact enumeration of a real native discrete/continuous future-teaching gradient."""
import argparse,hashlib,json,resource,time
from pathlib import Path
import torch
from torch.func import functional_call
import run_native_event_credit as run

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--tag',required=True);a=ap.parse_args();root=run.ROOT;out=root/f'experiments/results/credit/{a.tag}.json';assert not out.exists();start=time.monotonic();torch.set_num_threads(1);torch.set_default_dtype(torch.float64);run.RUN_WIDTH=8;run.extension();actor=run.initial(99910,4,10,1.,[-1.,0.,1.]);data=[s for s in run.native.load_split('taxi','train') if len(s[1])>=32];K=1+max(int(m.max()) for _,m in data);actor=run.initial(99910,4,K,1.,[-1.,0.,1.]);run.install_event_hooks(actor);actor.collect_events=True;actor.cut=True;t,m,mask=run.one_block(data[1328]);loss=actor(t,m,mask);c=torch.autograd.grad(loss,actor.terminal)[0];p=run.context(actor,c,loss,t);field=run.new_credit(actor,4)
    with torch.no_grad():
        for decoder in field.decoders.values():decoder.weight.normal_(0,.02);decoder.bias.normal_(0,.01)
        field.query.weight.normal_(0,.02)
    tf,mf,mfmask=run.one_block(data[1329]);actor.cut=False;actor.collect_events=False
    with torch.no_grad():baseline=actor(tf,mf,mfmask)
    values=[];logs=[];pool=torch.tensor([16,24]);named=dict(actor.named_parameters())
    for chosen in pool:
        direction=field(p,forced=[chosen],pool=pool);parts=direction.split([named[n].numel() for n in field.names]);gs={n:x.reshape_as(named[n]) for n,x in zip(field.names,parts)};norm=torch.stack([g.square().sum() for g in gs.values()]).sum().clamp_min(1e-24).sqrt();factor=(5/norm).clamp_max(1);issued={n:v.detach()-.001*factor*gs[n] if run.core_name(n) else v.detach() for n,v in named.items()};values.append(functional_call(actor,issued,(tf,mf,mfmask)));logs.append(field.last_logprob)
    probs=torch.stack(logs).exp();assert abs(float(probs.sum().detach())-1)<1e-12;exact=sum(pr*l for pr,l in zip(probs,values));surrogate=sum(pr.detach()*(l+(l.detach()-baseline)*lp) for pr,l,lp in zip(probs,values,logs));params=list(field.parameters());g1=torch.autograd.grad(exact,params,retain_graph=True,allow_unused=True);g2=torch.autograd.grad(surrogate,params,allow_unused=True);errors=[float((x-y).abs().max()) for x,y in zip(g1,g2) if x is not None and y is not None];error=max(errors);contrast=abs(float((values[0]-values[1]).detach()));query_index=next(i for i,x in enumerate(params) if x is field.query.weight);query_norm=float(g1[query_index].norm());gate=error<1e-10 and contrast>1e-10 and query_norm>1e-12;assert gate,(error,contrast,query_norm)
    sources=['experiments/credit/check_event_credit_gradient.py','experiments/credit/event_parameter_credit.py','experiments/credit/run_native_event_credit.py','experiments/credit/NATIVE_EVENT_CREDIT_PROTOCOL.md','experiments/tpp/race_tpp_v19.py','experiments/tpp/temporal_scan_cpu.py','experiments/tpp/temporal_scan_cpu.cpp'];record=dict(status='completed',battle='B1',tag=a.tag,gate=gate,max_gradient_error=error,future_loss_contrast=contrast,query_gradient_norm=query_norm,probability_sum=float(probs.sum().detach()),wall_s=time.monotonic()-start,peak_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,source_sha256={s:hashlib.sha256((root/s).read_bytes()).hexdigest() for s in sources},scope='Real native depth4,width8 Taxi TRAIN; two enumerated single-event query actions, local finite update expectation only. Algebra check, no learning utility claim.');out.write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record),flush=True)
if __name__=='__main__':main()
