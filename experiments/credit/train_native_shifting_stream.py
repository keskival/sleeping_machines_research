"""B1/G4 long prequential stream with controlled distribution and dependency changes."""
import argparse
import hashlib
import json
import resource
import time
import numpy as np
import torch
from torch.utils.flop_counter import FlopCounterMode
import train_native_future_update as base
ROOT=base.ROOT


def transform(sequence,phase,K,median):
    t,m=sequence;t=np.asarray(t).copy();m=np.asarray(m).copy()
    if phase==1:t*=8
    if phase==2:m=(m+1)%K
    if phase==3:
        gap=np.diff(t,prepend=t[0]);m=(m+(gap>median).astype(m.dtype))%K
    return t,m


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--tag',required=True);ap.add_argument('--stage',choices=['smoke','fit'],required=True)
    ap.add_argument('--seed',type=int,default=9);ap.add_argument('--gate');a=ap.parse_args()
    torch.set_num_threads(1);torch.set_default_dtype(torch.float64);start=time.monotonic()
    parent=ROOT/'experiments/results/credit/aws_native_future_update_reuse_20261011_s0.json';acquisition=json.loads(parent.read_text());assert acquisition['status']=='completed'
    sources={'experiments/credit/train_native_shifting_stream.py',*acquisition['source_sha256']}
    for source,digest in acquisition['source_sha256'].items():assert hashlib.sha256((ROOT/source).read_bytes()).hexdigest()==digest
    weight_path=ROOT/acquisition['final_weights'];assert hashlib.sha256(weight_path.read_bytes()).hexdigest()==acquisition['checkpoint_sha256']
    acquired=torch.load(weight_path,weights_only=True)
    inputs={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [parent,weight_path]}
    if a.stage=='fit':
        assert a.gate;gate_path=ROOT/a.gate;gate=json.loads(gate_path.read_text());assert gate['status']=='completed' and gate['stage']=='smoke'
        for source,digest in gate['source_sha256'].items():assert hashlib.sha256((ROOT/source).read_bytes()).hexdigest()==digest
        inputs[a.gate]=hashlib.sha256(gate_path.read_bytes()).hexdigest()
    raw=[(t,m) for t,m in base.native.load_split('retweet','train') if len(m)>=32];assert len(raw)>=20000
    calibration=raw[:256];K=1+max(int(m.max()) for _,m in calibration)
    positive=np.concatenate([np.diff(t[:32]) for t,_ in calibration]);positive=positive[positive>0]
    scale=float(np.median(positive));quantiles=np.log(np.quantile(positive,[.1,.35,.65,.9])).tolist()
    taxi=base.native.load_split('taxi','train');gaps=np.concatenate([np.diff(t[:32]) for t,m in taxi if len(m)>=32]);source_scale=float(np.median(gaps[gaps>0]))
    source_data=ROOT/'data/easytpp/taxi/train.json';assert hashlib.sha256(source_data.read_bytes()).hexdigest()==acquisition['data_sha256']['train'];inputs[str(source_data.relative_to(ROOT))]=acquisition['data_sha256']['train']
    count=16 if a.stage=='smoke' else 12800;length=8 if a.stage=='smoke' else 32;batch_size=4 if a.stage=='smoke' else 8
    selected=raw[19984:20000] if a.stage=='smoke' else raw[256:13056]
    stream=[transform((t[:length],m[:length]),i//(count//4),K,scale) for i,(t,m) in enumerate(selected)]
    assert len(stream)==count and all(int(m.max())<K for _,m in stream)
    blocks=[base.batch(stream[i:i+batch_size]) for i in range(0,count,batch_size)]
    per_regime=len(blocks)//4;rows=[];weights={}
    for depth in ([4] if a.stage=='smoke' else [4,8]):
        torch.manual_seed(3000+a.seed+depth);original=base.Actor(K,16,8,depth,2,4,4,0.,scale,quantiles).double();initial={n:p.detach().clone() for n,p in original.state_dict().items()}
        static=torch.tensor(next(r for r in acquisition['rows'] if r['depth']==depth and r['arm']=='future_taught')['history'][-1]['factors'])
        for arm in ['fixed_adam','static_acquired_groups','frozen_acquired_policy','online_acquired_policy']:
            actor=base.Actor(K,16,8,depth,2,4,4,0.,scale,quantiles).double();actor.load_state_dict(initial)
            controller=base.Controller().double();controller.load_state_dict(acquired[str(depth)]['future_taught']['controller']);controller.requires_grad_(arm=='online_acquired_policy')
            meta=torch.optim.Adam(controller.parameters(),lr=.001) if arm=='online_acquired_policy' else None
            names=list(dict(actor.named_parameters()));groups=[base.group(n) for n in names]
            moments={n:(torch.zeros_like(p),torch.zeros_like(p)) for n,p in actor.named_parameters()};issued=None
            history=[];losses=[];fit_wall=flops=0;targets=meta_steps=0;sum_nll=0.;regimes=[]
            for step,data in enumerate(blocks):
                tick=time.monotonic()
                with FlopCounterMode(display=False) as counter:
                    parameters=dict(actor.named_parameters());actor.zero_grad();loss,tl,ml,number=actor(*data);assert torch.isfinite(loss)
                    # Score before this batch's outcomes update either actor or controller.
                    losses.append(float(loss.detach()));targets+=int(number);sum_nll+=float(loss.detach())*int(number)
                    gradients=[g.detach() for g in torch.autograd.grad(loss,list(parameters.values()))]
                    if issued is not None:
                        factors_previous,directions_previous=issued
                        teacher=sum(factors_previous[base.group(n)]*(g*directions_previous[n]).sum() for n,g in zip(names,gradients))
                        objective=-.001*teacher+.001*torch.log(factors_previous).square().mean()
                        meta.zero_grad();objective.backward();torch.nn.utils.clip_grad_norm_(controller.parameters(),1.);meta.step();meta_steps+=1;issued=None
                    norm=torch.sqrt(sum(g.square().sum() for g in gradients));clip=(10/norm.clamp_min(1e-30)).clamp_max(1);gradients=[g*clip for g in gradients]
                    directions={}
                    for n,g in zip(names,gradients):
                        m,v=moments[n];m=.9*m+.1*g;v=.999*v+.001*g.square();moments[n]=(m,v)
                        directions[n]=(m/(1-.9**(step+1)))/(torch.sqrt(v/(1-.999**(step+1)))+1e-8)
                    if arm=='fixed_adam':factors=torch.ones(4)
                    elif arm=='static_acquired_groups':factors=static
                    else:
                        norms=[torch.log1p(torch.stack([g.square().mean() for g,k in zip(gradients,groups) if k==j]).mean().sqrt()) for j in range(4)]
                        features=torch.stack([(loss.detach()-np.log(scale/source_scale)).clamp(-100,100)/10,torch.tensor(step/len(blocks)),*norms])
                        factors=controller(features)
                    with torch.no_grad():
                        for n,p in parameters.items():p.sub_(.001*factors[base.group(n)]*directions[n])
                    if meta is not None and step+1<len(blocks):issued=(factors,directions)
                fit_wall+=time.monotonic()-tick;flops+=counter.get_total_flops()
                if (step+1)%64==0 or (step+1)%per_regime==0:
                    history.append(dict(updates=step+1,targets=targets,cumulative_nll=sum_nll/targets,fit_wall_s=fit_wall,supported_fit_flops=flops,scales=factors.detach().tolist()))
                if (step+1)%per_regime==0:
                    values=losses[-per_regime:];window=min(32,len(values))
                    regimes.append(dict(regime=len(regimes),mean_nll=float(np.mean(values)),first_window_nll=float(np.mean(values[:window])),last_window_nll=float(np.mean(values[-window:])),adaptation_window_updates=window))
            row=dict(depth=depth,arm=arm,cumulative_nll=sum_nll/targets,targets=targets,updates=len(blocks),controller_updates=meta_steps,
                     fit_wall_s=fit_wall,supported_fit_flops=flops,history=history,regimes=regimes,batch_prequential_nll=losses)
            rows.append(row);weights[f'{depth}_{arm}']=dict(actor=actor.state_dict(),controller=controller.state_dict(),moments=moments)
            print(json.dumps(dict(completed_depth=depth,completed_arm=arm,cumulative_nll=row['cumulative_nll'],fit_wall_s=fit_wall,regimes=regimes)),flush=True)
    out=ROOT/'experiments/results/credit'/f'{a.tag}.json';checkpoint=out.with_suffix('.pt');assert not out.exists() and not checkpoint.exists();torch.save(weights,checkpoint)
    result=dict(status='completed',tag=a.tag,battle='B1/G4 changing-stream learning',stage=a.stage,args=vars(a),rows=rows,
                source_sha256={s:hashlib.sha256((ROOT/s).read_bytes()).hexdigest() for s in sorted(sources)},input_sha256=inputs,
                data_sha256=hashlib.sha256((ROOT/'data/easytpp/retweet/train.json').read_bytes()).hexdigest(),
                final_weights=str(checkpoint.relative_to(ROOT)),checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                inherited_acquisition_wall_s=acquisition['wall_s'],stream_sequences=count,events_per_sequence=length,distinct_calibration_sequences=256,
                regime_description=['original','time scale x8','cyclic mark relabeling','gap-dependent mark mapping'],
                wall_s=time.monotonic()-start,max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                scope='Constructed ordered regimes over distinct real Retweet TRAIN sequences, not natural chronological distribution drift. Batch-prequential scoring before feedback; actor and Adam/controller states carried across regime boundaries; internal forward state reset between independent sequences. First256earlier sequences calibrate vocabulary/time only. Dense native writes; no source actor transfer, no target future/change labels supplied. Exact actor backprop retained; acquired update function taught from actual next gradient without replay. All work measured, acquisition separate; no TEST/Transformer supremacy or large-data asymptotic claim.')
    with out.open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps(result))

if __name__=='__main__':main()
