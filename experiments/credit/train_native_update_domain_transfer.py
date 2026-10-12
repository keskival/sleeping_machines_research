"""B1/G4 frozen Taxi update policy transfers to StackOverflow without refitting."""
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
SOURCES=['experiments/credit/train_native_update_domain_transfer.py','experiments/credit/train_native_update_transfer.py','experiments/credit/train_native_future_update.py','experiments/tpp/race_tpp_v19.py']


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--tag',required=True)
    parser.add_argument('--stage',choices=['smoke','fit'],required=True);parser.add_argument('--seed',type=int,default=3);parser.add_argument('--gate');parser.add_argument('--dataset',choices=['stackoverflow'],default='stackoverflow')
    args=parser.parse_args();torch.set_num_threads(1);torch.set_default_dtype(torch.float64);start=time.monotonic()
    parent=ROOT/'experiments/results/credit/aws_native_future_update_reuse_20261011_s0.json'
    acquired=json.loads(parent.read_text());assert acquired['status']=='completed'
    sources=set(SOURCES)
    for s,digest in acquired['source_sha256'].items():
        assert hashlib.sha256((ROOT/s).read_bytes()).hexdigest()==digest;sources.add(s)
    checkpoint=ROOT/acquired['final_weights'];assert hashlib.sha256(checkpoint.read_bytes()).hexdigest()==acquired['checkpoint_sha256']
    saved=torch.load(checkpoint,weights_only=True)
    inputs={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [parent,checkpoint]}
    if args.stage=='fit':
        assert args.gate;gate_path=ROOT/args.gate;gate=json.loads(gate_path.read_text());assert gate['status']=='completed' and gate['stage']=='smoke'
        for s,digest in gate['source_sha256'].items():assert hashlib.sha256((ROOT/s).read_bytes()).hexdigest()==digest
        inputs[args.gate]=hashlib.sha256(gate_path.read_bytes()).hexdigest()
    length=8 if args.stage=='smoke' else 32;epochs=1 if args.stage=='smoke' else 3
    raw=base.native.load_split(args.dataset,'train');validation=base.native.load_split(args.dataset,'dev')
    full=[(t[:32],m[:32]) for t,m in raw if len(m)>=32][:1400];assert len(full)==1400
    gaps=np.concatenate([np.diff(t) for t,_ in full]);positive=gaps[gaps>0]
    scale=float(np.median(positive))
    taxi=base.native.load_split('taxi','train');taxi_gaps=np.concatenate([np.diff(t[:32]) for t,m in taxi if len(m)>=32])
    source_scale=float(np.median(taxi_gaps[taxi_gaps>0]));density_shift=np.log(scale/source_scale)
    source_path=ROOT/'data/easytpp/taxi/train.json';assert hashlib.sha256(source_path.read_bytes()).hexdigest()==acquired['data_sha256']['train']
    inputs[str(source_path.relative_to(ROOT))]=acquired['data_sha256']['train']
    quantiles=np.log(np.quantile(positive,[.1,.35,.65,.9])).tolist()
    train=[(t[:length],m[:length]) for t,m in full][:16 if args.stage=='smoke' else 1400]
    dev=[(t[:length],m[:length]) for t,m in validation if len(m)>=length][:8 if args.stage=='smoke' else 128]
    K=1+max(int(m.max()) for _,m in raw);rows=[];weights={}
    for depth in ([2] if args.stage=='smoke' else [2,4,8]):
        torch.manual_seed(args.seed+3000+depth);original=base.Actor(K,16,8,depth,2,4,4,0.,scale,quantiles).double()
        initial={k:v.detach().clone() for k,v in original.state_dict().items()}
        source_row=next(r for r in acquired['rows'] if r['depth']==depth and r['arm']=='future_taught')
        static=torch.tensor(source_row['history'][-1]['factors'])
        for arm in ['fixed_adam','static_acquired_groups','frozen_acquired_policy']:
            actor=base.Actor(K,16,8,depth,2,4,4,0.,scale,quantiles).double();actor.load_state_dict(initial)
            controller=base.Controller().double();controller.load_state_dict(saved[str(depth)]['future_taught']['controller'])
            controller.requires_grad_(False)
            names=list(dict(actor.named_parameters()));groups=[base.group(n) for n in names]
            moments={n:(torch.zeros_like(p),torch.zeros_like(p)) for n,p in actor.named_parameters()}
            history=[];updates=targets=0;fit_s=eval_s=0.;flops=0;scale_sum=torch.zeros(4)
            for epoch in range(epochs):
                order=np.random.default_rng(args.seed+epoch).permutation(len(train));ordered=[train[i] for i in order]
                blocks=[base.batch(ordered[i:i+8]) for i in range(0,len(ordered),8)]
                checkpoints=set(range(44,len(blocks),44))|{len(blocks)}
                for index,data in enumerate(blocks,1):
                    tick=time.monotonic()
                    with FlopCounterMode(display=False) as counter:
                        parameters=dict(actor.named_parameters());actor.zero_grad();loss,_,_,count=actor(*data);assert torch.isfinite(loss)
                        gradients=[g.detach() for g in torch.autograd.grad(loss,list(parameters.values()))]
                        norm=torch.sqrt(sum(g.square().sum() for g in gradients));clip=(10/norm.clamp_min(1e-30)).clamp_max(1)
                        gradients=[g*clip for g in gradients];updates+=1;targets+=int(count)
                        if arm=='fixed_adam':factors=torch.ones(4)
                        elif arm=='static_acquired_groups':factors=static
                        else:
                            norms=[torch.log1p(torch.stack([g.square().mean() for g,k in zip(gradients,groups) if k==j]).mean().sqrt()) for j in range(4)]
                            features=torch.stack([(loss.detach()-density_shift).clamp(-100,100)/10,torch.tensor(epoch/3),*norms])
                            factors=controller(features)
                        scale_sum+=factors
                        with torch.no_grad():
                            for n,g in zip(names,gradients):
                                m,v=moments[n];m=.9*m+.1*g;v=.999*v+.001*g.square();moments[n]=(m,v)
                                direction=(m/(1-.9**updates))/(torch.sqrt(v/(1-.999**updates))+1e-8)
                                parameters[n].sub_(.001*factors[base.group(n)]*direction)
                    fit_s+=time.monotonic()-tick;flops+=counter.get_total_flops()
                    if index in checkpoints:
                        tick=time.monotonic();quality=base.score(actor,dev);eval_s+=time.monotonic()-tick
                        history.append(dict(pass_position=epoch+index/len(blocks),updates=updates,**quality,
                                            fit_wall_s=fit_s,evaluation_wall_s=eval_s,supported_fit_flops=flops))
            row=dict(depth=depth,arm=arm,history=history,fit_wall_s=fit_s,evaluation_wall_s=eval_s,
                     supported_fit_flops=flops,updates=updates,train_target_presentations=targets,
                     controller_updates=0,mean_group_scales=(scale_sum/updates).tolist(),source_static_scales=static.tolist())
            rows.append(row);weights[f'{depth}_{arm}']=dict(actor=actor.state_dict(),moments=moments)
            print(json.dumps(dict(completed_arm=arm,depth=depth,final_nll=history[-1]['nll'],fit_wall_s=fit_s)),flush=True)
    decisions=[]
    for depth in sorted(set(r['depth'] for r in rows)):
        reference=next(r for r in rows if r['depth']==depth and r['arm']=='fixed_adam');target=reference['history'][-1]['nll']
        for row in [r for r in rows if r['depth']==depth]:
            match=next((h for h in row['history'] if h['nll']<=target),None)
            baseline_work=reference['fit_wall_s']+reference['evaluation_wall_s']
            matched_work=match['fit_wall_s']+row['evaluation_wall_s'] if match else None
            decisions.append(dict(depth=depth,arm=row['arm'],final_nll=row['history'][-1]['nll'],final_quality_gain=target-row['history'][-1]['nll'],
                match_pass=match['pass_position'] if match else None,matched_fit_and_all_validation_s=matched_work,
                baseline_fit_and_validation_s=baseline_work,baseline_over_matched_work=baseline_work/matched_work if match else None))
    out=ROOT/'experiments/results/credit'/f'{args.tag}.json';weight_path=out.with_suffix('.pt');assert not out.exists() and not weight_path.exists()
    torch.save(weights,weight_path)
    result=dict(status='completed',tag=args.tag,battle='B1/G4 acquired update domain transfer',stage=args.stage,seed=args.seed,rows=rows,decisions=decisions,
        final_weights=str(weight_path.relative_to(ROOT)),checkpoint_sha256=hashlib.sha256(weight_path.read_bytes()).hexdigest(),
        inherited_acquisition_wall_s=acquired['wall_s'],source_sha256={s:hashlib.sha256((ROOT/s).read_bytes()).hexdigest() for s in sorted(sources)},
        input_sha256=inputs,dataset=args.dataset,target_scale=scale,source_scale=source_scale,density_feature_shift=float(density_shift),data_sha256={split:hashlib.sha256((ROOT/f'data/easytpp/{args.dataset}/{split}.json').read_bytes()).hexdigest() for split in ['train','dev']},wall_s=time.monotonic()-start,max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        scope='Taxi seed0 policy fixed before StackOverflow actor fitting; no target policy refit. Density feature subtracts log(target_scale/source_scale), a declared time-unit canonicalization; static control identical source scales. All native parameters updated; dense writes/separate sequence state. FixedAdam vs TRAIN-acquired constant group scales vs frozen conditional function; matched data/updates and fine scheduled checkpoints. Zero controller optimizer/replay; inherited acquisition charged separately and required in amortized totals. Supported FLOPs partial; no TEST/Transformer supremacy claim.')
    with out.open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps(result))

if __name__=='__main__':main()
