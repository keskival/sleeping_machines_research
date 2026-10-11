"""B1 same-policy-family direct fitness versus forward-label allocation regression."""
import argparse
import hashlib
import json
from pathlib import Path
import resource
import time
import torch
from budgeted_audit_policy import budgeted_probabilities

ROOT=Path(__file__).resolve().parents[2]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tag',required=True)
    parser.add_argument('--seed',type=int,required=True,choices=[0,1,2])
    args=parser.parse_args();torch.set_num_threads(1);torch.set_default_dtype(torch.float64)
    start=time.monotonic();rows=[];sources={'experiments/credit/fit_audit_objective_ablation.py','experiments/credit/budgeted_audit_policy.py'}
    parents=[ROOT/'experiments/results/credit'/f'aws_native_joint_audit_risk_ladder_20261011_s{args.seed}.json',
             ROOT/'experiments/results/credit'/f'aws_direct_audit_policy_20261011_s{args.seed}.json']
    ladder,direct=[json.loads(p.read_text()) for p in parents]
    for record in [ladder,direct]:
        assert record['status']=='completed'
        for source,digest in record['source_sha256'].items():
            assert hashlib.sha256((ROOT/source).read_bytes()).hexdigest()==digest
            sources.add(source)
    for depth in [2,4,8]:
        for size in [64,256,1024]:
            records=ladder['datasets'][str(depth)]['training_records'][:size]
            raw=torch.tensor([r['features'] for r in records]);center=raw.mean(0)
            scale=raw.std(0,unbiased=False).clamp_min(1e-6);X=(raw-center)/scale
            c2=torch.tensor([r['cotangent_squared'] for r in records])
            labels=torch.tensor([r['forward_target']*r['cotangent_squared'] for r in records])
            # Forward-only proxy for optimal batch allocation; no exact reverse fit labels.
            target=budgeted_probabilities(.5*torch.log(labels.clamp_min(1e-30)),size*.125).detach()
            offset=.5*torch.log(c2.clamp_min(1e-20));policy=torch.nn.Linear(12,1)
            with torch.no_grad():policy.weight.zero_();policy.bias.zero_()
            optimizer=torch.optim.Adam(policy.parameters(),lr=.01)
            for _ in range(300):
                optimizer.zero_grad();p=budgeted_probabilities(offset+policy(X).squeeze(-1),size*.125)
                # Relative scale gives unit loss at initialization, matching direct normalized fitness.
                if _==0:normalizer=(p.detach()-target).square().mean().clamp_min(1e-20)
                loss=(p-target).square().mean()/normalizer+.01*policy.weight.square().sum()
                loss.backward();optimizer.step()
            held=direct['heldout_records'][str(depth)]
            held_raw=torch.tensor([r['features'] for r in held]);held_c2=torch.tensor([r['cotangent_squared'] for r in held])
            with torch.no_grad():
                allocation=budgeted_probabilities(.5*torch.log(held_c2.clamp_min(1e-20))+policy((held_raw-center)/scale).squeeze(-1),16.)
            truth=torch.tensor([r['exact_risk'] for r in held]);variance=float((truth*(1/allocation-1)).sum())
            baseline=next(r for r in direct['rows'] if r['depth']==depth and r['train_contexts']==size)
            rows.append(dict(depth=depth,train_contexts=size,allocation_regression_variance=variance,
                direct_variance=baseline['variance_objective']['direct'],
                direct_reduction_vs_allocation_regression=1-baseline['variance_objective']['direct']/variance,
                allocation_regression_probabilities=allocation.tolist(),loss=float(loss.detach())))
    result=dict(status='completed',tag=args.tag,battle='B1 audit objective isolation',seed=args.seed,rows=rows,
        source_sha256={s:hashlib.sha256((ROOT/s).read_bytes()).hexdigest() for s in sorted(sources)},
        input_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in parents},
        wall_s=time.monotonic()-start,max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        scope='Same sigmoid-budget linear policy, initialization, features, support, budget,300 Adam updates and forward fitting labels. Alternative loss regresses the forward-label optimal allocation. Previously evaluated private TRAIN contexts reused for diagnostic comparison, no new independent confirmation or native future-quality claim. All inherited acquisition remains charged in parents.')
    with (ROOT/'experiments/results/credit'/f'{args.tag}.json').open('x') as output:json.dump(result,output,indent=2)
    print(json.dumps(result))


if __name__=='__main__':main()
