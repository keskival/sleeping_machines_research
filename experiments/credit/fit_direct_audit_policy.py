"""B1 direct audit-policy fitness from reused native forward residual labels."""
import argparse
import hashlib
import json
from pathlib import Path
import resource
import time

import numpy as np
import torch

from budgeted_audit_policy import budgeted_probabilities, forward_audit_policy_loss
from fit_native_joint_audit_risk import observe, probabilities, SharedProducerActor, native

ROOT=Path(__file__).resolve().parents[2]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tag',required=True);parser.add_argument('--seed',type=int,required=True,choices=[0,1,2])
    args=parser.parse_args();torch.set_num_threads(1);torch.set_default_dtype(torch.float64)
    torch.manual_seed(1100+args.seed);start=time.monotonic()
    gate_path=ROOT/'experiments/results/credit/aws_budgeted_audit_policy_20261011_v1.json'
    gate=json.loads(gate_path.read_text());assert gate['status']=='completed'
    parent=ROOT/'experiments/results/credit'/f'aws_native_joint_audit_risk_ladder_20261011_s{args.seed}.json'
    record=json.loads(parent.read_text());assert record['status']=='completed'
    transfer_path=ROOT/'experiments/results/credit'/f'aws_joint_credit_transfer_20261011_s{args.seed}.json'
    transfer=json.loads(transfer_path.read_text());assert transfer['status']=='completed'
    sources={'experiments/credit/fit_direct_audit_policy.py','experiments/credit/budgeted_audit_policy.py'}
    for r in [gate,record,transfer]:
        for source,digest in r['source_sha256'].items():
            assert hashlib.sha256((ROOT/source).read_bytes()).hexdigest()==digest;sources.add(source)
    weight_path=ROOT/transfer['final_weights'];assert hashlib.sha256(weight_path.read_bytes()).hexdigest()==transfer['checkpoint_sha256']
    bundle=torch.load(weight_path,weights_only=True)
    train=native.load_split('taxi','train');seqs=[s for s in train if len(s[1])>=4];assert len(seqs)>=1184
    held_contexts=[(torch.tensor(t[:4])[None],torch.tensor(m[:4])[None]) for t,m in seqs[1056:1184]]
    rows=[];held_records={}
    for depth in [2,4,8]:
        candidates=[];training=record['datasets'][str(depth)]['training_records']
        for size in [64,256,1024]:
            subset=training[:size]
            raw=torch.tensor([r['features'] for r in subset]);center=raw.mean(0)
            scale=raw.std(0,unbiased=False).clamp_min(1e-6);X=(raw-center)/scale
            c2=torch.tensor([r['cotangent_squared'] for r in subset])
            labels=torch.tensor([r['forward_target']*r['cotangent_squared'] for r in subset])
            normalizer=labels.mean().clamp_min(1e-20)
            offset=.5*torch.log(c2.clamp_min(1e-20))
            policy=torch.nn.Linear(12,1)
            with torch.no_grad():policy.weight.zero_();policy.bias.zero_()
            opt=torch.optim.Adam(policy.parameters(),lr=.01);tick=time.monotonic()
            for _ in range(300):
                opt.zero_grad();scores=offset+policy(X).squeeze(-1)
                loss=forward_audit_policy_loss(scores,labels,size*.125)/normalizer+.01*policy.weight.square().sum()
                loss.backward();opt.step()
            fit_s=time.monotonic()-tick
            assert all(torch.isfinite(p).all() for p in policy.parameters())
            candidates.append((size,policy,center,scale,fit_s,float(loss.detach())))
        # All fitting frozen; independent contexts were unused by prior allocation evaluations.
        saved=bundle[str(depth)];actor=SharedProducerActor(*saved['actor_args']).eval();actor.load_state_dict(saved['actor'])
        W=saved['estimator']['weight'];field=torch.nn.Linear(W.shape[1],W.shape[0],bias=False);field.load_state_dict(saved['estimator'])
        held=[observe(actor,field,*context,8) for context in held_contexts];held_records[str(depth)]=held
        raw=torch.tensor([r['features'] for r in held]);c2=torch.tensor([r['cotangent_squared'] for r in held])
        truth=np.array([r['exact_risk'] for r in held]);norm_only=probabilities(c2.numpy(),16.)
        oracle=probabilities(truth,16.);uniform=np.full(128,.125)
        variance=lambda p:float(np.sum(truth*(1/p-1)))
        for size,policy,center,scale,fit_s,loss in candidates:
            with torch.no_grad():
                score=.5*torch.log(c2.clamp_min(1e-20))+policy((raw-center)/scale).squeeze(-1)
                allocation=budgeted_probabilities(score,16.).numpy()
            mse_record=next(r for r in record['rows'] if r['depth']==depth and r['train_contexts']==size)
            mse_model=torch.nn.Sequential(torch.nn.Linear(12,1),torch.nn.Softplus())
            with torch.no_grad():
                mse_model[0].weight.copy_(torch.tensor(mse_record['fitted_weight']))
                mse_model[0].bias.copy_(torch.tensor(mse_record['fitted_bias']))
                mse_risk=mse_model((raw-torch.tensor(mse_record['feature_center']))/torch.tensor(mse_record['feature_scale'])).squeeze(-1)
                mse_risk=mse_risk*mse_record['target_scale']*c2
            mse_allocation=probabilities(mse_risk.numpy(),16.)
            values={name:variance(p) for name,p in [('direct',allocation),('mse_risk',mse_allocation),
                        ('cotangent_norm',norm_only),('uniform',uniform),('oracle',oracle)]}
            assert all(np.isfinite(list(values.values()))) and abs(allocation.sum()-16)<1e-8
            rows.append(dict(depth=depth,train_contexts=size,heldout_contexts=128,expected_audit_budget=16.,
                policy_parameters=13,identifiable_parameters=12,optimizer_updates=300,
                normalized_training_objective=loss,variance_objective=values,
                reduction_vs_uniform=1-values['direct']/values['uniform'],
                reduction_vs_cotangent_norm=1-values['direct']/values['cotangent_norm'],
                reduction_vs_mse=1-values['direct']/values['mse_risk'],
                probabilities=allocation.tolist(),fitted_weight=policy.weight.detach().tolist(),
                fitted_bias=policy.bias.detach().tolist(),feature_center=center.tolist(),feature_scale=scale.tolist(),
                policy_fitting_s=fit_s,reused_training_acquisition_s=mse_record['standalone_acquisition_wall_s'],
                new_heldout_acquisition_s=sum(r['acquisition_s'] for r in held),
                new_heldout_reference_s=sum(r['reverse_validation_s'] for r in held)))
    inputs={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [gate_path,parent,transfer_path,weight_path]}
    result=dict(status='completed',tag=args.tag,seed=args.seed,battle='B1 direct native audit policy',rows=rows,
        heldout_records=held_records,config=dict(sizes=[64,256,1024],heldout_indices=[1056,1184],steps=300,lr=.01,penalty=.01),
        source_sha256={s:hashlib.sha256((ROOT/s).read_bytes()).hexdigest() for s in sorted(sources)},input_sha256=inputs,
        inherited_ladder_wall_s=record['wall_s'],wall_s=time.monotonic()-start,max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        scope='Direct conditional variance fitness uses saved forward residual labels, no exact fitting labels/new training-model passes. Three sizes retained, new128privateTRAIN contexts, saved MSE policy evaluated on same contexts, identical expected12.5%budget. Positive support and budget adjoint are first-order exact. Analytic raw-gradient trace only, not sampled sparse updates, future task learning or complete resource advantage; acquisition inherited/fullactor backward charged.')
    with (ROOT/'experiments/results/credit'/f'{args.tag}.json').open('x') as output:json.dump(result,output,indent=2)
    print(json.dumps(result))


if __name__=='__main__':main()
