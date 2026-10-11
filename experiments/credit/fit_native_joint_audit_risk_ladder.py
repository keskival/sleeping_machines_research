"""B1 risk-acquisition data ladder; fixed estimator, shared forward labels.

Earlier small development validation enters acquisition here; a new private
TRAIN tail is reserved before fitting. All registered sizes are reported.
"""
import argparse
import hashlib
import json
from pathlib import Path
import resource
import time

import numpy as np
import torch

from fit_native_joint_audit_risk import observe, probabilities, SharedProducerActor, native

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tag',required=True);parser.add_argument('--seed',type=int,required=True,choices=[0,1,2])
    args=parser.parse_args();torch.set_num_threads(1);torch.set_default_dtype(torch.float64)
    torch.manual_seed(1020+args.seed);start=time.monotonic()
    parent=ROOT/'experiments/results/credit'/f'aws_joint_credit_transfer_20261011_s{args.seed}.json'
    record=json.loads(parent.read_text());assert record['status']=='completed'
    sources={'experiments/credit/fit_native_joint_audit_risk_ladder.py','experiments/credit/fit_native_joint_audit_risk.py'}
    for source,digest in record['source_sha256'].items():
        assert hashlib.sha256((ROOT/source).read_bytes()).hexdigest()==digest;sources.add(source)
    weights=ROOT/record['final_weights'];assert hashlib.sha256(weights.read_bytes()).hexdigest()==record['checkpoint_sha256']
    bundle=torch.load(weights,weights_only=True)
    train=native.load_split('taxi','train');seqs=[s for s in train if len(s[1])>=4];assert len(seqs)>=1312
    contexts=[(torch.tensor(t[:4])[None],torch.tensor(m[:4])[None]) for t,m in seqs]
    rows=[];datasets={}
    for depth in [2,4,8]:
        saved=bundle[str(depth)];actor=SharedProducerActor(*saved['actor_args']).eval();actor.load_state_dict(saved['actor'])
        W=saved['estimator']['weight'];field=torch.nn.Linear(W.shape[1],W.shape[0],bias=False);field.load_state_dict(saved['estimator'])
        # One acquisition shared by the three size arms; all per-context records retained.
        acquired=[observe(actor,field,*context,8) for context in contexts[8:1032]]
        candidates=[]
        for size in [64,256,1024]:
            subset=acquired[:size];raw=torch.tensor([r['features'] for r in subset]);center=raw.mean(0)
            scale=raw.std(0,unbiased=False).clamp_min(1e-6);X=(raw-center)/scale
            target=torch.tensor([r['forward_target'] for r in subset]);normalizer=target.mean().clamp_min(1e-20)
            target=target/normalizer
            risk=torch.nn.Sequential(torch.nn.Linear(12,1),torch.nn.Softplus())
            with torch.no_grad():risk[0].weight.zero_();risk[0].bias.fill_(np.log(np.expm1(1.)))
            opt=torch.optim.Adam(risk.parameters(),lr=.01);tick=time.monotonic()
            for _ in range(300):
                opt.zero_grad();prediction=risk(X).squeeze(-1)
                loss=(prediction-target).square().mean()+.01*risk[0].weight.square().sum()
                loss.backward();opt.step()
            fit_s=time.monotonic()-tick
            assert all(torch.isfinite(p).all() for p in risk.parameters())
            candidates.append((size,risk,center,scale,normalizer,fit_s,float(loss.detach())))
        # No held-out label, exact risk or feature participates in fitting/selection.
        held=[observe(actor,field,*context,8) for context in contexts[1184:1312]]
        norm2=np.array([r['cotangent_squared'] for r in held]);truth=np.array([r['exact_risk'] for r in held])
        oracle=probabilities(truth,16.);uniform=np.full(128,.125)
        variance=lambda p:float(np.sum(truth*(1/p-1)))
        for size,risk,center,scale,normalizer,fit_s,loss in candidates:
            with torch.no_grad():normalized=risk((torch.tensor([r['features'] for r in held])-center)/scale).squeeze(-1)*normalizer
            estimate=normalized.numpy()*norm2;norm_only=float(normalizer)*norm2
            allocation=probabilities(estimate,16.);prior=probabilities(norm_only,16.)
            values={name:variance(p) for name,p in [('learned',allocation),('cotangent_norm',prior),('uniform',uniform),('oracle',oracle)]}
            assert all(np.isfinite(list(values.values()))) and abs(allocation.sum()-16)<1e-8
            rows.append(dict(depth=depth,train_contexts=size,heldout_contexts=128,queries_per_context=8,
                             unique_training_task_targets=3*size,unique_validation_task_targets=384,
                             risk_parameters=13,optimizer_updates=300,fit_loss=loss,budget_expected_audits=16.,
                             probability_floor=.01,variance_objective=values,
                             reduction_vs_uniform=1-values['learned']/values['uniform'],
                             reduction_vs_cotangent_norm=1-values['learned']/values['cotangent_norm'],
                             heldout_risk_mse=float(np.mean((estimate-truth)**2)),
                             learned_probabilities=allocation.tolist(),heldout_predicted_risks=estimate.tolist(),
                             fitted_weight=risk[0].weight.detach().tolist(),fitted_bias=risk[0].bias.detach().tolist(),
                             feature_center=center.tolist(),feature_scale=scale.tolist(),target_scale=float(normalizer),
                             standalone_acquisition_wall_s=sum(r['acquisition_s'] for r in acquired[:size]),
                             standalone_reverse_reference_s=sum(r['reverse_validation_s'] for r in acquired[:size]),
                             risk_fitting_s=fit_s))
        datasets[str(depth)]=dict(training_records=acquired,heldout_records=held,
                                  shared_acquisition_s=sum(r['acquisition_s'] for r in acquired+held),
                                  shared_reverse_reference_s=sum(r['reverse_validation_s'] for r in acquired+held))
    result=dict(status='completed',tag=args.tag,seed=args.seed,battle='B1 native audit-risk data scaling',rows=rows,
                datasets=datasets,config=dict(train_indices=[8,1032],heldout_indices=[1184,1312],
                    ladder=[64,256,1024],queries=8,risk_features=12,steps=300,lr=.01,penalty=.01),
                source_sha256={s:hashlib.sha256((ROOT/s).read_bytes()).hexdigest() for s in sorted(sources)},
                input_sha256={str(parent.relative_to(ROOT)):hashlib.sha256(parent.read_bytes()).hexdigest(),
                              str(weights.relative_to(ROOT)):record['checkpoint_sha256']},
                data_sha256=hashlib.sha256((ROOT/'data/easytpp/taxi/train.json').read_bytes()).hexdigest(),
                wall_s=time.monotonic()-start,max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                scope='Fixed13parameter risk estimator/actor/carried field;64/256/1024 acquisitioncontexts share forward labels,128fresh private TRAIN tail contexts. Earlier exploratory validation enters acquisition explicitly; new endpoint registered. No size selection/test tuning, all rows retained. Exact residuals only validate allocation, fullactor backward and earlier/repeated acquisition charged. Analytic conditional raw-gradient trace at equal12.5%expected audits, not actual sparseupdates/future task utility/fulllearning or Transformer efficiency.')
    with (ROOT/'experiments/results/credit'/f'{args.tag}.json').open('x') as output:json.dump(result,output,indent=2)
    print(json.dumps(result))


if __name__=='__main__':main()
