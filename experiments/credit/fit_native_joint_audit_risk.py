"""B1 native state-conditioned audit risk from forward-only residual labels.

Private TRAIN-only calibration, fixed risk model, frozen carried credit map.
Exact risk is used only to evaluate allocation after prediction is frozen.
"""
import argparse
import hashlib
import json
from pathlib import Path
import resource
import time

import numpy as np
import torch

from check_joint_forward_risk import SharedProducerActor, producer_directions, native
from joint_forward_residual_risk import joint_residual_observations

ROOT = Path(__file__).resolve().parents[2]


def probabilities(risk, budget):
    risk = np.maximum(risk, 1e-30)
    low, high = 0., float(risk.max()) * 1e8
    for _ in range(100):
        middle = (low + high) / 2
        p = np.clip(np.sqrt(risk / max(middle, 1e-300)), .01, 1.)
        if p.sum() > budget: low = middle
        else: high = middle
    return np.clip(np.sqrt(risk / high), .01, 1.)


def observe(actor, field, times, marks, queries):
    tick = time.monotonic();actor.zero_grad()
    tl, ml, count, _ = actor.loglik(times, marks, torch.ones_like(marks, dtype=torch.bool))
    loss = -(tl+ml)/count;loss.backward()
    c = actor.packet.output.grad.detach()
    x, elapsed, dt, f, log_tempo = actor.produce(times, marks)
    with torch.no_grad(): predicted = field(c.flatten())
    px, pt = predicted[:x.numel()].reshape_as(x), predicted[x.numel():].reshape_as(dt)
    c2 = max(float(c.square().sum()), 1e-20)
    cnorm = np.sqrt(c2)
    inv1 = torch.cat([p.norm[0][1].flatten() for p in actor.packet.packets])
    inv2 = torch.cat([p.norm[1][1].flatten() for p in actor.packet.packets])
    # Available before any residual label or exact producer reference is evaluated.
    features = [float(torch.log1p(dt.max())),float(torch.log1p(dt.mean())),
                float(torch.log1p(dt.std(unbiased=False))),float(x.detach().square().mean().sqrt()),
                float(x.detach().abs().max()),float(px.norm())/cnorm,float(pt.norm())/cnorm,
                float(inv1.mean()),float(inv1.std(unbiased=False)),float(inv1.max()),
                float(inv2.mean()),float(inv2.std(unbiased=False))]
    params = [actor.embed.weight,actor.gap.weight,actor.gap.bias,actor.tempo.weight,actor.tempo.bias]
    P = sum(p.numel() for p in params)
    directions = torch.randint(2,(queries,P)).to(times.dtype)*2-1
    pv,pw = producer_directions(actor,marks,dt,f,log_tempo,directions)
    response = actor.packet.apply_with_time(pv,pw,context='taxi',version=0)
    _,labels = joint_residual_observations(c,px,pt,pv,pw,response)
    acquisition_s = time.monotonic()-tick
    # This reverse instrument validates risk, and is neither a feature nor a fit target.
    tick = time.monotonic()
    gx,gt = actor.packet.transpose_with_time(c[None],context='taxi',version=0)
    projection = torch.autograd.grad((x*(gx[0]-px)).sum()+(elapsed*(gt[0]-pt)).sum(),params)
    exact = sum(float(g.square().sum()) for g in projection)
    validation_s = time.monotonic()-tick
    return dict(features=features,forward_target=float(labels.mean())/c2,
                cotangent_squared=c2,exact_risk=exact,acquisition_s=acquisition_s,
                reverse_validation_s=validation_s)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tag',required=True);parser.add_argument('--seed',type=int,required=True,choices=[0,1,2])
    args=parser.parse_args();torch.set_num_threads(1);torch.set_default_dtype(torch.float64)
    torch.manual_seed(980+args.seed);start=time.monotonic()
    parent=ROOT/'experiments/results/credit'/f'aws_joint_credit_transfer_20261011_s{args.seed}.json'
    record=json.loads(parent.read_text());assert record['status']=='completed'
    sources={'experiments/credit/fit_native_joint_audit_risk.py'}
    for source,digest in record['source_sha256'].items():
        assert hashlib.sha256((ROOT/source).read_bytes()).hexdigest()==digest;sources.add(source)
    weights=ROOT/record['final_weights'];assert hashlib.sha256(weights.read_bytes()).hexdigest()==record['checkpoint_sha256']
    bundle=torch.load(weights,weights_only=True)
    train=native.load_split('taxi','train');seqs=[s for s in train if len(s[1])>=4]
    assert len(seqs)>=128
    contexts=[(torch.tensor(t[:4])[None],torch.tensor(m[:4])[None]) for t,m in seqs]
    rows=[]
    for depth in [2,4,8]:
        saved=bundle[str(depth)];actor=SharedProducerActor(*saved['actor_args']).eval();actor.load_state_dict(saved['actor'])
        W=saved['estimator']['weight'];field=torch.nn.Linear(W.shape[1],W.shape[0],bias=False);field.load_state_dict(saved['estimator'])
        acquired=[observe(actor,field,*context,8) for context in contexts[8:72]]
        raw=torch.tensor([r['features'] for r in acquired]);center=raw.mean(0);scale=raw.std(0,unbiased=False).clamp_min(1e-6)
        X=(raw-center)/scale;targets=torch.tensor([r['forward_target'] for r in acquired])
        normalizer=targets.mean().clamp_min(1e-20);targets=targets/normalizer
        risk=torch.nn.Sequential(torch.nn.Linear(12,1),torch.nn.Softplus())
        with torch.no_grad():risk[0].weight.zero_();risk[0].bias.fill_(np.log(np.expm1(1.)))
        optimizer=torch.optim.Adam(risk.parameters(),lr=.01);tick=time.monotonic()
        for _ in range(300):
            optimizer.zero_grad();pred=risk(X).squeeze(-1)
            fit_loss=(pred-targets).square().mean()+.01*risk[0].weight.square().sum()
            fit_loss.backward();optimizer.step()
        fitting_s=time.monotonic()-tick
        # Freeze learning before the32new contexts; labels are recorded but not fitted.
        held=[observe(actor,field,*context,8) for context in contexts[96:128]]
        with torch.no_grad(): normalized=risk((torch.tensor([r['features'] for r in held])-center)/scale).squeeze(-1)*normalizer
        norm2=np.array([r['cotangent_squared'] for r in held]);truth=np.array([r['exact_risk'] for r in held])
        estimate=normalized.numpy()*norm2
        norm_only=float(normalizer)*norm2
        allocation=probabilities(estimate,4.);prior=probabilities(norm_only,4.);oracle=probabilities(truth,4.)
        uniform=np.full(32,.125)
        variance=lambda p:float(np.sum(truth*(1/p-1)))
        values={name:variance(p) for name,p in [('learned',allocation),('cotangent_norm',prior),('uniform',uniform),('oracle',oracle)]}
        assert all(np.isfinite(list(values.values()))) and abs(allocation.sum()-4)<1e-8
        rows.append(dict(depth=depth,train_contexts=64,heldout_contexts=32,queries_per_context=8,
                         risk_parameters=13,optimizer_updates=300,fit_loss=float(fit_loss.detach()),
                         budget_expected_audits=4.,probability_floor=.01,variance_objective=values,
                         reduction_vs_uniform=1-values['learned']/values['uniform'],
                         reduction_vs_cotangent_norm=1-values['learned']/values['cotangent_norm'],
                         heldout_risk_mse=float(np.mean((estimate-truth)**2)),
                         cotangent_norm_risk_mse=float(np.mean((norm_only-truth)**2)),
                         learned_probabilities=allocation.tolist(),oracle_probabilities=oracle.tolist(),
                         heldout_true_risks=truth.tolist(),heldout_predicted_risks=estimate.tolist(),
                         fitted_weight=risk[0].weight.detach().tolist(),fitted_bias=risk[0].bias.detach().tolist(),
                         feature_center=center.tolist(),feature_scale=scale.tolist(),target_scale=float(normalizer),
                         acquisition_wall_s=sum(r['acquisition_s'] for r in acquired+held),
                         reverse_validation_s=sum(r['reverse_validation_s'] for r in acquired+held),risk_fitting_s=fitting_s))
    result=dict(status='completed',tag=args.tag,seed=args.seed,battle='B1 native forward-taught audit risk',rows=rows,
                config=dict(train_indices=[8,72],heldout_indices=[96,128],queries=8,risk_features=12,steps=300,lr=.01,penalty=.01),
                source_sha256={s:hashlib.sha256((ROOT/s).read_bytes()).hexdigest() for s in sorted(sources)},
                input_sha256={str(parent.relative_to(ROOT)):hashlib.sha256(parent.read_bytes()).hexdigest(),
                              str(weights.relative_to(ROOT)):record['checkpoint_sha256']},
                data_sha256=hashlib.sha256((ROOT/'data/easytpp/taxi/train.json').read_bytes()).hexdigest(),
                wall_s=time.monotonic()-start,max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                scope='First native state-conditioned audit-risk development fit, frozen actors/carried credit fields, TRAIN-only4event contexts.8forward producer probes supply labels; exact residual references only evaluate allocation. Full actor backward obtains loss cotangent in this fixture and is charged. Variance is the analytic conditional raw-gradient trace at equal expected audit budget, not sampled audits, actual optimizer updates, future task utility or whole-learning savings. No TEST/Transformer claim; first fixed-recipe result is about this estimator.')
    with (ROOT/'experiments/results/credit'/f'{args.tag}.json').open('x') as output:json.dump(result,output,indent=2)
    print(json.dumps(result))


if __name__=='__main__':main()
