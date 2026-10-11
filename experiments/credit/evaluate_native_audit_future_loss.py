"""B1 enumerated native one-step future-loss diagnosis for saved audit policies."""
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import resource
import time
import torch
from budgeted_audit_policy import budgeted_probabilities
from fit_native_joint_audit_risk import SharedProducerActor, native, observe, probabilities

ROOT=Path(__file__).resolve().parents[2]


def producers(actor):
    return [actor.embed.weight,actor.gap.weight,actor.gap.bias,actor.tempo.weight,actor.tempo.bias]


def loss(actor,context):
    t,m=context
    tl,ml,count,_=actor.loglik(t,m,torch.ones_like(m,dtype=torch.bool))
    return -(tl+ml)/count


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tag',required=True);parser.add_argument('--seed',type=int,required=True,choices=[0,1,2])
    args=parser.parse_args();torch.set_num_threads(1);torch.set_default_dtype(torch.float64)
    torch.manual_seed(1200+args.seed);start=time.monotonic()
    paths=[ROOT/'experiments/results/credit'/f'aws_joint_credit_transfer_20261011_s{args.seed}.json',
           ROOT/'experiments/results/credit'/f'aws_direct_audit_policy_20261011_s{args.seed}.json']
    transfer,policy_record=[json.loads(p.read_text()) for p in paths]
    sources={'experiments/credit/evaluate_native_audit_future_loss.py','experiments/credit/budgeted_audit_policy.py','experiments/credit/fit_native_joint_audit_risk.py'}
    for record in [transfer,policy_record]:
        assert record['status']=='completed'
        for source,digest in record['source_sha256'].items():
            assert hashlib.sha256((ROOT/source).read_bytes()).hexdigest()==digest;sources.add(source)
    weights=ROOT/transfer['final_weights'];assert hashlib.sha256(weights.read_bytes()).hexdigest()==transfer['checkpoint_sha256']
    bundle=torch.load(weights,weights_only=True)
    seqs=[s for s in native.load_split('taxi','train') if len(s[1])>=4]
    contexts=[(torch.tensor(t[:4])[None],torch.tensor(m[:4])[None]) for t,m in seqs[1312:1332]]
    assert len(contexts)==20
    rows=[]
    for depth in [2,4,8]:
        saved=bundle[str(depth)];actor=SharedProducerActor(*saved['actor_args']).eval();actor.load_state_dict(saved['actor'])
        W=saved['estimator']['weight'];field=torch.nn.Linear(W.shape[1],W.shape[0],bias=False);field.load_state_dict(saved['estimator'])
        exact=[];approx=[];observations=[]
        for context in contexts[:4]:
            observations.append(observe(actor,field,*context,8))
            actor.zero_grad();loss(actor,context).backward();c=actor.packet.output.grad.detach()
            params=producers(actor);exact.append(torch.cat([p.grad.flatten() for p in params]).detach())
            x,elapsed,dt,_,_=actor.produce(*context)
            predicted=field(c.flatten()).detach();px=predicted[:x.numel()].reshape_as(x);pt=predicted[x.numel():].reshape_as(dt)
            gradients=torch.autograd.grad((x*px).sum()+(elapsed*pt).sum(),params)
            approx.append(torch.cat([g.flatten() for g in gradients]).detach())
        exact=torch.stack(exact);approx=torch.stack(approx);residual=exact-approx
        fit=next(r for r in policy_record['rows'] if r['depth']==depth and r['train_contexts']==1024)
        features=torch.tensor([r['features'] for r in observations]);c2=torch.tensor([r['cotangent_squared'] for r in observations])
        score=.5*torch.log(c2.clamp_min(1e-20))+((features-torch.tensor(fit['feature_center']))/torch.tensor(fit['feature_scale']))@torch.tensor(fit['fitted_weight'])[0]+torch.tensor(fit['fitted_bias'])[0]
        allocations={'uniform':torch.full((4,),.125),'magnitude':torch.tensor(probabilities(c2.numpy(),.5)),
                     'direct':budgeted_probabilities(score,.5).detach()}
        initial=[p.detach().clone() for p in producers(actor)]
        baseline=sum(float(loss(actor,c).detach()) for c in contexts[4:])/16
        for step in [.0001,.001,.01]:
            expected={};mean_errors={};variance={}
            for name,p in allocations.items():
                value=0.;mean=torch.zeros_like(exact[0]);mass_total=0.
                for bits in itertools.product([0,1],repeat=4):
                    bits=torch.tensor(bits)
                    mass=float(torch.where(bits.bool(),p,1-p).prod())
                    gradient=(approx+(bits/p)[:,None]*residual).mean(0)
                    mass_total+=mass;mean+=mass*gradient
                    pieces=gradient.split([q.numel() for q in initial])
                    with torch.no_grad():
                        for parameter,original,piece in zip(producers(actor),initial,pieces):
                            parameter.copy_(original-step*piece.reshape_as(parameter))
                    future=sum(float(loss(actor,c).detach()) for c in contexts[4:])/16
                    assert torch.isfinite(torch.tensor(future))
                    value+=mass*future
                assert abs(mass_total-1)<1e-12
                mean_errors[name]=float((mean-exact.mean(0)).abs().max())
                assert mean_errors[name]<1e-10
                expected[name]=value
                variance[name]=float((residual.square().sum(1)*(1/p-1)).sum()/16)
            rows.append(dict(depth=depth,step=step,baseline_future_nll=baseline,expected_future_nll=expected,
                direct_gain_vs_uniform=expected['uniform']-expected['direct'],
                direct_gain_vs_magnitude=expected['magnitude']-expected['direct'],
                exact_mean_error=mean_errors,conditional_gradient_trace=variance,
                probabilities={k:v.tolist() for k,v in allocations.items()}))
        with torch.no_grad():
            for parameter,original in zip(producers(actor),initial):parameter.copy_(original)
    result=dict(status='completed',tag=args.tag,battle='B1 native future-loss audit diagnosis',seed=args.seed,rows=rows,
        source_sha256={s:hashlib.sha256((ROOT/s).read_bytes()).hexdigest() for s in sorted(sources)},
        input_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths+[weights]},
        data_sha256=hashlib.sha256((ROOT/'data/easytpp/taxi/train.json').read_bytes()).hexdigest(),
        wall_s=time.monotonic()-start,max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        scope='Evaluation-only frozen acquired fields/policy on new private TRAIN contexts1312:1332. Four-context acquisition,16future contexts, exact enumeration of16 audit outcomes per policy, three predeclared SGD steps; updates embedding/gap/tempo producers only. Native nonlinear future NLL and unbiased mean checked; full backward/reference/future-scoring work paid. Diagnostic about one-step producer updates, not full learner/depth quality, chronological streaming or optimized sparse-learning efficiency.')
    with (ROOT/'experiments/results/credit'/f'{args.tag}.json').open('x') as output:json.dump(result,output,indent=2)
    print(json.dumps(result))


if __name__=='__main__':main()
