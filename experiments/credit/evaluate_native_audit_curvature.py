"""B1 explain native audit future-loss differences with directional curvature."""
import argparse
import hashlib
import json
from pathlib import Path
import resource
import time
import torch
from evaluate_native_audit_future_loss import SharedProducerActor, native, loss, producers

ROOT=Path(__file__).resolve().parents[2]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tag',required=True)
    args=parser.parse_args();torch.set_num_threads(1);torch.set_default_dtype(torch.float64)
    start=time.monotonic()
    paths=[ROOT/'experiments/results/credit/aws_joint_credit_transfer_20261011_s0.json',
           ROOT/'experiments/results/credit/aws_native_audit_future_loss_20261011_s0.json']
    transfer,parent=[json.loads(p.read_text()) for p in paths]
    sources={'experiments/credit/evaluate_native_audit_curvature.py'}
    for record in [transfer,parent]:
        assert record['status']=='completed'
        for source,digest in record['source_sha256'].items():
            assert hashlib.sha256((ROOT/source).read_bytes()).hexdigest()==digest;sources.add(source)
    weights=ROOT/transfer['final_weights'];bundle=torch.load(weights,weights_only=True)
    assert hashlib.sha256(weights.read_bytes()).hexdigest()==transfer['checkpoint_sha256']
    seqs=[s for s in native.load_split('taxi','train') if len(s[1])>=4]
    contexts=[(torch.tensor(t[:4])[None],torch.tensor(m[:4])[None]) for t,m in seqs[1312:1332]]
    rows=[]
    for depth in [2,4,8]:
        saved=bundle[str(depth)];actor=SharedProducerActor(*saved['actor_args']).eval();actor.load_state_dict(saved['actor'])
        W=saved['estimator']['weight'];field=torch.nn.Linear(W.shape[1],W.shape[0],bias=False);field.load_state_dict(saved['estimator'])
        residuals=[]
        for context in contexts[:4]:
            actor.zero_grad();loss(actor,context).backward();c=actor.packet.output.grad.detach()
            exact=torch.cat([p.grad.flatten() for p in producers(actor)]).detach()
            x,elapsed,dt,_,_=actor.produce(*context);predicted=field(c.flatten()).detach()
            gradients=torch.autograd.grad((x*predicted[:x.numel()].reshape_as(x)).sum()+
                                         (elapsed*predicted[x.numel():].reshape_as(dt)).sum(),producers(actor))
            residuals.append(exact-torch.cat([g.flatten() for g in gradients]).detach())
        initial=[p.detach().clone() for p in producers(actor)]
        def evaluate(direction,delta):
            pieces=direction.split([p.numel() for p in initial])
            with torch.no_grad():
                for p,original,piece in zip(producers(actor),initial,pieces):p.copy_(original+delta*piece.reshape_as(p))
            return sum(float(loss(actor,c).detach()) for c in contexts[4:])/16
        zero=torch.zeros_like(residuals[0]);base=evaluate(zero,0.)
        for epsilon in [.01,.001,.0001]:
            curvature=[]
            for residual in residuals:
                norm=float(residual.norm());direction=residual/max(norm,1e-30)
                second=(evaluate(direction,epsilon)+evaluate(direction,-epsilon)-2*base)/epsilon**2
                curvature.append(second*norm**2)
            for original in [r for r in parent['rows'] if r['depth']==depth]:
                ps=original['probabilities'];q=torch.tensor(curvature)
                predicted={name:float(.5*original['step']**2*(q*(1/torch.tensor(ps[name])-1/torch.tensor(ps['direct']))).sum()/16)
                           for name in ['uniform','magnitude']}
                rows.append(dict(depth=depth,epsilon=epsilon,step=original['step'],directional_curvatures=curvature,
                    predicted_gain=predicted,observed_gain=dict(uniform=original['direct_gain_vs_uniform'],
                    magnitude=original['direct_gain_vs_magnitude'])))
        evaluate(zero,0.)
    result=dict(status='completed',tag=args.tag,battle='B1 future metric diagnosis',rows=rows,
        source_sha256={s:hashlib.sha256((ROOT/s).read_bytes()).hexdigest() for s in sorted(sources)},
        input_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths+[weights]},
        wall_s=time.monotonic()-start,max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        scope='Paid central finite-difference future-NLL curvature along actual native correction directions. Same seed0 contexts reused to explain completed one-step diagnostic; no fit, independent confirmation, PSD assertion or efficient curvature implementation claim.')
    with (ROOT/'experiments/results/credit'/f'{args.tag}.json').open('x') as output:json.dump(result,output,indent=2)
    print(json.dumps(result))


if __name__=='__main__':main()
