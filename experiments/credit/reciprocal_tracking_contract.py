"""B1/R1 moving-native-parameter transport diagnostic on a Taxi TRAIN prefix."""
import argparse
import hashlib
import json
import resource
import sys
import time
from pathlib import Path
import numpy as np
import torch
from reciprocal_transport import DirectionalPacket, ReciprocalTransport

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'experiments/tpp'))
import race_tpp_v19 as native


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--tag',required=True);a=p.parse_args()
    gate=ROOT/'experiments/results/credit/aws_reciprocal_transport_contract_20261010T2152Z.json'
    g=json.loads(gate.read_text());assert g['status']=='completed'
    for s,h in g['source_sha256'].items():assert hashlib.sha256((ROOT/s).read_bytes()).hexdigest()==h
    torch.set_num_threads(1);torch.set_default_dtype(torch.float64);torch.manual_seed(213)
    rng=np.random.default_rng(213);start=time.monotonic()
    train=native.load_split('taxi','train');seq=next(s for s in train if len(s[1])>=4)
    K=1+max(int(m.max()) for _,m in train)
    t=torch.tensor(seq[0][:4])[None];m=torch.tensor(seq[1][:4])[None];mask=torch.ones_like(m,dtype=torch.bool)
    gaps=np.diff(seq[0][:4]);pos=gaps[gaps>0];assert len(pos)>0
    scale=float(np.median(pos));qs=np.log(np.quantile(pos,[.1,.5,.9])).tolist()
    model=native.RaceTPP(K,8,4,1,2,3,2,0.0,scale,qs).double()
    opt=torch.optim.Adam(model.parameters(),lr=.001)
    dt=torch.diff(t,dim=1,prepend=t[:,:1]).clamp_min(0)/scale
    age=(t-t[:,:1])/scale
    gap_features=torch.stack([torch.log1p(dt),(dt>0).to(t.dtype),torch.log1p(age)],-1)
    n=32;basis=torch.eye(n)*np.sqrt(n);context='taxi-train-first-four-events'
    def current_input():return (model.embed(m)+model.gap(gap_features)).detach().flatten()
    def layer_f(x):return model.layers[0](x.reshape(1,4,8),dt).flatten()
    x=current_input();transport=ReciprocalTransport(torch.zeros(n,n),context=context,version=0)
    # Paid dense initial teaching; never hide this warm-start cost.
    tick=time.monotonic()
    for v in basis:
        _,jv=torch.func.jvp(layer_f,(x,),(v,));packet=DirectionalPacket.capture(context,0,v,jv)
        for lam in basis:transport.teach(packet,lam,step=1/(n*n))
    warm_s=time.monotonic()-tick;frozen=transport.matrix.clone();rows=[]
    probe_s=audit_s=actor_s=0.;stale_rejected=False
    for version in range(1,13):
        tick=time.monotonic();tl,ml,count,_=model.loglik(t,m,mask);loss=-(tl+ml)/count
        opt.zero_grad();loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
        assert torch.isfinite(loss) and torch.isfinite(norm);opt.step();actor_s+=time.monotonic()-tick
        x=current_input()
        # Explicitly carry an approximate prediction to a new parameter version.
        # Historical targets remain inadmissible; only new forward packets teach.
        transport=ReciprocalTransport(transport.matrix,context=context,version=version)
        directions=rng.choice(n,size=2,replace=False);tick=time.monotonic()
        for i in directions:
            v=basis[int(i)];_,jv=torch.func.jvp(layer_f,(x,),(v,))
            packet=DirectionalPacket.capture(context,version,v,jv)
            for lam in basis:transport.teach(packet,lam,step=1/(n*n))
        probe_s+=time.monotonic()-tick
        if version==1:
            try:transport.teach(DirectionalPacket.capture(context,0,basis[0],torch.zeros(n)),basis[0],step=1/(n*n))
            except ValueError:stale_rejected=True
        # Independent current-version reverse audit. Never used as teaching labels.
        tick=time.monotonic();target=torch.func.jacrev(layer_f)(x).detach().T;audit_s+=time.monotonic()-tick
        error=transport.matrix-target;old_error=frozen-target;denom=float(target.square().sum())
        refreshed=[int(i) for i in directions]
        fresh_error=float(error[refreshed].abs().max());assert fresh_error<1e-10
        rows.append(dict(version=version,train_loss=float(loss.detach()),directions=refreshed,
                         refreshed_row_max_error=fresh_error,tracked_relative_error=float(error.square().sum())/denom,
                         frozen_relative_error=float(old_error.square().sum())/denom))
    assert stale_rejected
    tracked=float(np.mean([r['tracked_relative_error'] for r in rows]));old=float(np.mean([r['frozen_relative_error'] for r in rows]))
    sources=['experiments/credit/reciprocal_tracking_contract.py','experiments/credit/reciprocal_transport.py','experiments/tpp/race_tpp_v19.py']
    result=dict(status='completed',tag=a.tag,battle='B1/R1',stage='reciprocal_native_tracking',
                history=rows,mean_tracked_relative_error=tracked,mean_frozen_relative_error=old,
                lower_mean_tracking_error=tracked<old,stale_label_rejected=stale_rejected,
                actor_updates=12,train_targets_per_update=int(count),initial_forward_probes=n,
                subsequent_forward_probes=24,local_outer_product_updates=(n+24)*n,
                full_current_reverse_audits=12,initial_teaching_s=warm_s,tracking_teaching_s=probe_s,
                reverse_audit_s=audit_s,actor_fit_s=actor_s,wall_s=time.monotonic()-start,
                max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                data_sha256=hashlib.sha256((ROOT/'data/easytpp/taxi/train.json').read_bytes()).hexdigest(),
                source_sha256={s:hashlib.sha256((ROOT/s).read_bytes()).hexdigest() for s in sources},
                gate=dict(path=str(gate.relative_to(ROOT)),sha256=hashlib.sha256(gate.read_bytes()).hexdigest()),
                scope='One four-event TRAIN prefix; 12 exact-gradient native actor updates and uniformly refreshed local transport rows. '
                      'Backward prediction does not yet train actor. Dense initial teaching and all independent audits charged; no task-quality/efficiency/TEST claim.',
                decision='Use drift/error and measured teaching cost to design contextual reciprocal transport; lower error alone cannot admit a long fit.')
    out=ROOT/'experiments/results/credit'/f'{a.tag}.json'
    with out.open('x') as stream:stream.write(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
