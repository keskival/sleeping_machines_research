"""B1/R1 native actor update contract using forward-taught cut transport."""
import argparse
import copy
import hashlib
import json
import resource
import sys
import time
from pathlib import Path
import numpy as np
import torch
from reciprocal_transport import DirectionalPacket
from reciprocal_transport_projection import ProjectedTransport as ReciprocalTransport

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'experiments/tpp'))
import race_tpp_v19 as native


class CutActor(native.RaceTPP):
    def encode(self,t,m,mask):
        dt=torch.diff(t,dim=1,prepend=t[:,:1]).clamp_min(0)/self.scale
        age=(t-t[:,:1])/self.scale
        x=self.embed(m)+self.gap(torch.stack([torch.log1p(dt),(dt>0).to(t.dtype),torch.log1p(age)],-1))
        self.cut_input=x;self.cut_dt=dt
        h=x.detach()
        for layer in self.layers:h=layer(h,dt)
        h.retain_grad();self.cut_output=h
        return h,self.marks(h,m,dt)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--tag',required=True);a=p.parse_args()
    gate=ROOT/'experiments/results/credit/aws_reciprocal_cut_contract_20261010T2200Z.json'
    g=json.loads(gate.read_text());assert g['status']=='completed'
    for s,h in g['source_sha256'].items():assert hashlib.sha256((ROOT/s).read_bytes()).hexdigest()==h
    torch.set_num_threads(1);torch.set_default_dtype(torch.float64);torch.manual_seed(214);start=time.monotonic()
    train=native.load_split('taxi','train');seq=next(s for s in train if len(s[1])>=4)
    K=1+max(int(m.max()) for _,m in train);pos=np.diff(seq[0][:4]);pos=pos[pos>0]
    scale=float(np.median(pos));qs=np.log(np.quantile(pos,[.1,.5,.9])).tolist()
    args=(K,8,4,1,2,3,2,0.,scale,qs)
    ref=native.RaceTPP(*args);actor=CutActor(*args);actor.load_state_dict(ref.state_dict())
    ref_opt=torch.optim.Adam(ref.parameters(),lr=.001);opt=torch.optim.Adam(actor.parameters(),lr=.001)
    t=torch.tensor(seq[0][:4])[None];m=torch.tensor(seq[1][:4])[None];mask=torch.ones_like(m,dtype=torch.bool)
    rows=[];probe_s=fit_s=0.;n=32;basis=torch.eye(n)*np.sqrt(n)
    for version in range(4):
        tick=time.monotonic();ref_opt.zero_grad();opt.zero_grad()
        tl,ml,count,_=ref.loglik(t,m,mask);ref_loss=-(tl+ml)/count;ref_loss.backward()
        tl,ml,count,_=actor.loglik(t,m,mask);loss=-(tl+ml)/count
        x=actor.cut_input.detach().flatten();dt=actor.cut_dt
        def f(z):return actor.layers[0](z.reshape(1,4,8),dt).flatten()
        transport=ReciprocalTransport(torch.zeros(n,n),context='taxi-prefix',version=version)
        probe_tick=time.monotonic()
        for v in basis:
            _,jv=torch.func.jvp(f,(x,),(v,));packet=DirectionalPacket.capture('taxi-prefix',version,v,jv)
            transport.teach_packet(packet)
        elapsed=time.monotonic()-probe_tick;probe_s+=elapsed
        loss.backward()
        assert actor.cut_input.grad_fn is not None
        # Existing downstream credit teaches the forward input producer through
        # B learned solely from current forward directional packets.
        carried=transport.predict(actor.cut_output.grad.flatten(),context='taxi-prefix',version=version)
        torch.autograd.backward(actor.cut_input,carried.reshape_as(actor.cut_input))
        grad_error=max(float((p.grad-q.grad).abs().max())/(1+float(p.grad.abs().max())) for p,q in zip(ref.parameters(),actor.parameters()))
        assert grad_error<1e-10
        ref_opt.step();opt.step()
        param_error=max(float((p-q).detach().abs().max()) for p,q in zip(ref.parameters(),actor.parameters()))
        assert param_error<1e-9
        adam_error=max(float((ref_opt.state[p][k]-opt.state[q][k]).abs().max()) for p,q in zip(ref.parameters(),actor.parameters()) for k in ('exp_avg','exp_avg_sq'))
        assert adam_error<1e-9
        fit_s+=time.monotonic()-tick-elapsed
        rows.append(dict(version=version,loss=float(loss.detach()),normalized_gradient_error=grad_error,
                         parameter_error=param_error,adam_state_error=adam_error))
    sources=['experiments/credit/reciprocal_cut_projection_contract.py','experiments/credit/reciprocal_transport.py','experiments/credit/reciprocal_transport_projection.py','experiments/tpp/race_tpp_v19.py']
    result=dict(status='completed',tag=a.tag,battle='B1/R1',stage='projected_forward_taught_actor_cut_contract',history=rows,
                native_actor_updates=4,forward_probes=128,local_teaching_updates=128,forward_teaching_s=probe_s,
                two_arm_actor_and_audit_s=fit_s,wall_s=time.monotonic()-start,
                max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                data_sha256=hashlib.sha256((ROOT/'data/easytpp/taxi/train.json').read_bytes()).hexdigest(),
                gate=dict(path=str(gate.relative_to(ROOT)),sha256=hashlib.sha256(gate.read_bytes()).hexdigest()),
                source_sha256={s:hashlib.sha256((ROOT/s).read_bytes()).hexdigest() for s in sources},
                scope='One four-event TRAIN prefix, one native temporal layer. Forward-taught dense cut transport trains embedding/gap producer; layer/head parameters retain exact local derivatives. '
                      'Each forward packet teaches all output directions in one rank-one update. All cut directions refreshed at each version. Exact reference, probe generation and teaching charged. No sparse-learning, task-quality, TEST or efficiency claim.',
                decision='Bidirectional local closure numerically validated if gradients/Adam/parameters match; next must reduce probe/storage cost and generalize across contexts before scaling.')
    out=ROOT/'experiments/results/credit'/f'{a.tag}.json'
    with out.open('x') as stream:stream.write(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
