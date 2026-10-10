"""R1 native deterministic-credit contract, no fitting or benchmark data.

Cuts exactly the key/query producer inputs detached by current local mode.
No latent route or alternative inference architecture is added. All native
parameter gradients and independent audit outcomes are checked at fixed weights.
"""
import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'experiments/tpp'))
sys.path.insert(0,str(ROOT/'experiments/credit'))
from recall_tpp_v4 import KeyedRaceTPP
from race_tpp_v5 import RaceTPP
from continuous_residual_credit import enumerate_audits
import torch
import torch.nn.functional as F


class ProducerCuts(KeyedRaceTPP):
    """Identical local-mode numerics; expose detached producer leaves/cotangents."""
    def encode(self,t,m,mask):
        h,slots=RaceTPP.encode(self,t,m,mask)
        self.producer_h=h
        self.cut_h=h.detach().requires_grad_(True)
        previous=torch.cat([torch.zeros_like(h[:,:1]),self.embed(m[:,:-1])],1)
        self.producer_previous=previous
        self.cut_previous=previous.detach().requires_grad_(True)
        dt=torch.diff(t,dim=1,prepend=t[:,:1]).clamp_min(0)/self.scale
        k=self.key_w(torch.cat([self.cut_h,self.cut_previous],-1))
        if self.qk_norm:k=F.normalize(k,dim=-1)
        decay=torch.exp(-self.key_log_rate.exp()*dt.unsqueeze(-1))
        onehot=F.one_hot(m,self.K).to(h.dtype)
        B,L,_=h.shape;s=h.new_zeros(B,self.K,self.dk);keyed=[]
        for i in range(L):
            s=s*decay[:,i].unsqueeze(1)+onehot[:,i].unsqueeze(-1)*k[:,i].unsqueeze(1)
            keyed.append(s)
        self._keyed=torch.stack(keyed,1)
        q=self.query_w(self.cut_h)
        self._query=F.normalize(q,dim=-1)*self.log_sharp.exp() if self.qk_norm else q
        return h,slots


def loss(model,t,m):
    tl,ml,n,_=model.loglik(t,m,torch.ones_like(m,dtype=torch.bool))
    return -(tl+ml)/n


def gradient(value,parameters):
    values=torch.autograd.grad(value,parameters,retain_graph=True,allow_unused=True)
    return torch.cat([(torch.zeros_like(p) if g is None else g).reshape(-1) for p,g in zip(parameters,values)])


def check_case(normalized):
    torch.manual_seed(163)
    args=(6,8,3,2,2,2,2,0.,1.,[-1.,0.])
    full=KeyedRaceTPP(*args,dk=4,local=False,prev_msg=True,qk_norm=normalized).eval()
    local=KeyedRaceTPP(*args,dk=4,local=True,prev_msg=True,qk_norm=normalized).eval()
    cut=ProducerCuts(*args,dk=4,local=True,prev_msg=True,qk_norm=normalized).eval()
    local.load_state_dict(full.state_dict());cut.load_state_dict(full.state_dict())
    t=torch.tensor([[0.,.2,1.1,1.3,4.,4.2]])
    m=torch.tensor([[0,3,1,4,0,3]])
    full_loss=loss(full,t,m);local_loss=loss(local,t,m);cut_loss=loss(cut,t,m)
    assert max(abs(float(full_loss-local_loss)),abs(float(full_loss-cut_loss)))<1e-12
    params=list(cut.parameters())
    target=gradient(full_loss,list(full.parameters()))
    local_gradient=gradient(local_loss,list(local.parameters()))
    direct=gradient(cut_loss,params)
    local_error=float((direct-local_gradient).abs().max())
    assert local_error<1e-11
    ah,ap=torch.autograd.grad(cut_loss,[cut.cut_h,cut.cut_previous],retain_graph=True)
    def producer_credit(h_cotangent,previous_cotangent):
        scalar=(cut.producer_h*h_cotangent.detach()).sum()+(cut.producer_previous*previous_cotangent.detach()).sum()
        return gradient(scalar,params)
    missing=producer_credit(ah,ap)
    decomposition_error=float((direct+missing-target).abs().max())
    assert decomposition_error<1e-10
    assert float(missing.abs().max())>1e-5,'This witness must expose detached deep producer credit'
    h_prediction=.13*torch.sin(cut.producer_h.detach())
    previous_prediction=.07*torch.cos(cut.producer_previous.detach())
    probabilities=[.25+.1*j for j in range(m.shape[1])]
    def audit(hat_h,hat_previous):
        baseline=direct+producer_credit(hat_h,hat_previous)
        corrections=[]
        for j in range(m.shape[1]):
            residual_h=torch.zeros_like(ah);residual_p=torch.zeros_like(ap)
            residual_h[:,j]=ah[:,j]-hat_h[:,j];residual_p[:,j]=ap[:,j]-hat_previous[:,j]
            corrections.append(producer_credit(residual_h,residual_p))
        return enumerate_audits(baseline.tolist(),[c.tolist() for c in corrections],probabilities,target.tolist())
    poor=audit(h_prediction,previous_prediction)
    better=audit(.75*ah+.25*h_prediction,.75*ap+.25*previous_prediction)
    perfect=audit(ah,ap)
    for result in (poor,better,perfect):
        assert result['maximum_mean_error']<1e-10
        assert math.isclose(result['variance_trace'],result['declared_variance_trace'],rel_tol=1e-9,abs_tol=1e-20)
    assert math.isclose(better['variance_trace'],poor['variance_trace']/16,rel_tol=1e-9,abs_tol=1e-20)
    assert perfect['variance_trace']<1e-20
    names=list(cut.named_parameters());offset=0;groups={}
    for name,p in names:
        v=missing[offset:offset+p.numel()];offset+=p.numel()
        group=name.split('.')[0]
        groups[group]=groups.get(group,0.)+float((v*v).sum())
    return dict(normalized=normalized,loss=float(full_loss),local_gradient_parity_error=local_error,
                full_gradient_decomposition_error=decomposition_error,
                missing_gradient_l2=float(missing.norm()),missing_gradient_squared_norm_by_parameter_group=groups,
                arbitrary_predictor=poor,improved_predictor=better,perfect_predictor=perfect,
                parameters=target.numel(),producer_sites=m.shape[1],factual_layers=2)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--tag',required=True);args=parser.parse_args()
    output=ROOT/'experiments/results/credit'/f'{args.tag}.json'
    if output.exists():raise FileExistsError(output)
    torch.set_num_threads(1);torch.set_default_dtype(torch.float64);start=time.monotonic()
    rows=[check_case(False),check_case(True)]
    paths=('experiments/credit/check_continuous_producer_credit.py','experiments/credit/continuous_residual_credit.py',
           'experiments/tpp/recall_tpp_v4.py','experiments/tpp/race_tpp_v5.py')
    record=dict(status='completed',battle='R1/B1',tag=args.tag,training=False,metrics=rows,
                decision='Admit or reject corrected continuous producer credit for the actual R1 local-credit gap',
                scope='Initialized native two-layer model, six-event fixed sequence, normalized/plain keys; '
                      'every parameter and all 64 independent audit outcomes. No fit or quality/optimizer claim.',
                source_sha256={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in paths},wall_s=time.monotonic()-start)
    temporary=output.with_suffix('.tmp');output.parent.mkdir(parents=True,exist_ok=True)
    temporary.write_text(json.dumps(record,indent=2)+'\n');temporary.replace(output)
    print(json.dumps(record),flush=True)


if __name__ == '__main__':main()
