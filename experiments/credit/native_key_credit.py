"""R1/B1 state-sized native receiver credit with exact elapsed-time transport.

Unnormalized native keys only. Forward packets contain output-clock
responsibilities, mark errors, actual queries and key decay factors. Exact
reverse credit is a reference; learned finite-window tails are separate.
No global deep producer backward is hidden in these formulas, nor eliminated:
contracting receiver cotangents with producer sensitivities is still charged.
"""
import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'experiments/tpp'))
from recall_tpp_v4 import KeyedRaceTPP
from race_tpp_v5 import RaceTPP
from reciprocal_native_v1 import native_pair, parameters, batch


class ReceiverCuts(KeyedRaceTPP):
    def encode(self,t,m,mask):
        h,slots=RaceTPP.encode(self,t,m,mask)
        self.producer_h=h
        self.cut_h=h.detach().requires_grad_(True)
        previous=torch.cat((torch.zeros_like(h[:,:1]),self.embed(m[:,:-1])),1)
        self.producer_previous=previous
        self.cut_previous=previous.detach().requires_grad_(True)
        dt=torch.diff(t,dim=1,prepend=t[:,:1]).clamp_min(0)/self.scale
        self.written_keys=self.key_w(torch.cat((self.cut_h,self.cut_previous),-1))
        if self.qk_norm:raise ValueError('Normalized keys need their normalization Jacobian; not admitted here')
        self.key_decay=torch.exp(-self.key_log_rate.exp()*dt.unsqueeze(-1))
        onehot=F.one_hot(m,self.K).to(h.dtype)
        B,L,_=h.shape;s=h.new_zeros(B,self.K,self.dk);keys=[]
        for j in range(L):
            s=s*self.key_decay[:,j].unsqueeze(1)+onehot[:,j].unsqueeze(-1)*self.written_keys[:,j].unsqueeze(1)
            keys.append(s)
        self._keyed=torch.stack(keys,1)
        self._query=self.query_w(self.cut_h)
        return h,slots


def packets(model,t,m):
    tl,ml,n,(h,slots,clock,tau,valid)=model.loglik(t,m,torch.ones_like(m,dtype=torch.bool))
    lh,_=model.clock_terms(tau,*clock[:4])
    # Posterior output-clock cause, conditional on the observed mark and gap.
    selected=clock[4].gather(-1,m[:,1:,None,None].expand(-1,-1,model.M,1)).squeeze(-1)
    responsibilities=F.softmax(lh+selected,-1)
    onehot=F.one_hot(m[:,1:],model.K).to(h.dtype)
    mark_error=(responsibilities.unsqueeze(-1)*(clock[4].exp()-onehot.unsqueeze(-2))).sum(-2)
    return -(tl+ml)/n,dict(mark_error=mark_error.detach(),queries=model._query.detach(),
                           key_decay=model.key_decay.detach(),keyed=model._keyed.detach(),
                           marks=m.detach(),times=t.detach(),scored_targets=int(n),
                           responsibilities=responsibilities.detach())


def exact_key_cotangents(packet,dk):
    """One state vector per slot, with actual reverse elapsed-time transport.

At query j the error has already been observed. Accumulating it to earlier
writes is a delayed credit operation; no earlier prediction reads this error.
"""
    marks=packet['marks'];B,L=marks.shape;K=packet['mark_error'].shape[-1]
    state=packet['queries'].new_zeros(B,K,dk);out=[]
    for j in range(L-1,-1,-1):
        if j<L-1:
            state=state*packet['key_decay'][:,j+1].unsqueeze(1)
            state=state+packet['mark_error'][:,j].unsqueeze(-1)*packet['queries'][:,j].unsqueeze(1)/math.sqrt(dk)
        out.append(state.gather(1,marks[:,j,None,None].expand(-1,1,dk)).squeeze(1))
    return torch.stack(list(reversed(out)),1)/packet['scored_targets']


def query_cotangents(packet,dk):
    answer=packet['queries'].new_zeros(packet['queries'].shape)
    answer[:,:-1]=(packet['keyed'][:,:-1]*packet['mark_error'].unsqueeze(-1)).sum(-2)/math.sqrt(dk)
    return answer/packet['scored_targets']


def contracts():
    rows=[]
    for depth in (2,4,8):
        full,_=native_pair(depth,169)
        model=ReceiverCuts(8,8,3,depth,2,2,2,0.,1.,[-1.,0.],dk=4,local=True,prev_msg=True,qk_norm=False).double()
        model.load_state_dict(full.model.state_dict())
        t,m=batch(169,pairs=3)
        original=full(t,m)
        loss,packet=packets(model,t,m)
        assert abs(float(original.detach()-loss.detach()))<1e-12
        ak,aq,ah,ap=torch.autograd.grad(loss,(model.written_keys,model._query,model.cut_h,model.cut_previous),retain_graph=True)
        predicted_keys=exact_key_cotangents(packet,4)
        predicted_queries=query_cotangents(packet,4)
        d=model.cut_h.shape[-1]
        reconstructed_h=predicted_keys@model.key_w.weight[:,:d]+predicted_queries@model.query_w.weight
        reconstructed_previous=predicted_keys@model.key_w.weight[:,d:]
        errors=dict(key_cotangent=float((ak-predicted_keys).abs().max()),query_cotangent=float((aq-predicted_queries).abs().max()),
                    producer_state_cotangent=float((ah-reconstructed_h).abs().max()),
                    predecessor_cotangent=float((ap-reconstructed_previous).abs().max()))
        assert max(errors.values())<1e-10,errors
        assert torch.equal(predicted_keys[:,-1],torch.zeros_like(predicted_keys[:,-1]))
        rows.append(dict(depth=depth,scored_targets=packet['scored_targets'],errors=errors,
                         key_cotangent_elements_per_write=4,credit_transport_state_elements_per_stream=8*4,
                         available_key_slots=8,writes_per_forward_event=1,
                         keys_scored_per_query=8,credits_accumulated_per_query=8,
                         all_producer_parameter_work_eliminated=False,
                         scope='Exact receiver credit; subsequent deep producer contractions and candidate discovery remain charged. Completed-suffix delayed reference, not online scheduling.'))
    return rows


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--tag',required=True);a=ap.parse_args()
    out=ROOT/'experiments/results/credit'/(a.tag+'.json')
    if out.exists():raise FileExistsError(out)
    torch.set_num_threads(1);torch.set_default_dtype(torch.float64);start=time.monotonic();rows=contracts()
    ss=['experiments/credit/native_key_credit.py','experiments/credit/reciprocal_native_v1.py',
        'experiments/tpp/recall_tpp_v4.py','experiments/tpp/race_tpp_v5.py']
    result=dict(status='completed',tag=a.tag,battle='R1/B1',training=False,metrics=rows,wall_s=time.monotonic()-start,
                decision='Whether state-sized exact temporal key credit admits learned-tail and asynchronous scheduling integration',
                source_sha256={s:hashlib.sha256((ROOT/s).read_bytes()).hexdigest() for s in ss})
    tmp=out.with_suffix('.tmp');tmp.write_text(json.dumps(result,indent=2)+'\n');tmp.replace(out)
    print('RESULT',json.dumps(result),flush=True)


if __name__=='__main__':main()
