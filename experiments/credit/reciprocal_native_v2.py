"""R1/B1 selective teacher gate, preserving native inference and objective.

Only audited producer parameter blocks receive exact full-gradient teacher
queries. The local actor backward, predicted block updates and candidate/clock
scoring remain dense and charged: this is not an end-to-end sparse learner.
One-step meta-credit is exact conditional on detached previous learner state;
no differentiation through all past adaptation is claimed.
"""
import argparse
import hashlib
import json
import math
import resource
import socket
import time
from pathlib import Path

import torch
from torch.func import functional_call
from torch.profiler import profile, ProfilerActivity

from reciprocal_native_v1 import (ROOT, DT, Credit, Update, actor_evidence, batch,
                                  flat, group, native_pair, parameters)


class Ledger:
    def __init__(self):
        self.counts = {}

    def add(self, name, count=1):
        self.counts[name] = self.counts.get(name, 0) + count


class CreditBank:
    """Only visited blocks advance; untouched timestamps remain unchanged."""
    def __init__(self):
        self.values = {}
        self.timestamps = {}

    def read(self, names, now, credit):
        states = {}
        for name in names:
            if name in self.values:
                dt = now - self.timestamps[name]
                states[name] = credit.transport(self.values[name], dt)
        return states

    def store(self, name, value, now):
        if name in self.timestamps and now < self.timestamps[name]:
            raise ValueError('Credit events must respect causal order')
        self.values[name] = value.detach()
        self.timestamps[name] = now


def producer(name):
    return name.startswith(('model.embed.', 'model.gap.', 'model.layers.'))


def selective_evidence(full, local, p, data, selected, ledger):
    local_loss = functional_call(local, p, data)
    ledger.add('actor_local_forward_calls')
    ledger.add('actor_local_scored_targets', int(data[1].numel()-len(data[1])))
    local_grads = torch.autograd.grad(local_loss, tuple(p.values()), allow_unused=True)
    gl = {k: torch.zeros_like(v) if g is None else g.detach()
          for (k,v),g in zip(p.items(),local_grads)}
    ledger.add('actor_local_backward_calls')
    ledger.add('actor_local_gradient_parameter_elements', sum(v.numel() for v in p.values()))
    names = [k for k in p if group(k) in selected]
    target = {}
    if names:
        loss = functional_call(full, p, data)
        ledger.add('teacher_full_forward_calls')
        ledger.add('teacher_full_scored_targets', int(data[1].numel()-len(data[1])))
        gradients = torch.autograd.grad(loss, tuple(p[k] for k in names), allow_unused=True)
        target = {k: torch.zeros_like(p[k]) if g is None else g.detach()
                  for k,g in zip(names, gradients)}
        ledger.add('teacher_selected_backward_calls')
        ledger.add('teacher_selected_parameter_elements', sum(p[k].numel() for k in names))
    packet = (float(full.model.embed(data[1]).detach().mean()),
              float(full.model.embed(data[1]).detach().std()),
              float((data[0][:,-1]-data[0][:,0]).mean()))
    return gl, target, packet


def assemble(gl, predictions, targets, selected, probabilities):
    """Unaudited blocks cannot read a teacher entry they did not request."""
    corrected = dict(gl)
    for k,h in predictions.items():
        corrected[k] = gl[k] + h
        if group(k) in selected:
            corrected[k] = corrected[k] + (targets[k]-gl[k]-h)/probabilities[group(k)]
    return corrected


def contracts():
    rows=[]
    for depth in (2,4,8):
        full,local=native_pair(depth,165);p=parameters(full)
        names=list(dict.fromkeys(group(k) for k in p if producer(k)))
        selected=set(names[::2]);ledger=Ledger()
        gl,target,packet=selective_evidence(full,local,p,batch(165),selected,ledger)
        _,gf,reference_local,_=actor_evidence(full,local,p,batch(165))
        teacher_error=max(float((target[k]-gf[k]).abs().max()) for k in target)
        local_error=float((flat(gl)-flat(reference_local)).abs().max())
        assert teacher_error<1e-11 and local_error<1e-11
        # All remaining blocks have structurally zero missing producer paths.
        certified_error=max(float((gf[k]-gl[k]).abs().max()) for k in p if not producer(k))
        assert certified_error<1e-11
        credit=Credit();bank=CreditBank()
        bank.store(names[0],torch.tensor([.1,.2],dtype=DT),0.)
        bank.store(names[-1],torch.tensor([.3,.4],dtype=DT),0.)
        first=bank.read([names[0]],1.,credit)[names[0]]
        bank.store(names[0],first,1.)
        second=bank.read([names[0]],2.,credit)[names[0]]
        expected=credit.transport(torch.tensor([.1,.2],dtype=DT),2.)
        lazy_error=float((second-expected).abs().max())
        assert lazy_error<1e-12 and bank.timestamps[names[-1]]==0.
        qp={k:v for k,v in p.items() if producer(k)}
        pred=credit.predict(qp,{k:gl[k] for k in qp},packet,bank.read(names,2.,credit),0.)
        g=assemble(gl,pred,target,selected,{name:.5 for name in names})
        assert all(torch.isfinite(v).all() for v in g.values())
        assert set(target)=={k for k in p if group(k) in selected}
        assert ledger.counts['teacher_selected_parameter_elements']<sum(v.numel() for v in p.values())
        rows.append(dict(depth=depth,selected_teacher_parity_error=teacher_error,
                         local_gradient_parity_error=local_error,zero_missing_path_certificate_error=certified_error,
                         lazy_credit_transport_error=lazy_error,total_parameters=sum(v.numel() for v in p.values()),
                         ledger=ledger.counts))
    return rows


def smoke(depth, steps):
    full,local=native_pair(depth,165);p=parameters(full)
    credit,update=Credit(),Update();bank=CreditBank();optimizer_state={}
    auxiliary=torch.optim.Adam(list(credit.parameters())+list(update.parameters()),lr=.001)
    ledger=Ledger();history=[];pending=None
    groups=list(dict.fromkeys(group(k) for k in p if producer(k)))
    probs={k:.5 for k in groups};rng=torch.Generator().manual_seed(165)
    for step in range(steps):
        now=float(step+1)
        state=bank.read(groups,now,credit)
        # Reconstruct the PREVIOUS observed credit transition so its retained
        # activations receive supervision from the next forward consequence.
        # Other previous state is detached: exact one-step conditional meta
        # credit, not full-history meta differentiation.
        if pending is not None:
            oldstate,oldresidual,oldpacket,oldtime=pending
            observed=credit.observe(oldstate,oldresidual,oldpacket,0.)
            for name,value in observed.items():
                state[name]=credit.transport(value,now-oldtime)
        selected={k for k in groups if float(torch.rand((),generator=rng))<probs[k]}
        gl,target,packet=selective_evidence(full,local,p,batch(5000+step),selected,ledger)
        qp={k:v for k,v in p.items() if producer(k)}
        predicted=credit.predict(qp,{k:gl[k] for k in qp},packet,state,0.)
        ledger.add('predicted_credit_parameter_elements',sum(v.numel() for v in qp.values()))
        ledger.add('selected_audit_blocks',len(selected));ledger.add('eligible_audit_blocks',len(groups))
        g=assemble(gl,predicted,target,selected,probs)
        changed,next_optimizer_state=update.step(p,g,optimizer_state,1.)
        future_data=batch(6000+step)
        future_loss=functional_call(full,changed,future_data)
        ledger.add('future_meta_forward_calls');ledger.add('future_meta_scored_targets',int(future_data[1].numel()-len(future_data[1])))
        calibration=sum((predicted[k]-(target[k]-gl[k])).square().sum()/probs[group(k)]
                        for k in predicted if group(k) in selected)/sum(v.numel() for v in qp.values())
        if not selected:calibration=future_loss.new_zeros(())
        objective=future_loss+calibration
        trainable=list(credit.parameters())+list(update.parameters())
        derivatives=torch.autograd.grad(objective,trainable,allow_unused=True)
        auxiliary.zero_grad()
        for parameter,gradient in zip(trainable,derivatives):
            parameter.grad=gradient
        credit_norm=math.sqrt(sum(float(v.grad.square().sum()) for v in credit.parameters() if v.grad is not None))
        update_norm=math.sqrt(sum(float(v.grad.square().sum()) for v in update.parameters() if v.grad is not None))
        assert math.isfinite(float(objective)) and credit_norm>0 and update_norm>0
        auxiliary.step()
        ledger.add('credit_optimizer_meta_backward_calls')
        ledger.add('auxiliary_optimizer_parameter_visits',sum(v.numel() for v in trainable))
        ledger.add('actor_update_parameter_visits',sum(v.numel() for v in p.values()))
        # Consume this update once, using its frozen pre-audit predictor version.
        p={k:v.detach().requires_grad_(True) for k,v in changed.items()}
        optimizer_state={k:tuple(x.detach() for x in pair) for k,pair in next_optimizer_state.items()}
        residual={k:target[k]-gl[k] for k in target}
        retained_state={k:state[k].detach() for k in state if k in {group(n) for n in residual}}
        pending=(retained_state,residual,packet,now)
        observed=credit.observe(retained_state,residual,packet,0.)
        for name,value in observed.items():bank.store(name,value,now)
        history.append(dict(step=step,future_nll=float(future_loss.detach()),calibration_mse=float(calibration.detach()),
                            selected_audit_blocks=len(selected),eligible_audit_blocks=len(groups),
                            credit_gradient_norm=credit_norm,optimizer_gradient_norm=update_norm))
        print(json.dumps(history[-1]),flush=True)
    return dict(history=history,ledger=ledger.counts,
                persistent_credit_elements=sum(x.numel() for x in bank.values.values()),
                optimizer_state_elements=sum(x.numel() for pair in optimizer_state.values() for x in pair),
                unresolved_record_elements=0 if pending is None else sum(x.numel() for x in pending[0].values())+sum(x.numel() for x in pending[1].values()),
                teacher_evaluates_only_requested_parameter_blocks=True,end_to_end_sparse=False,
                scope='Measured integration smoke with actual native actor updates; no benchmark quality or efficiency claim')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--tag',required=True)
    parser.add_argument('--mode',choices=('contract','smoke'),required=True)
    parser.add_argument('--depth',type=int,default=2);parser.add_argument('--steps',type=int,default=3)
    args=parser.parse_args();out=ROOT/'experiments/results/credit'/(args.tag+'.json')
    if out.exists():raise FileExistsError(out)
    torch.set_num_threads(1);torch.set_default_dtype(DT);torch.manual_seed(165)
    start=time.monotonic()
    with profile(activities=[ProfilerActivity.CPU],with_flops=True,profile_memory=True) as prof:
        result=contracts() if args.mode=='contract' else smoke(args.depth,args.steps)
    keys=prof.key_averages()
    paths=['experiments/credit/reciprocal_native_v2.py','experiments/credit/reciprocal_native_v1.py',
           'experiments/tpp/recall_tpp_v4.py','experiments/tpp/race_tpp_v5.py']
    record=dict(status='completed',battle='R1/B1',tag=args.tag,args=vars(args),metrics=result,
                decision='Whether selective teacher queries and persistent credit/update integration admit a measured native DEV fit',
                training=args.mode=='smoke',wall_s=time.monotonic()-start,host=socket.gethostname(),
                peak_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                profiled_flops=sum(x.flops for x in keys),profiled_operator_calls=sum(x.count for x in keys),
                profiler_scope='PyTorch supported-operation FLOPs; special functions, unsupported operations, traffic and wall overhead are not fully represented',
                source_sha256={s:hashlib.sha256((ROOT/s).read_bytes()).hexdigest() for s in paths})
    out.parent.mkdir(parents=True,exist_ok=True);tmp=out.with_suffix('.tmp');tmp.write_text(json.dumps(record,indent=2)+'\n');tmp.replace(out)
    print('RESULT',json.dumps(record),flush=True)


if __name__=='__main__':main()
