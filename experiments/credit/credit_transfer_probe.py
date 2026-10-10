"""R1/B1 bounded frozen-teacher credit-calibration and transfer development.

Actor snapshots are fixed. A shared credit predictor learns from exact native
forward-derived derivatives, with previous-event credit activations retained.
This parameter-coordinate probe is a bridge, not state-sized sparse learning.
No TEST, grokking verdict, actor-quality or efficiency claim.
"""
import argparse
import hashlib
import json
import resource
import time
from copy import deepcopy

import torch
from torch.utils.flop_counter import FlopCounterMode
from reciprocal_native_v1 import ROOT, DT, Credit, batch, native_pair, parameters, flat
from reciprocal_native_v4 import actor_evidence


def detach(values):
    return {k:v.detach() for k,v in values.items()}


def calibration(credit,p,gl,packet,state):
    return credit.predict(p,gl,packet,state,1.)


def assess(credit,frozen,dev_batches):
    # All assessment credit states reset: no teacher labels enter predictions.
    cases=[('same_depth',2,167,2),('long_history',2,167,4),('unseen_weights',2,168,2),
           ('unseen_depth4',4,167,2),('unseen_depth8',8,167,2)]
    rows=[];targets=0
    for name,depth,seed,pairs in cases:
        full,local=native_pair(depth,seed);p=parameters(full)
        denominator=learned_error=frozen_error=0.;cosines=[]
        for i in range(dev_batches):
            data=batch(20000+i,pairs=pairs)
            _,gf,gl,packet=actor_evidence(full,local,p,data)
            missing={k:gf[k]-gl[k] for k in p}
            with torch.no_grad():
                estimate=calibration(credit,p,gl,packet,{})
                initial=calibration(frozen,p,gl,packet,{})
                denominator+=float(flat(missing).square().sum())
                learned_error+=float((flat(estimate)-flat(missing)).square().sum())
                frozen_error+=float((flat(initial)-flat(missing)).square().sum())
                total=flat(gl)+flat(estimate);true=flat(gf)
                cosines.append(float(torch.nn.functional.cosine_similarity(total,true,dim=0)))
            targets+=data[1].numel()-len(data[1])
        rows.append(dict(case=name,depth=depth,pairs=pairs,teacher_seed=seed,
                         missing_credit_squared_norm=denominator,
                         relative_missing_credit_mse=learned_error/max(denominator,1e-30),
                         frozen_relative_mse=frozen_error/max(denominator,1e-30),
                         zero_missing_credit_relative_mse=1.,
                         reconstructed_full_gradient_cosine=sum(cosines)/len(cosines),
                         assessment_credit_state='reset; no prior teacher labels'))
    return rows,targets


def run(a):
    full,local=native_pair(2,167);p=parameters(full)
    torch.manual_seed(167);credit=Credit();frozen=deepcopy(credit)
    opt=torch.optim.AdamW(credit.parameters(),lr=.01,weight_decay=.001)
    checkpoints=[];history=[];pending=None;targets=0;evaltargets=0
    observation_steps=sorted(set([0,a.steps//4,a.steps//2,a.steps]))
    initial,_=assess(credit,frozen,a.dev_batches)
    checkpoints.append(dict(step=0,assessment=initial));evaltargets+=a.dev_batches*(4*14+30)
    for step in range(1,a.steps+1):
        _,gf,gl,packet=actor_evidence(full,local,p,batch(10000+step,pairs=2))
        missing={k:gf[k]-gl[k] for k in p}
        if pending is None:state={}
        else:
            prestate,residual,oldpacket=pending
            state=credit.observe(prestate,residual,oldpacket,0.)
        predicted=calibration(credit,p,gl,packet,state)
        error=(flat(predicted)-flat(missing)).square().sum()
        energy=flat(missing).square().sum().detach()
        # Per-example relative calibration; regularize instead of assigning
        # huge updates to nearly-zero missing gradients.
        loss=error/(energy+.01)
        opt.zero_grad();loss.backward();opt.step()
        assert torch.isfinite(loss)
        pending=(detach(state),detach(missing),packet)
        targets+=14
        history.append(dict(step=step,train_relative_mse=float(error.detach()/energy.clamp_min(1e-30))))
        if step in observation_steps:
            rows,n=assess(credit,frozen,a.dev_batches);evaltargets+=n
            checkpoints.append(dict(step=step,assessment=rows))
            print(json.dumps(checkpoints[-1]),flush=True)
    return dict(history=history,checkpoints=checkpoints,
                teacher_gradient_targets=2*(targets+evaltargets),
                original_scored_targets=targets+evaltargets,
                train_sequences=2*a.steps,credit_parameters=sum(p.numel() for p in credit.parameters()),
                fixed_forward_teacher=True,final_step_fixed=True,credit_state_gradient_horizon='one previous observed learner transition; no full-history gradient',
                scope='Frozen native initialized teachers; short bounded DEV probe. MSE transfer evidence only. All exact teacher derivative work charged; no sparse-runtime/grokking/actor-quality claim')


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--tag',required=True);ap.add_argument('--steps',type=int,default=64)
    ap.add_argument('--dev-batches',type=int,default=4);a=ap.parse_args()
    out=ROOT/'experiments/results/credit'/(a.tag+'.json')
    if out.exists():raise FileExistsError(out)
    torch.set_num_threads(1);torch.set_default_dtype(DT);start=time.monotonic()
    with FlopCounterMode(display=False) as counter:result=run(a)
    paths=['experiments/credit/credit_transfer_probe.py','experiments/credit/reciprocal_native_v4.py',
           'experiments/credit/reciprocal_native_v1.py','experiments/tpp/recall_tpp_v4.py','experiments/tpp/race_tpp_v5.py']
    record=dict(status='completed',tag=a.tag,battle='R1/B1',args=vars(a),training=True,metrics=result,
                decision='Whether learned native credit transfers to unseen sequences, histories and depths before state-cotangent integration',
                supported_flops=counter.get_total_flops(),flop_scope='Supported PyTorch operation formulas only; special/unsupported work not included in this number',
                wall_s=time.monotonic()-start,peak_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                source_sha256={s:hashlib.sha256((ROOT/s).read_bytes()).hexdigest() for s in paths})
    tmp=out.with_suffix('.tmp');tmp.write_text(json.dumps(record,indent=2)+'\n');tmp.replace(out)
    print('RESULT',json.dumps({k:v for k,v in record.items() if k!='metrics'}),flush=True)


if __name__=='__main__':main()
