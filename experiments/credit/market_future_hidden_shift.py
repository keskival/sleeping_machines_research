"""B10 future-loss intervention under hidden conditional reversal; synthetic diagnostic."""
import argparse, copy, hashlib, json, resource, time
from pathlib import Path
import numpy as np
import torch
from real_feedback_credit_metric_v2 import Credit, packet
from real_boundary_stable import model_data, control
ROOT=Path(__file__).resolve().parents[2]

def continue_stream(bundle,times,marks,start,blocks,group,arm,seed,train_base=False):
    model,critic,opt,cop,state,carried,cross,energy,policy,popt=copy.deepcopy(bundle)
    rng=np.random.default_rng(seed);scores=[];errors=[];zero=[];audits=0;steps=0;begin=time.monotonic()
    before={k:v.clone() for k,v in critic.state_dict().items()};policy_before=copy.deepcopy(policy.state_dict());pending=None;policy_updates=0;interventions=[];contrasts=[]
    for b in range(blocks):
        produced,leaf,end,losses,features,events=packet(model,times,marks,control.detach_state(state),start+b*group,group,b)
        with torch.set_grad_enabled(train_base):
            prediction,next_credit=critic(features[None],events[None],carried.detach());prediction=prediction[0]
        if pending is not None:
            reference_model,reference_state,logprob=pending
            with torch.no_grad():
                _,_,_,reference_losses,_,_=packet(reference_model,times,marks,reference_state,start+b*group,group,b)
            contrast=float(torch.stack(losses).mean().detach()-torch.stack(reference_losses).mean())
            popt.zero_grad(set_to_none=True);(contrast*logprob).backward();torch.nn.utils.clip_grad_norm_(policy.parameters(),1.,error_if_nonfinite=True);popt.step();policy_updates+=1;contrasts.append(contrast)
        pending=None
        trust=max(0.,min(1.,cross/max(energy,1e-12)))
        with torch.set_grad_enabled(arm=='online'):
            mean=policy(events.detach().mean(0)).squeeze();distribution=torch.distributions.Normal(mean,mean.new_tensor(.5));sample=distribution.sample();logprob=distribution.log_prob(sample);multiplier=2*sample.sigmoid()
        interventions.append(float(multiplier.detach()))
        reference_model=copy.deepcopy(model) if arm=='online' else None

        audit=arm=='bptt' or rng.random()<.25;target=None
        if audit:
            target=torch.autograd.grad(torch.stack(losses[1:]).sum()/len(losses),control.tensors(leaf),retain_graph=True);audits+=1
        flat=torch.zeros_like(prediction) if arm=='zero' else multiplier.detach()*trust*prediction.detach()
        pred=[x.reshape_as(y) for x,y in zip(flat.split([x.numel() for x in control.tensors(produced)]),control.tensors(produced))]
        credit=target if arm=='bptt' else ([h+(g-h)/.25 for h,g in zip(pred,target)] if audit else pred)
        control.backward(model,produced,leaf,losses,credit)
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step()
        if audit and arm in ('online','frozen'):
            exact=torch.cat([x.detach().flatten() for x in target]);errors.append(float((flat-exact).square().mean()));zero.append(float(exact.square().mean()))
            if train_base:
                cross=.95*cross+.05*float((prediction.detach()*exact).mean());energy=.95*energy+.05*float(prediction.detach().square().mean())
                cop.zero_grad(set_to_none=True);(prediction-exact).square().mean().backward()
                torch.nn.utils.clip_grad_norm_(critic.parameters(),1.,error_if_nonfinite=True);cop.step();steps+=1
        if arm=='online':pending=(reference_model,control.detach_state(end),logprob)
        carried=next_credit.detach();state=control.detach_state(end);scores.extend(float(x.detach()) for x in losses)
    if not train_base:assert all(torch.equal(v,critic.state_dict()[k]) for k,v in before.items())
    if arm!='online':assert all(torch.equal(v,policy.state_dict()[k]) for k,v in policy_before.items())
    result=dict(policy_updates=policy_updates,intervention_multipliers=interventions,future_loss_contrasts=contrasts,arm=arm,nll=float(np.mean(scores)),targets=len(scores),audits=audits,critic_updates=steps,wall_s=time.monotonic()-begin,block_nll=[float(np.mean(scores[i:i+group])) for i in range(0,len(scores),group)],audited_mse=errors,audited_zero_mse=zero)
    return (model,critic,opt,cop,state,carried,cross,energy,policy,popt),result

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--tag',required=True);ap.add_argument('--mode',choices=['smoke','fit'],required=True);ap.add_argument('--seed',type=int,default=184);a=ap.parse_args();torch.set_num_threads(1)
    group=8 if a.mode=='smoke' else 32;context=32 if a.mode=='smoke' else 128;acq=4 if a.mode=='smoke' else 64;reuse=8 if a.mode=='smoke' else 128
    out=ROOT/'experiments/results/credit'/f'{a.tag}.json';assert not out.exists();begin=time.monotonic()
    model,times,marks=model_data(a.seed,context+(acq+reuse)*group,context)
    generator=np.random.default_rng(a.seed+40000);side=int(generator.integers(2));synthetic=[]
    switch=context+acq*group
    for k in range(len(marks)):
        stay=.95 if k<switch else .05
        if k and generator.random()>stay:side=1-side
        synthetic.append(3*side+int(generator.integers(3)))
    marks=torch.tensor(synthetic,dtype=torch.long)
    width=sum(x.numel() for x in control.tensors(control.market_consume(model,times[:1],marks[:1])[1]));torch.manual_seed(a.seed+10000);critic=Credit(width).double()
    opt=torch.optim.AdamW(model.parameters(),lr=.0003,weight_decay=0);cop=torch.optim.AdamW(critic.parameters(),lr=.001,weight_decay=0);state=None
    with torch.no_grad():
        for k in range(context-1):_,state=control.market_consume(model,times[k:k+1],marks[k:k+1],state)
    policy=torch.nn.Sequential(torch.nn.LayerNorm(width+8),torch.nn.Linear(width+8,32),torch.nn.Tanh(),torch.nn.Linear(32,1)).double()
    torch.nn.init.zeros_(policy[-1].weight);torch.nn.init.zeros_(policy[-1].bias);popt=torch.optim.Adam(policy.parameters(),lr=.003)
    bundle=(model,critic,opt,cop,state,torch.zeros(1,64,dtype=torch.float64),0.,0.,policy,popt)
    bundle,acquisition=continue_stream(bundle,times,marks,context-1,acq,group,'online',a.seed+20000,train_base=True)
    rows=[]
    for arm in ('online','frozen','zero','bptt'):
        _,row=continue_stream(bundle,times,marks,context-1+acq*group,reuse,group,arm,a.seed+30000);rows.append(row)
    assert rows[0]['audits']==rows[1]['audits']==rows[2]['audits']
    sources=[Path(__file__),ROOT/'experiments/credit/market_future_credit_intervention.py',ROOT/'experiments/credit/market_credit_continuation.py',ROOT/'experiments/credit/real_feedback_credit_metric_v2.py',ROOT/'experiments/credit/real_boundary_stable.py',ROOT/'experiments/credit/real_learned_boundary_v2.py',ROOT/'experiments/market/episode_state.py',ROOT/'experiments/tpp/race_tpp_v5.py']
    gains={r['arm']:r['nll']-rows[0]['nll'] for r in rows[1:]}
    result=dict(status='completed',battle='B10',tag=a.tag,args=vars(a),acquisition=acquisition,rows=rows,online_nll_gain=gains,whole_wall_s=time.monotonic()-begin,peak_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},scope='Controlled conditional reversal: synthetic symmetric side transition .95 -> .05 on Aug1 TRAIN timestamps, equal population marginals and hidden regime identity. Future-loss intervention policy; not real-market benchmark evidence. Base critic and trust calibration frozen during reuse; online alone trains intervention policy from actual next-window loss minus no-update parameter replay at identical carried numerical state. Stochastic scalar multiplier changes boundary prediction before independent residual audit; actor clipping/Adam can make utility depend on it. Score-function estimator, not exact pathwise gradient. Reference clones/replays charged; no audit-variance or task win assumed. Identical acquired actor/optimizer/critic/state; frozen critic also freezes trust calibration. Persistent credit activations still evolve. Online trains only after revealed audited credit; local gradients exact, boundary residual audits unbiased before nonlinear optimizer/clipping. All initialization/acquisition/critic/audit/optimizer/scoring CPU charged. No TEST, changing-bias attribution, sparse hardware or benchmark win.')
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(tag=a.tag,gains=gains,wall_s=result['whole_wall_s'])),flush=True)
if __name__=='__main__':main()
