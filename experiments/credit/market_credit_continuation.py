"""B10 chronological learned-credit continuation; no sealed TEST access."""
import argparse, copy, hashlib, json, resource, time
from pathlib import Path
import numpy as np
import torch
from real_feedback_credit_metric_v2 import Credit, packet
from real_boundary_stable import model_data, control
ROOT=Path(__file__).resolve().parents[2]

def continue_stream(bundle,times,marks,start,blocks,group,arm,seed):
    model,critic,opt,cop,state,carried,cross,energy=copy.deepcopy(bundle)
    rng=np.random.default_rng(seed);scores=[];errors=[];zero=[];audits=0;steps=0;begin=time.monotonic()
    before={k:v.clone() for k,v in critic.state_dict().items()}
    for b in range(blocks):
        produced,leaf,end,losses,features,events=packet(model,times,marks,control.detach_state(state),start+b*group,group,b)
        with torch.set_grad_enabled(arm=='online'):
            prediction,next_credit=critic(features[None],events[None],carried.detach());prediction=prediction[0]
        trust=max(0.,min(1.,cross/max(energy,1e-12)))
        audit=arm=='bptt' or rng.random()<.25;target=None
        if audit:
            target=torch.autograd.grad(torch.stack(losses[1:]).sum()/len(losses),control.tensors(leaf),retain_graph=True);audits+=1
        flat=torch.zeros_like(prediction) if arm=='zero' else trust*prediction.detach()
        pred=[x.reshape_as(y) for x,y in zip(flat.split([x.numel() for x in control.tensors(produced)]),control.tensors(produced))]
        credit=target if arm=='bptt' else ([h+(g-h)/.25 for h,g in zip(pred,target)] if audit else pred)
        control.backward(model,produced,leaf,losses,credit)
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step()
        if audit and arm in ('online','frozen'):
            exact=torch.cat([x.detach().flatten() for x in target]);errors.append(float((flat-exact).square().mean()));zero.append(float(exact.square().mean()))
            if arm=='online':
                cross=.95*cross+.05*float((prediction.detach()*exact).mean());energy=.95*energy+.05*float(prediction.detach().square().mean())
                cop.zero_grad(set_to_none=True);(prediction-exact).square().mean().backward()
                torch.nn.utils.clip_grad_norm_(critic.parameters(),1.,error_if_nonfinite=True);cop.step();steps+=1
        carried=next_credit.detach();state=control.detach_state(end);scores.extend(float(x.detach()) for x in losses)
    if arm=='frozen':assert all(torch.equal(v,critic.state_dict()[k]) for k,v in before.items())
    result=dict(arm=arm,nll=float(np.mean(scores)),targets=len(scores),audits=audits,critic_updates=steps,wall_s=time.monotonic()-begin,block_nll=[float(np.mean(scores[i:i+group])) for i in range(0,len(scores),group)],audited_mse=errors,audited_zero_mse=zero)
    return (model,critic,opt,cop,state,carried,cross,energy),result

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--tag',required=True);ap.add_argument('--mode',choices=['smoke','fit'],required=True);ap.add_argument('--seed',type=int,default=184);a=ap.parse_args();torch.set_num_threads(1)
    group=8 if a.mode=='smoke' else 32;context=32 if a.mode=='smoke' else 128;acq=4 if a.mode=='smoke' else 64;reuse=8 if a.mode=='smoke' else 128
    out=ROOT/'experiments/results/credit'/f'{a.tag}.json';assert not out.exists();begin=time.monotonic()
    model,times,marks=model_data(a.seed,context+(acq+reuse)*group,context)
    width=sum(x.numel() for x in control.tensors(control.market_consume(model,times[:1],marks[:1])[1]));torch.manual_seed(a.seed+10000);critic=Credit(width).double()
    opt=torch.optim.AdamW(model.parameters(),lr=.0003,weight_decay=0);cop=torch.optim.AdamW(critic.parameters(),lr=.001,weight_decay=0);state=None
    with torch.no_grad():
        for k in range(context-1):_,state=control.market_consume(model,times[k:k+1],marks[k:k+1],state)
    bundle=(model,critic,opt,cop,state,torch.zeros(1,64,dtype=torch.float64),0.,0.)
    bundle,acquisition=continue_stream(bundle,times,marks,context-1,acq,group,'online',a.seed+20000)
    rows=[]
    for arm in ('online','frozen','zero','bptt'):
        _,row=continue_stream(bundle,times,marks,context-1+acq*group,reuse,group,arm,a.seed+30000);rows.append(row)
    assert rows[0]['audits']==rows[1]['audits']==rows[2]['audits']
    sources=[Path(__file__),ROOT/'experiments/credit/real_feedback_credit_metric_v2.py',ROOT/'experiments/credit/real_boundary_stable.py',ROOT/'experiments/credit/real_learned_boundary_v2.py',ROOT/'experiments/market/episode_state.py',ROOT/'experiments/tpp/race_tpp_v5.py']
    gains={r['arm']:r['nll']-rows[0]['nll'] for r in rows[1:]}
    result=dict(status='completed',battle='B10',tag=a.tag,args=vars(a),acquisition=acquisition,rows=rows,online_nll_gain=gains,whole_wall_s=time.monotonic()-begin,peak_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},scope='Single chronological Aug1 TRAIN fragment. Identical acquired actor/optimizer/critic/state; frozen critic also freezes trust calibration. Persistent credit activations still evolve. Online trains only after revealed audited credit; local gradients exact, boundary residual audits unbiased before nonlinear optimizer/clipping. All initialization/acquisition/critic/audit/optimizer/scoring CPU charged. No TEST, changing-bias attribution, sparse hardware or benchmark win.')
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(tag=a.tag,gains=gains,wall_s=result['whole_wall_s'])),flush=True)
if __name__=='__main__':main()
