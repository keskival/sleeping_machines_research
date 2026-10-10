"""B10: causal learned state credit with pre-draw residual audits.
Local factual derivatives stay exact; boundary credit is predicted/audited.
"""
import argparse, hashlib, json, resource, sys, time
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'experiments/credit'))
from real_online_episode_bridge import market, market_consume, revealed_log_probability, detach_state

def tensors(state): return state['layers']+[state['slots']]
def leaves(state):
    out=dict(state);out['layers']=[x.detach().requires_grad_() for x in state['layers']];out['slots']=state['slots'].detach().requires_grad_();return out

def packet(model,times,marks,state,start,length,version,hook=None):
    issued,produced=market_consume(model,times[start:start+1],marks[start:start+1],state,parameter_version=version)
    if hook is not None:hook(produced)
    detached=leaves(produced);current=detached;losses=[]
    for k in range(start+1,start+length+1):
        assert issued['parameter_version']==version
        losses.append(-revealed_log_probability(model,issued,times[k:k+1],marks[k:k+1]).mean())
        if k<start+length:issued,current=market_consume(model,times[k:k+1],marks[k:k+1],current,parameter_version=version)
    return produced,detached,current,losses

def gradients(model):return torch.cat([p.grad.flatten() if p.grad is not None else torch.zeros_like(p).flatten() for p in model.parameters()])
def backward(model,produced,detached,losses,credit):
    model.zero_grad(set_to_none=True);torch.stack(losses).mean().backward(retain_graph=True)
    torch.autograd.backward(tensors(produced),credit)

def model_data(seed,n,context):
    raw=np.load(market.NPZ/f'{market.TRAIN[0]}.npz');times=torch.tensor((raw['t'][:n]-raw['t'][0])*1e-6,dtype=torch.float64);marks=torch.tensor(raw['mark'][:n].astype(np.int64));gaps=torch.diff(times[:context]);torch.manual_seed(seed)
    model=market.Race(gaps[gaps>0].numpy(),16,4).m.double().eval();return model,times,marks

def contract():
    model,times,marks=model_data(181,80,32);state=None
    with torch.no_grad():
        for k in range(31):_,state=market_consume(model,times[k:k+1],marks[k:k+1],state)
    state=detach_state(state);produced,leaf,end,losses=packet(model,times,marks,state,31,8,0)
    target=torch.autograd.grad(torch.stack(losses[1:]).sum()/len(losses),tensors(leaf),retain_graph=True)
    backward(model,produced,leaf,losses,target);split=gradients(model).clone()
    model.zero_grad(set_to_none=True);current=state;whole=[]
    for k in range(31,39):
        issued,current=market_consume(model,times[k:k+1],marks[k:k+1],current)
        whole.append(-revealed_log_probability(model,issued,times[k+1:k+2],marks[k+1:k+2]).mean())
    torch.stack(whole).mean().backward();error=float((split-gradients(model)).abs().max());assert error<1e-8,error
    # All-or-none group audit: exhaustive outcomes recover the exact cotangent.
    pred=[torch.ones_like(x)*.01 for x in target];p=.25
    expected=[(1-p)*h+p*(h+(g-h)/p) for h,g in zip(pred,target)]
    residual=max(float((a-b).abs().max()) for a,b in zip(expected,target));assert residual<1e-12
    return dict(split_full_bptt_gradient_error=error,exhaustive_residual_mean_error=residual,version_policy='No actor update until all losses and boundary VJP are consumed; next packet uses new weights and detached numerical state.')

def run(a,arm,seed):
    model,times,marks=model_data(seed,a.context+a.targets,a.context);state=None
    width=sum(x.numel() for x in tensors(market_consume(model,times[:1],marks[:1])[1]))
    torch.manual_seed(seed+10000);critic=torch.nn.Sequential(torch.nn.LayerNorm(width),torch.nn.Linear(width,128),torch.nn.Tanh(),torch.nn.Linear(128,width)).double()
    torch.nn.init.zeros_(critic[-1].weight);torch.nn.init.zeros_(critic[-1].bias)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.0003,weight_decay=0);cop=torch.optim.AdamW(critic.parameters(),lr=.001,weight_decay=0)
    rng=np.random.default_rng(seed+20000);scores=[];audit_errors=[];audits=0;versions=0;begin=time.monotonic();critic_steps=0;cross=0.;energy=0.;calibration=[]
    with torch.no_grad():
        for k in range(a.context-1):_,state=market_consume(model,times[k:k+1],marks[k:k+1],state)
    for start in range(a.context-1,len(times)-a.group,a.group):
        decision={}
        def issue_credit(produced):
            features=torch.cat([x.detach().flatten() for x in tensors(produced)])
            decision['prediction']=critic(features)
            decision['audit']=arm=='bptt' or (arm in ('learned_audit','calibrated_audit','untrained_audit') and rng.random()<a.audit_probability)
        produced,leaf,end,losses=packet(model,times,marks,detach_state(state),start,a.group,versions,issue_credit)
        prediction=decision['prediction'];audit=decision['audit']
        trust=max(0.,min(1.,cross/max(energy,1e-12))) if arm=='calibrated_audit' else 1.
        calibration.append(trust)
        target=None
        if audit:
            target=torch.autograd.grad(torch.stack(losses[1:]).sum()/len(losses),tensors(leaf),retain_graph=True);audits+=1
        chunks=list((trust*prediction.detach()).split([x.numel() for x in tensors(produced)]));pred=[x.reshape_as(y) for x,y in zip(chunks,tensors(produced))]
        if arm=='bptt':credit=target
        elif arm=='local':credit=[torch.zeros_like(x) for x in pred]
        else:credit=[h+(g-h)/a.audit_probability for h,g in zip(pred,target)] if audit else pred
        backward(model,produced,leaf,losses,credit);torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);optimizer.step();versions+=1
        if audit and arm in ('learned_audit','calibrated_audit','untrained_audit'):
            exact=torch.cat([x.detach().flatten() for x in target]);error=(prediction-exact).square().mean();audit_errors.append(float(error.detach()))
            if arm=='calibrated_audit':
                cross=.95*cross+.05*float((prediction.detach()*exact).mean());energy=.95*energy+.05*float(prediction.detach().square().mean())
            if arm in ('learned_audit','calibrated_audit'):cop.zero_grad(set_to_none=True);error.backward();torch.nn.utils.clip_grad_norm_(critic.parameters(),1.,error_if_nonfinite=True);cop.step();critic_steps+=1
        scores.extend(float(x.detach()) for x in losses)
        # End state precedes last scored event; consume that event next packet.
        state=detach_state(end)
    checkpoint=ROOT/'experiments/results/credit'/f'{a.tag}_{arm}.pt'
    assert not checkpoint.exists();torch.save(dict(actor=model.state_dict(),critic=critic.state_dict(),optimizer=optimizer.state_dict(),credit_optimizer=cop.state_dict(),state=detach_state(state),cross=cross,energy=energy,versions=versions),checkpoint)
    return dict(checkpoint=str(checkpoint.relative_to(ROOT)),checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),credit_trust=calibration,arm=arm,seed=seed,nll=float(np.mean(scores)),targets=len(scores),updates=versions,audits=audits,critic_steps=critic_steps,wall_s=time.monotonic()-begin,block_nll=[float(np.mean(scores[i:i+128])) for i in range(0,len(scores),128)],audited_mse=audit_errors,critic_parameters=sum(x.numel() for x in critic.parameters()),boundary_width=width)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--tag',required=True);ap.add_argument('--mode',choices=['contract','smoke','measure'],required=True);ap.add_argument('--targets',type=int,default=8192);ap.add_argument('--context',type=int,default=128);ap.add_argument('--group',type=int,default=32);ap.add_argument('--audit-probability',type=float,default=.25);ap.add_argument('--seed',type=int,default=181);a=ap.parse_args();assert 0<a.audit_probability<=1
    torch.set_num_threads(1);out=ROOT/'experiments/results/credit'/f'{a.tag}.json';assert not out.exists();begin=time.monotonic()
    if a.mode=='contract':metrics=contract()
    else:
        if a.mode=='smoke':a.targets=64;a.context=32;a.group=8
        metrics=dict(rows=[run(a,arm,a.seed) for arm in ('bptt','local','untrained_audit','learned_audit','calibrated_audit')])
    out.write_text(json.dumps(dict(status='completed',battle='B10/R1',tag=a.tag,args=vars(a),metrics=metrics,whole_wall_s=time.monotonic()-begin,peak_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),ROOT/'experiments/credit/real_online_episode_bridge.py',ROOT/'experiments/market/episode_state.py',ROOT/'experiments/market/b10_tpp.py',ROOT/'experiments/tpp/race_tpp_v5.py']},scope='Real TRAIN prefix; causal online actor and credit updates. Learned boundary cotangents extend local credit, not full backward replacement. Audits unbiased before clipping/Adam; nonlinear optimizer updates are not unbiased. Exact local derivatives, predictor, audit, optimizer and setup charged in wall; no efficiency or benchmark claim.'),indent=2)+'\n')
if __name__=='__main__':main()
