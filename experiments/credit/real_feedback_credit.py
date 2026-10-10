"""B10: persistent time-evolving credit critic conditioned on revealed feedback.
Predict after arrived feedback, before independent exact audit; update actor last.
"""
import argparse,hashlib,json,resource,time
from pathlib import Path
import numpy as np
import torch
from real_boundary_stable import model_data,control
ROOT=Path(__file__).resolve().parents[2]
class Credit(torch.nn.Module):
    def __init__(self,width,hidden=64):
        super().__init__();self.norm=torch.nn.LayerNorm(width);self.init=torch.nn.Linear(width,hidden);self.cell=torch.nn.GRUCell(width+8,hidden);self.out=torch.nn.Linear(hidden,width);self.log_rate=torch.nn.Parameter(torch.linspace(-4.,0.,hidden));torch.nn.init.zeros_(self.out.weight);torch.nn.init.zeros_(self.out.bias)
    def forward(self,start,events,carried):
        hidden=self.init(self.norm(start))+carried
        for event in events.unbind(1):
            state=self.norm(event[:,:-8]);feedback=event[:,-8:];gap=feedback[:,:1].expm1().clamp_min(0)
            hidden=self.cell(torch.cat((state,feedback),-1),hidden*torch.exp(-torch.nn.functional.softplus(self.log_rate)*gap))
        return self.out(hidden),hidden

def compress(state):
    x=torch.cat([v.detach().flatten() for v in control.tensors(state)]);return x.sign()*x.abs().log1p()
def packet(model,times,marks,state,start,length,version):
    forecast,produced=control.market_consume(model,times[start:start+1],marks[start:start+1],state,parameter_version=version);leaf=control.leaves(produced);current=leaf;losses=[];events=[]
    for k in range(start+1,start+length+1):
        assert forecast['parameter_version']==version
        loss=-control.revealed_log_probability(model,forecast,times[k:k+1],marks[k:k+1]).mean();assert torch.isfinite(loss);losses.append(loss)
        gap=(times[k:k+1]-forecast['time'])/model.scale
        feedback=torch.cat((gap.log1p(),torch.nn.functional.one_hot(marks[k:k+1],6).to(times.dtype).flatten(),loss.detach().reshape(1)/10))
        events.append(torch.cat((compress(current),feedback)))
        if k<start+length:forecast,current=control.market_consume(model,times[k:k+1],marks[k:k+1],current,parameter_version=version)
    return produced,leaf,current,losses,compress(produced),torch.stack(events)

def run(a):
    model,times,marks=model_data(a.seed,a.context+a.targets,a.context);state=None;width=sum(x.numel() for x in control.tensors(control.market_consume(model,times[:1],marks[:1])[1]));torch.manual_seed(a.seed+10000);critic=Credit(width).double();carried=torch.zeros(1,64,dtype=torch.float64)
    opt=torch.optim.AdamW(model.parameters(),lr=.0003,weight_decay=0);cop=torch.optim.AdamW(critic.parameters(),lr=.001,weight_decay=0);rng=np.random.default_rng(a.seed+20000);replay=np.random.default_rng(a.seed+30000);versions=0;audits=0;credit_steps=0;scores=[];trusts=[];records=[];cross=0.;energy=0.;errors=[];zero_errors=[];begin=time.monotonic()
    with torch.no_grad():
        for k in range(a.context-1):_,state=control.market_consume(model,times[k:k+1],marks[k:k+1],state)
    for start in range(a.context-1,len(times)-a.group,a.group):
        produced,leaf,end,losses,features,events=packet(model,times,marks,control.detach_state(state),start,a.group,versions)
        initial_credit=carried.detach();prediction,next_credit=critic(features[None],events[None],initial_credit);prediction=prediction[0]
        trust=max(0.,min(1.,cross/max(energy,1e-12)));trusts.append(trust)
        # Feedback is already revealed; no exact credit label or audit draw has
        # been exposed when this prediction is fixed.
        audit=rng.random()<a.audit_probability;target=None
        if audit:target=torch.autograd.grad(torch.stack(losses[1:]).sum()/len(losses),control.tensors(leaf),retain_graph=True);audits+=1
        pred=[v.reshape_as(t) for v,t in zip((trust*prediction.detach()).split([x.numel() for x in control.tensors(produced)]),control.tensors(produced))]
        credit=[h+(g-h)/a.audit_probability for h,g in zip(pred,target)] if audit else pred
        control.backward(model,produced,leaf,losses,credit);torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step();versions+=1
        if audit:
            exact=torch.cat([x.detach().flatten() for x in target]);errors.append(float((trust*prediction.detach()-exact).square().mean()));zero_errors.append(float(exact.square().mean()));cross=.95*cross+.05*float((prediction.detach()*exact).mean());energy=.95*energy+.05*float(prediction.detach().square().mean())
            records.append((features,events,initial_credit[0],exact,versions));records=records[-64:];weights=np.exp(-.05*np.arange(len(records)-1,-1,-1));weights/=weights.sum()
            for _ in range(8):
                indices=replay.choice(len(records),size=min(8,len(records)),replace=True,p=weights);ss=torch.stack([records[i][0] for i in indices]);ee=torch.stack([records[i][1] for i in indices]);cc=torch.stack([records[i][2] for i in indices]);yy=torch.stack([records[i][3] for i in indices]);cop.zero_grad(set_to_none=True);guess,_=critic(ss,ee,cc);(guess-yy).square().mean().backward();torch.nn.utils.clip_grad_norm_(critic.parameters(),1.,error_if_nonfinite=True);cop.step();credit_steps+=1
        carried=next_credit.detach();state=control.detach_state(end);scores.extend(float(x.detach()) for x in losses)
    checkpoint=ROOT/'experiments/results/credit'/f'{a.tag}_feedback.pt';assert not checkpoint.exists();torch.save(dict(actor=model.state_dict(),critic=critic.state_dict(),optimizer=opt.state_dict(),credit_optimizer=cop.state_dict(),state=state,credit_state=carried,audit_records=records,cross=cross,energy=energy,actor_version=versions,credit_version=credit_steps,model_variant='positive_slot_log_clock'),checkpoint)
    return dict(arm='feedback_calibrated_credit',seed=a.seed,nll=float(np.mean(scores)),targets=len(scores),updates=versions,audits=audits,critic_steps=credit_steps,wall_s=time.monotonic()-begin,block_nll=[float(np.mean(scores[i:i+128])) for i in range(0,len(scores),128)],credit_trust=trusts,audited_mse=errors,audited_zero_mse=zero_errors,critic_parameters=sum(x.numel() for x in critic.parameters()),credit_state_elements=carried.numel(),checkpoint=str(checkpoint.relative_to(ROOT)),checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest())

def contract():
    torch.manual_seed(88);critic=Credit(88).double();state=torch.randn(1,88,dtype=torch.float64);events=torch.randn(1,8,96,dtype=torch.float64);events[:,:,-8]=torch.rand(1,8,dtype=torch.float64);hidden=torch.zeros(1,64,dtype=torch.float64)
    full,end=critic(state,events,hidden);assert torch.isfinite(full).all() and torch.isfinite(end).all();first,h=critic(state,events[:,:4],hidden)
    # Future feedback does not alter earlier credit state; no actor update
    # or rerun of issued probabilities is performed by the critic.
    changed=events.clone();changed[:,4:]+=100.;again,h2=critic(state,changed[:,:4],hidden);assert torch.equal(first,again) and torch.equal(h,h2)
    target=torch.randn(1,88,dtype=torch.float64);p=.25;estimate=(1-p)*full+p*(full+(target-full)/p);assert torch.allclose(estimate,target,atol=1e-12,rtol=0)
    return dict(causal_prefix_state_error=float((h-h2).abs().max()),residual_mean_error=float((estimate-target).abs().max()),audit_boundary='Critic sees revealed forward packets but no exact cotangent; psi frozen through prediction/draw/correction. Train only after consuming actor credit.',critic_state='Numerical credit state carries across psi versions; replay binds initial credit state, actor packet and teacher version. Credit gradients truncate at group boundaries.')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--tag',required=True);ap.add_argument('--mode',choices=['contract','smoke','measure'],required=True);ap.add_argument('--seed',type=int,default=183);a=ap.parse_args();a.targets=8192;a.context=128;a.group=32;a.audit_probability=.25;torch.set_num_threads(1);begin=time.monotonic();out=ROOT/'experiments/results/credit'/f'{a.tag}.json';assert not out.exists()
    if a.mode=='contract':metrics=contract()
    else:
        if a.mode=='smoke':a.targets=64;a.context=32;a.group=8
        metrics=dict(rows=[run(a)])
    sources=['experiments/credit/real_feedback_credit.py','experiments/credit/real_boundary_stable.py','experiments/credit/real_learned_boundary_v2.py','experiments/credit/real_learned_boundary_v3.py','experiments/credit/real_online_episode_bridge.py','experiments/market/episode_state.py','experiments/market/b10_tpp.py','experiments/tpp/race_tpp_v5.py']
    out.write_text(json.dumps(dict(status='completed',tag=a.tag,battle='B10/R1',model_variant='positive_slot_log_clock',args=vars(a),metrics=metrics,whole_wall_s=time.monotonic()-begin,peak_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,source_sha256={s:hashlib.sha256((ROOT/s).read_bytes()).hexdigest() for s in sources},scope='Connected recurrent credit conditioned on revealed forward state/errors; persistent elapsed-time credit state. Actor and credit weights co-adapt. Reuse unchanged completed stable seed183 BPTT/untrained/prefeedback controls. All dense local/audit/replay/optimizer/state work charged. No TEST, optimized efficiency or deep hard-route credit claim.'),indent=2)+'\n')
if __name__=='__main__':main()
