"""B10 connected boundary credit taught by actual next-window Adam utility."""
import argparse,copy,hashlib,json,resource,time
from pathlib import Path
import numpy as np
import torch
from torch.func import functional_call
from real_feedback_credit_metric_v2 import Credit,packet
from real_boundary_stable import model_data,control
ROOT=Path(__file__).resolve().parents[2]
class Window(torch.nn.Module):
    def __init__(self,model):super().__init__();self.model=model
    def forward(self,t,m,state,start,group,version):
        forecast,current=control.market_consume(self.model,t[start:start+1],m[start:start+1],state,parameter_version=version);losses=[]
        for k in range(start+1,start+group+1):
            losses.append(-control.revealed_log_probability(self.model,forecast,t[k:k+1],m[k:k+1]).mean())
            if k<start+group:forecast,current=control.market_consume(self.model,t[k:k+1],m[k:k+1],current,parameter_version=version)
        return torch.stack(losses).mean()
def adam_proposal(model,opt,gradients):
    norm=torch.stack([g.square().sum() for g in gradients]).sum().sqrt();factor=(1./(norm+1e-6)).clamp_max(1.)
    proposed={}
    for (name,p),g in zip(model.named_parameters(),gradients):
        g=g*factor;s=opt.state[p];step=int(s.get('step',0))+1
        m=.9*s.get('exp_avg',torch.zeros_like(p)).detach()+.1*g
        v=.999*s.get('exp_avg_sq',torch.zeros_like(p)).detach()+.001*g.square()
        proposed['model.'+name]=p.detach()-.0003*(m/(1-.9**step))/((v/(1-.999**step)).clamp_min(1e-24).sqrt()+1e-8)
    return proposed

def replay_proposal(saved,critic):
    base,oldopt,oldstate,t,m,k,version,features,events,carried,local,target,audit=saved
    forecast,produced=control.market_consume(base,t[k:k+1],m[k:k+1],oldstate,parameter_version=version)
    prediction,_=critic(features[None],events[None],carried);prediction=prediction[0]
    pred=[x.reshape_as(y) for x,y in zip(prediction.split([x.numel() for x in control.tensors(produced)]),control.tensors(produced))]
    credit=[h+(g-h)/.25 for h,g in zip(pred,target)] if audit else pred
    params=tuple(base.parameters());boundary=torch.autograd.grad(control.tensors(produced),params,grad_outputs=credit,allow_unused=True)
    gradients=[l+(torch.zeros_like(p) if g is None else g) for p,l,g in zip(params,local,boundary)]
    return adam_proposal(base,oldopt,gradients)

def stream(bundle,t,m,start,blocks,group,arm,seed):
    model,critic,opt,cop,state,carried=copy.deepcopy(bundle);rng=np.random.default_rng(seed);pending=None;scores=[];teachers=0;audits=0;parity=0.;begin=time.monotonic();versions=0;before=copy.deepcopy(critic.state_dict());accepted=0;rejected=0;teacher_changes=[];candidate_changes=[]
    for b in range(blocks):
        k=start+b*group
        if pending is not None:
            proposed,previous_state,saved=pending
            teacher=functional_call(Window(model),proposed,(t,m,previous_state,k,group,versions))
            oldcritic=copy.deepcopy(critic.state_dict());oldcop=copy.deepcopy(cop.state_dict());oldloss=float(teacher.detach())
            cop.zero_grad(set_to_none=True);derivatives=torch.autograd.grad(teacher,tuple(critic.parameters()),allow_unused=True)
            for p,g in zip(critic.parameters(),derivatives):p.grad=g
            torch.nn.utils.clip_grad_norm_(critic.parameters(),1.,error_if_nonfinite=True);cop.step();teachers+=1
            candidate=copy.deepcopy(critic.state_dict());good=False
            for trial in range(5):
                fraction=.5**trial
                critic.load_state_dict({n:oldcritic[n]+fraction*(candidate[n]-oldcritic[n]) for n in oldcritic})
                candidate_proposed=replay_proposal(saved,critic)
                with torch.no_grad():candidate_loss=float(functional_call(Window(model),candidate_proposed,(t,m,previous_state,k,group,versions)))
                if trial==0:candidate_changes.append(candidate_loss-oldloss)
                if candidate_loss<=oldloss+1e-9:
                    accepted+=1;teacher_changes.append(candidate_loss-oldloss);good=True;break
            if not good:
                critic.load_state_dict(oldcritic);cop.load_state_dict(oldcop);rejected+=1;teacher_changes.append(0.)
        pending=None
        working=copy.deepcopy(model) if arm=='online' else model
        produced,leaf,end,losses,features,events=packet(working,t,m,control.detach_state(state),k,group,versions)
        with torch.set_grad_enabled(arm=='online'):
            prediction,next_credit=critic(features[None],events[None],carried.detach());prediction=prediction[0]
        audit=arm=='bptt' or rng.random()<.25;target=None
        if audit:target=torch.autograd.grad(torch.stack(losses[1:]).sum()/len(losses),control.tensors(leaf),retain_graph=True);audits+=1
        pred=[x.reshape_as(y) for x,y in zip(prediction.split([x.numel() for x in control.tensors(produced)]),control.tensors(produced))]
        if arm=='zero':pred=[torch.zeros_like(x) for x in pred]
        credit=target if arm=='bptt' else ([h+(g.detach()-h)/.25 for h,g in zip(pred,target)] if audit else pred)
        params=tuple(working.parameters());local=torch.autograd.grad(torch.stack(losses).mean(),params,retain_graph=True,allow_unused=True)
        boundary=torch.autograd.grad(control.tensors(produced),params,grad_outputs=credit,retain_graph=True,create_graph=arm=='online',allow_unused=True)
        gradients=[(torch.zeros_like(p) if l is None else l.detach())+(torch.zeros_like(p) if g is None else g) for p,l,g in zip(params,local,boundary)]
        proposed=adam_proposal(model,opt,gradients)
        if arm=='online':
            # Bind old optimizer parameters/moments together before the live step.
            saved_model,saved_opt=copy.deepcopy((model,opt))
            saved=(saved_model,saved_opt,control.detach_state(state),t,m,k,versions,features.detach(),events.detach(),carried.detach(),[torch.zeros_like(p) if l is None else l.detach().clone() for p,l in zip(params,local)],None if target is None else [g.detach().clone() for g in target],audit)
        opt.zero_grad(set_to_none=True)
        for p,g in zip(model.parameters(),gradients):p.grad=g.detach().clone()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step();versions+=1
        parity=max(parity,max(float((p.detach()-proposed['model.'+name].detach()).abs().max()) for name,p in model.named_parameters()))
        assert parity<1e-10,parity
        if arm=='online':pending=(proposed,control.detach_state(end),saved)
        carried=next_credit.detach();state=control.detach_state(end);scores.extend(float(x.detach()) for x in losses)
    if arm!='online':assert all(torch.equal(v,critic.state_dict()[k]) for k,v in before.items())
    result=dict(accepted_meta_updates=accepted,rejected_meta_updates=rejected,candidate_teacher_loss_changes=candidate_changes,accepted_teacher_loss_changes=teacher_changes,arm=arm,nll=float(np.mean(scores)),targets=len(scores),audits=audits,future_teachers=teachers,adam_proposal_error=parity,wall_s=time.monotonic()-begin,block_nll=[float(np.mean(scores[i:i+group])) for i in range(0,len(scores),group)])
    return (model,critic,opt,cop,state,carried),result

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--tag',required=True);ap.add_argument('--mode',choices=['smoke','fit'],required=True);ap.add_argument('--seed',type=int,default=187);a=ap.parse_args();torch.set_num_threads(1);group=8 if a.mode=='smoke' else 32;context=32 if a.mode=='smoke' else 128;acq=4 if a.mode=='smoke' else 64;reuse=8 if a.mode=='smoke' else 128
    out=ROOT/f'experiments/results/credit/{a.tag}.json';assert not out.exists();begin=time.monotonic();model,t,m=model_data(a.seed,context+(acq+reuse)*group,context)
    width=sum(x.numel() for x in control.tensors(control.market_consume(model,t[:1],m[:1])[1]));torch.manual_seed(a.seed+10000);critic=Credit(width).double();opt=torch.optim.AdamW(model.parameters(),lr=.0003,weight_decay=0);cop=torch.optim.Adam(critic.parameters(),lr=.001);state=None
    with torch.no_grad():
        for k in range(context-1):_,state=control.market_consume(model,t[k:k+1],m[k:k+1],state)
    bundle=(model,critic,opt,cop,state,torch.zeros(1,64,dtype=torch.float64));bundle,acquisition=stream(bundle,t,m,context-1,acq,group,'online',a.seed+20000);rows=[]
    for arm in ['online','frozen','zero','bptt']:
        _,r=stream(bundle,t,m,context-1+acq*group,reuse,group,arm,a.seed+30000);rows.append(r)
    assert rows[0]['audits']==rows[1]['audits']==rows[2]['audits']
    sources=[Path(__file__),ROOT/'experiments/credit/real_feedback_credit_metric_v2.py',ROOT/'experiments/credit/real_boundary_stable.py',ROOT/'experiments/credit/real_learned_boundary_v2.py',ROOT/'experiments/market/episode_state.py',ROOT/'experiments/tpp/race_tpp_v5.py']
    result=dict(status='completed',battle='B10',tag=a.tag,args=vars(a),acquisition=acquisition,rows=rows,online_gain={r['arm']:r['nll']-rows[0]['nll'] for r in rows[1:]},whole_wall_s=time.monotonic()-begin,peak_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},scope='Backtracking critic updates must reduce the previously issued one-step next-window objective on already revealed data; future continuation alone measures generalization. All saved-version actor/optimizer copies, candidate producer replays and up to five teacher replays charged. Connected recurrent credit vector trained through actual one-step clipped Adam and next revealed window NLL; no gradient imitation target. Actor moments and carried numerical state held fixed in the meta derivative, credit activations persist but derivatives truncate. Independent residual audits, exact local derivatives, all teacher replays/optimizers charged. Identical acquisition and Adam state in four arms. Dense v5 boundary diagnostic, no optional-route/TEST/efficiency claim.')
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(tag=a.tag,gain=result['online_gain'])),flush=True)
if __name__=='__main__':main()
