"""B10 native credit conditioned on causal predictive-clock responses."""
import argparse,copy,hashlib,json,resource,time,math
from pathlib import Path
import numpy as np
import torch
from torch.func import functional_call
from real_feedback_credit_metric_v2 import Credit,packet
from real_boundary_stable import model_data,control
ROOT=Path(__file__).resolve().parents[2]
class FunctionalCredit(Credit):
    def __init__(self,width):
        super().__init__(3*width);self.out=torch.nn.Linear(64,2*width);torch.nn.init.zeros_(self.out.weight);torch.nn.init.zeros_(self.out.bias)
class Producer(torch.nn.Module):
    def __init__(self,model):super().__init__();self.model=model
    def forward(self,t,m,state,version):
        _,produced=control.market_consume(self.model,t,m,state,parameter_version=version)
        return torch.cat([x.flatten() for x in control.tensors(produced)])
class Window(torch.nn.Module):
    def __init__(self,model):super().__init__();self.model=model
    def forward(self,t,m,state,start,group,version):
        forecast,current=control.market_consume(self.model,t[start:start+1],m[start:start+1],state,parameter_version=version);losses=[]
        for k in range(start+1,start+group+1):
            losses.append(-control.revealed_log_probability(self.model,forecast,t[k:k+1],m[k:k+1]).mean())
            if k<start+group:forecast,current=control.market_consume(self.model,t[k:k+1],m[k:k+1],current,parameter_version=version)
        return torch.stack(losses).mean()
def adam_proposal(model,opt,gradients,multiplier=1.):
    norm=torch.stack([g.square().sum() for g in gradients]).sum().sqrt();factor=(1./(norm+1e-6)).clamp_max(1.)
    proposed={}
    for (name,p),g in zip(model.named_parameters(),gradients):
        g=g*factor;s=opt.state[p];step=int(s.get('step',0))+1
        m=.9*s.get('exp_avg',torch.zeros_like(p)).detach()+.1*g
        v=.999*s.get('exp_avg_sq',torch.zeros_like(p)).detach()+.001*g.square()
        proposed['model.'+name]=p.detach()-.0003*multiplier*(m/(1-.9**step))/((v/(1-.999**step)).clamp_min(1e-24).sqrt()+1e-8)
    return proposed

class ForecastProducer(torch.nn.Module):
    def __init__(self,model):super().__init__();self.model=model
    def forward(self,t,m,state,version):
        forecast,produced=control.market_consume(self.model,t,m,state,parameter_version=version)
        values=torch.cat([x.flatten() for x in forecast['clocks']]);typed=values.sign()*values.abs().log1p()
        return torch.cat([x.flatten() for x in control.tensors(produced)]),typed

def stream(bundle,t,m,start,blocks,group,arm,seed,forecast_enabled=True):
    model,critic,opt,cop,state,carried=copy.deepcopy(bundle);rng=np.random.default_rng(seed);pending=None;scores=[];teachers=0;audits=0;parity=0.;begin=time.monotonic();versions=0;before=copy.deepcopy(critic.state_dict());credit_norms=[];anchor_norms=[];response_norms=[];bias_norms=[];calibration_norms=[];rate_history=[];step_deviation_history=[];forecast_response_norms=[];phase={key:0. for key in ['future_teacher','local_forward_backward','credit_response_inference','boundary_transport_proposals','optimizer_state']}
    for b in range(blocks):
        k=start+b*group;phase_begin=time.monotonic()
        if pending is not None:
            proposed,previous_state,regularizer=pending
            teacher=functional_call(Window(model),proposed,(t,m,previous_state,k,group,versions))+regularizer
            cop.zero_grad(set_to_none=True);derivatives=torch.autograd.grad(teacher,tuple(critic.parameters()),allow_unused=True)
            for p,g in zip(critic.parameters(),derivatives):p.grad=g
            torch.nn.utils.clip_grad_norm_(critic.parameters(),1.,error_if_nonfinite=True);cop.step();teachers+=1
        pending=None
        phase['future_teacher']+=time.monotonic()-phase_begin;phase_begin=time.monotonic()
        working=copy.deepcopy(model) if arm in ('online','field_only') else model
        produced,leaf,end,losses,features,events=packet(working,t,m,control.detach_state(state),k,group,versions)
        params=tuple(working.parameters());local=torch.autograd.grad(torch.stack(losses).mean(),params,retain_graph=True,allow_unused=True)
        phase['local_forward_backward']+=time.monotonic()-phase_begin;phase_begin=time.monotonic()
        if arm in ['online','frozen','field_only']:
            first=torch.autograd.grad(losses[0],control.tensors(produced),retain_graph=True,allow_unused=True)
            c0=torch.cat([torch.zeros_like(x).flatten() if c is None else c.detach().flatten() for x,c in zip(control.tensors(produced),first)])
            scale=c0.square().mean().sqrt().clamp_min(1e-6)
            local_proposed=adam_proposal(model,opt,[torch.zeros_like(p) if g is None else g.detach() for p,g in zip(params,local)])
            with torch.no_grad():
                alternative,alternative_forecast=functional_call(ForecastProducer(model),local_proposed,(t[k:k+1],m[k:k+1],control.detach_state(state),versions))
                _,factual_forecast=ForecastProducer(model)(t[k:k+1],m[k:k+1],control.detach_state(state),versions)
                forecast_delta=alternative_forecast-factual_forecast
                factual_scale=factual_forecast.square().mean().sqrt().clamp_min(1e-6)
                delta_scale=forecast_delta.square().mean().sqrt().clamp_min(1e-6)
                forecast_features=torch.cat((factual_forecast/factual_scale,forecast_delta/delta_scale,torch.stack((factual_scale.log(),delta_scale.log()))))
                assert forecast_features.numel()==162 and torch.isfinite(forecast_features).all()
                forecast_response_norms.append(float(delta_scale))
                if not forecast_enabled:forecast_features=torch.zeros_like(forecast_features)
            factual=torch.cat([x.detach().flatten() for x in control.tensors(produced)]);response=alternative-factual;response_scale=response.square().mean().sqrt().clamp_min(1e-6)
            extra=torch.cat((c0/scale,response/response_scale,forecast_features));features=torch.cat((features,extra));events=torch.cat((events[:,:-8],extra[None].expand(len(events),-1),events[:,-8:]),-1)
            with torch.set_grad_enabled(arm in ('online','field_only')):
                raw,next_credit=critic(features[None],events[None],carried.detach());calibration,prospective=(2*scale*raw[0,:-1].tanh()).chunk(2);prediction=calibration
                multiplier=(math.log(4)*raw[0,-1].tanh()).exp() if arm!='field_only' else raw.new_tensor(1.)
            bias_norms.append(float(prospective.detach().square().mean().sqrt()));calibration_norms.append(float(calibration.detach().square().mean().sqrt()))
            credit_norms.append(float(prediction.detach().square().mean().sqrt()));anchor_norms.append(float(scale));response_norms.append(float(response_scale))
        else:
            prediction=features.new_zeros(sum(x.numel() for x in control.tensors(produced)));prospective=torch.zeros_like(prediction);next_credit=carried;multiplier=features.new_tensor(1.)

        audit=arm!='zero';target=None
        if audit:phase['credit_response_inference']+=time.monotonic()-phase_begin;phase_begin=time.monotonic()
        target=torch.autograd.grad(torch.stack(losses[1:]).sum()/len(losses),control.tensors(leaf),retain_graph=True);audits+=1
        pred=[x.reshape_as(y) for x,y in zip(prediction.split([x.numel() for x in control.tensors(produced)]),control.tensors(produced))]
        if arm=='zero':pred=[torch.zeros_like(x) for x in pred]
        credit=([torch.zeros_like(x) for x in control.tensors(produced)] if arm=='zero' else [g.detach() for g in target])
        if arm in ['online','frozen','field_only']:
            bias=[x.reshape_as(y) for x,y in zip(prospective.split([x.numel() for x in control.tensors(produced)]),control.tensors(produced))];credit=[g+b for g,b in zip(credit,bias)]
        boundary=torch.autograd.grad(control.tensors(produced),params,grad_outputs=credit,retain_graph=True,create_graph=arm in ('online','field_only'),allow_unused=True)
        gradients=[(torch.zeros_like(p) if l is None else l.detach())+(torch.zeros_like(p) if g is None else g) for p,l,g in zip(params,local,boundary)]
        proposed=adam_proposal(model,opt,gradients,multiplier=multiplier)
        if arm in ['online','frozen','field_only']:
            # Same actual moment/clipping law; reference is complete exact credit.
            exact_boundary=torch.autograd.grad(control.tensors(produced),params,grad_outputs=[g.detach() for g in target],retain_graph=True,allow_unused=True)
            exact_gradients=[(torch.zeros_like(p) if l is None else l.detach())+(torch.zeros_like(p) if g is None else g.detach()) for p,l,g in zip(params,local,exact_boundary)]
            reference=adam_proposal(model,opt,exact_gradients)
            reference_norm=sum((reference['model.'+name].detach()-p.detach()).square().sum() for name,p in model.named_parameters()).clamp_min(1e-20)
            deviation=sum((proposed['model.'+name]-reference['model.'+name].detach()).square().sum() for name,p in model.named_parameters())/reference_norm
            regularizer=.05*deviation
            step_deviation_history.append(float(deviation.detach()))
        else:regularizer=features.new_zeros(())

        rate_history.append(float(multiplier.detach()))
        phase['boundary_transport_proposals']+=time.monotonic()-phase_begin;phase_begin=time.monotonic()
        for param_group in opt.param_groups:param_group['lr']=.0003*float(multiplier.detach())
        opt.zero_grad(set_to_none=True)
        for p,g in zip(model.parameters(),gradients):p.grad=g.detach().clone()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step();versions+=1
        parity=max(parity,max(float((p.detach()-proposed['model.'+name].detach()).abs().max()) for name,p in model.named_parameters()))
        assert parity<1e-10,parity
        if arm in ('online','field_only'):pending=(proposed,control.detach_state(end),regularizer)
        carried=next_credit.detach();state=control.detach_state(end);scores.extend(float(x.detach()) for x in losses)
        phase['optimizer_state']+=time.monotonic()-phase_begin
    if arm not in ('online','field_only'):assert all(torch.equal(v,critic.state_dict()[k]) for k,v in before.items())
    result=dict(phase_wall_s=phase,forecast_response_enabled=forecast_enabled,forecast_response_rms=forecast_response_norms,relative_parameter_step_deviation=step_deviation_history,learning_rate_multipliers=rate_history,prospective_credit_rms=bias_norms,calibration_credit_rms=calibration_norms,predicted_credit_rms=credit_norms,first_event_credit_rms=anchor_norms,local_update_response_rms=response_norms,arm=arm,nll=float(np.mean(scores)),targets=len(scores),audits=audits,future_teachers=teachers,adam_proposal_error=parity,wall_s=time.monotonic()-begin,block_nll=[float(np.mean(scores[i:i+group])) for i in range(0,len(scores),group)])
    return (model,critic,opt,cop,state,carried),result
