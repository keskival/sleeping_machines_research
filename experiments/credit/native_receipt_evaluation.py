"""Native receipt evaluator: unused reference cannot consume factual clocks."""
import copy,time
import market_actual_utility_fit_v2 as original
from native_optimizer_receipts import descriptor
ROOT,torch,np,engine,teacher=original.ROOT,original.torch,original.np,original.engine,original.teacher

def stream(model,critic,anchor,scale,t,m,blocks,training):
    model=copy.deepcopy(model);critic=copy.deepcopy(critic);anchor=copy.deepcopy(anchor)
    opt=torch.optim.Adam(model.parameters(),lr=.0003);cop=torch.optim.Adam(critic.parameters(),lr=.0001)
    state=engine.acquired.warm(model,t,m,128);carry=anchor_carry=None;rows=[];begin=time.monotonic()
    derivative=[];parity=[];trusts=[]
    for block in range(blocks):
        k=127+16*block;old=engine.acquired.control.detach_state(state)
        end,grads,scores=engine.window(model,state,t,m,k,8,'own')
        first,last=engine.acquired.flat(old),engine.acquired.flat(end)
        features=torch.cat((first,last,last-first))[None];feedback=last.new_zeros(8)
        feedback[0]=torch.log1p((t[k+8]-t[k])/model.scale)
        feedback[1:7]=torch.nn.functional.one_hot(m[k+1:k+9],6).double().mean(0);feedback[7]=float(np.mean(scores))
        events=torch.cat((features[0],feedback))[None,None]
        critic.core.train(training);anchor.core.train(False)
        with torch.set_grad_enabled(training):output,carry=critic(features,events,carry,descriptor(model,opt,grads))
        if training:
            with torch.random.fork_rng(devices=[]),torch.no_grad():
                reference,anchor_carry=anchor(features,events,anchor_carry)
        credit=output[0]*torch.as_tensor(scale)
        # Build the differentiable proposal before actual actor/moment mutation.
        if training:
            params,_=teacher.adam_parameters(model,opt,teacher.guided(model,grads,credit))
            reference_params,_=teacher.adam_parameters(model,opt,teacher.guided(model,grads,reference[0]*torch.as_tensor(scale)))
            baseline_step=sum((reference_params['actor.'+n]-p.detach()).square().sum() for n,p in model.named_parameters()).detach().clamp_min(1e-18)
            trust=sum((params[n]-reference_params[n].detach()).square().sum() for n in params)/baseline_step
        engine.step(model,opt,teacher.guided(model,grads,credit.detach()))
        with torch.no_grad():
            refreshed=old
            for index in range(k,k+8):_,refreshed=engine.acquired.control.market_consume(model,t[index:index+1],m[index:index+1],refreshed)
            future_scores=[];state=refreshed
            for index in range(k+8,k+16):
                forecast,state=engine.acquired.control.market_consume(model,t[index:index+1],m[index:index+1],state)
                future_scores.append(float(-engine.acquired.control.revealed_log_probability(model,forecast,t[index+1:index+2],m[index+1:index+2]).mean()))
        state=engine.acquired.control.detach_state(state)
        if training:
            loss=torch.func.functional_call(teacher.RefreshedFuture(model),params,(old,t,m,k,8))
            error=abs(float(loss.detach())-float(np.mean(future_scores)));assert error<1e-7
            cop.zero_grad(set_to_none=True)
            (loss+.001*trust).backward()
            norm=torch.nn.utils.clip_grad_norm_(critic.parameters(),1.,error_if_nonfinite=True)
            cop.step();derivative.append(float(norm));parity.append(error);trusts.append(float(trust.detach()))
        rows.append(dict(producer_nll=float(np.mean(scores)),future_nll=float(np.mean(future_scores))))
    return critic,dict(targets=16*blocks,updates=blocks,nll=float(np.mean([.5*(r['producer_nll']+r['future_nll']) for r in rows])),rows=rows,
                       training=training,teacher_future_loss_max_error=max(parity,default=0.),teacher_frames=len(parity),
                       teacher_replayed_targets=16*len(parity),refresh_targets=8*blocks,
                       critic_gradient_norm_mean=float(np.mean(derivative)) if derivative else 0.,
                       trust_ratio_mean=float(np.mean(trusts)) if trusts else 0.,wall_s=time.monotonic()-begin)

