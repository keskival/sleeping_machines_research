"""B10 finite future learner consequences train native credit actions."""
import argparse,copy,json,time
import market_actual_utility_fit_v2 as native
teacher=native.teacher;engine=native.engine;ROOT,torch,np=native.ROOT,native.torch,native.np


def token(model,opt,old,end,grads,scores,t,m,k):
    first,last=engine.acquired.flat(old),engine.acquired.flat(end)
    features=torch.cat((first,last,last-first))[None];feedback=last.new_zeros(8)
    feedback[0]=torch.log1p((t[k+8]-t[k])/model.scale)
    feedback[1:7]=torch.nn.functional.one_hot(m[k+1:k+9],6).double().mean(0);feedback[7]=float(np.mean(scores))
    return features,torch.cat((features[0],feedback))[None,None],native.descriptor(model,opt,grads)


def consume(model,state,t,m,k,count):
    scores=[]
    with torch.no_grad():
        for index in range(k,k+count):
            forecast,state=engine.acquired.control.market_consume(model,t[index:index+1],m[index:index+1],state)
            scores.append(float(-engine.acquired.control.revealed_log_probability(model,forecast,t[index+1:index+2],m[index+1:index+2]).mean()))
    return engine.acquired.control.detach_state(state),scores


def refresh(model,state,t,m,k):
    with torch.no_grad():
        for index in range(k,k+8):_,state=engine.acquired.control.market_consume(model,t[index:index+1],m[index:index+1],state)
    return engine.acquired.control.detach_state(state)


def world(model,opt,critic,carry,scale,old,grads,credit,t,m,k):
    # Clone the issued learner, preserving the optimizer's parameter bindings.
    model,opt=copy.deepcopy((model,opt));critic=copy.deepcopy(critic);critic.core.train(False)
    carry=copy.deepcopy(carry.detach()) if carry is not None else None
    engine.step(model,opt,teacher.guided(model,grads,credit.detach()))
    state=refresh(model,copy.deepcopy(old),t,m,k)
    state,early=consume(model,state,t,m,k+8,8)
    # Subsequent policy decisions and own gradients respond to the alternative.
    incoming=engine.acquired.control.detach_state(state)
    end,later_grads,producer=engine.window(model,state,t,m,k+16,8,'own')
    features,events,operator=token(model,opt,incoming,end,later_grads,producer,t,m,k+16)
    with torch.no_grad():output,carry=critic(features,events,carry,operator)
    engine.step(model,opt,teacher.guided(model,later_grads,output[0]*torch.as_tensor(scale)))
    state=refresh(model,incoming,t,m,k+16)
    state,late=consume(model,state,t,m,k+24,8)
    return model,opt,state,carry,dict(early=early,later_producer=producer,late=late,nll=float(np.mean(early+producer+late)),early_nll=float(np.mean(early)),late_nll=float(np.mean(producer+late)))


def fit(model,critic,scale,t,m,frames,smoke):
    opt=torch.optim.Adam(model.parameters(),lr=.0003);cop=torch.optim.Adam(critic.parameters(),lr=.0001)
    state=engine.acquired.warm(model,t,m,128);carry=None;rows=[];gain_scale=1e-4;rng=np.random.default_rng(19140);begin=time.monotonic()
    for frame in range(frames):
        k=127+32*frame;old=engine.acquired.control.detach_state(state)
        end,grads,producer=engine.window(model,state,t,m,k,8,'own')
        features,events,operator=token(model,opt,old,end,grads,producer,t,m,k)
        critic.core.train(True);output,issued_carry=critic(features,events,carry,operator)
        direction=rng.normal(size=4);direction=.5*direction/np.linalg.norm(direction)
        current=output[0].detach()*torch.as_tensor(scale)
        candidate=(output[0].detach()+torch.as_tensor(direction))*torch.as_tensor(scale)
        issue_rng=torch.get_rng_state().clone()
        factual=world(model,opt,critic,issued_carry,scale,old,grads,current,t,m,k)
        factual_rng=torch.get_rng_state().clone()
        torch.set_rng_state(issue_rng)
        alternate=world(model,opt,critic,issued_carry,scale,old,grads,candidate,t,m,k)
        assert torch.equal(torch.get_rng_state(),factual_rng), 'Counterfactual race tape length differs'
        torch.set_rng_state(factual_rng)
        zero_action_parity=None;independent_zero_noise=None
        if smoke:
            torch.set_rng_state(issue_rng)
            zero=world(model,opt,critic,issued_carry,scale,old,grads,current,t,m,k)
            zero_action_parity=max(abs(x-y) for field in ('early','later_producer','late') for x,y in zip(zero[4][field],factual[4][field]))
            assert zero_action_parity==0
            torch.set_rng_state(factual_rng)
            unpaired=world(model,opt,critic,issued_carry,scale,old,grads,current,t,m,k)
            independent_zero_noise=unpaired[4]['nll']-factual[4]['nll']
            torch.set_rng_state(factual_rng)
        gain=factual[4]['nll']-alternate[4]['nll'];short_gain=factual[4]['early_nll']-alternate[4]['early_nll']
        accepted=gain>1e-9
        parity=None
        if smoke:
            actual,_,_=teacher.actual(model,opt,grads,current,old,t,m,k,8)
            parity=abs(actual-factual[4]['early_nll']);assert parity<1e-9
        # Both future blocks and intervening producer outcomes have been scored.
        # The label is an issued finite action, not a differentiated BP gradient.
        target=output[0].detach()+torch.as_tensor(direction if accepted else np.zeros(4))
        # Gain scale is predictable: no current/future outcome enters it yet.
        weight=max(gain,0.)/gain_scale if accepted else 0.
        loss=weight*.5*(output[0]-target).square().sum();cop.zero_grad(set_to_none=True);loss.backward()
        prior_scale=gain_scale;gain_scale=.95*gain_scale+.05*abs(gain)
        norm=torch.nn.utils.clip_grad_norm_(critic.parameters(),1.,error_if_nonfinite=True);cop.step()
        moment_distance=sum(float((factual[1].state[p]['exp_avg']-alternate[1].state[q]['exp_avg']).square().sum()) for p,q in zip(factual[0].parameters(),alternate[0].parameters()))**.5
        row=dict(frame=frame,paired_zero_action_parity=zero_action_parity,independent_zero_noise=independent_zero_noise,gain_weight=weight,predictable_gain_scale=prior_scale,accepted=accepted,gain=gain,short_gain=short_gain,sign_reversal=(gain>1e-9)!=(short_gain>1e-9),early_nll=factual[4]['early_nll'],late_nll=factual[4]['late_nll'],source_nll=float(np.mean(producer+factual[4]['early']+factual[4]['later_producer']+factual[4]['late'])),critic_grad_norm=float(norm),future_moment_distance=moment_distance,first_future_parity=parity)
        rows.append(row);model,opt,state,carry=factual[:4]
    return critic,dict(frames=frames,targets=32*frames,updates=2*frames,critic_updates=frames,accepted=sum(r['accepted'] for r in rows),sign_reversals=sum(r['sign_reversal'] for r in rows),rows=rows,counterfactual_worlds=frames,counterfactual_scored_targets=24*frames,counterfactual_refresh_targets=16*frames,counterfactual_own_gradient_targets=8*frames,factual_refresh_targets=16*frames,wall_s=time.monotonic()-begin)


def main():
    p=argparse.ArgumentParser();p.add_argument('--tag',required=True);p.add_argument('--mode',choices=('smoke','fit'),required=True);a=p.parse_args()
    torch.set_num_threads(1);torch.set_default_dtype(torch.float64);torch.manual_seed(19131);begin=time.monotonic()
    model,inputs=engine.acquired.restored();original=copy.deepcopy(model)
    path=ROOT/'experiments/results/market/curie_b10_future_updates_20261013_v1_fit_critic.pt';packet=torch.load(path,map_location='cpu',weights_only=False);inputs[str(path.relative_to(ROOT))]=engine.acquired.sha(path)
    anchor=native.NativeMarketCredit(264).double();anchor.core.head=torch.nn.Linear(anchor.core.total_payload,4).double();anchor.load_state_dict(packet['state_dict']);scale=np.array(packet['scale']);critic=native.UtilityCredit(copy.deepcopy(anchor))
    frames=3 if a.mode=='smoke' else 64;blocks=3 if a.mode=='smoke' else 128
    t,m,path=engine.acquired.load('2026-08-24',128+32*frames);inputs[str(path.relative_to(ROOT))]=engine.acquired.sha(path)
    critic,source=fit(model,critic,scale,t,m,frames,a.mode=='smoke')
    t,m,path=engine.acquired.load('2026-08-25',128+16*blocks);inputs[str(path.relative_to(ROOT))]=engine.acquired.sha(path)
    torch.manual_seed(19125);_,dev=native.stream(original,critic,anchor,scale,t,m,blocks,False)
    torch.manual_seed(19125);reference=engine.stream(original,anchor,scale,t,m,blocks,'learned_refresh')
    names=['experiments/credit/market_delayed_action_teacher_paired.py','experiments/credit/market_actual_utility_fit_v2.py','experiments/credit/market_actual_update_teacher.py','experiments/credit/market_future_credit_updates.py']
    result=dict(status='completed',battle='B10',args=vars(a),acquisition=source,dev=dev,reference=reference,dev_gain=reference['nll']-dev['nll'],input_sha256=inputs,source_sha256={n:engine.acquired.sha(ROOT/n) for n in names},wall_s=time.monotonic()-begin,scope='Native temporal critic plus causal operator context, original actor. One finite spherical radius0.5 normalized-action alternative per32TRAINevents; two actual actorAdamupdates recomputed in each world with pairedfuture native race randomness; global RNG restored to factual nextstate, teacherbranches do not advance it, subsequent native policy responds. Score24postaction targets before current critic learns accepted action target, weighted by positive measured utilitygain divided by predictable past gain EMA(floorinitial1e-4,decay0.95). Factual source trajectory retained regardless of teacher choice, stopgrad delayed label. No exact-BP imitation/learner-unroll derivative/TEST.64sourceframes/2048events, frozen2048DEV, singlecritic; counterfactual own learning/refresh/native work additionally charged. Unknown biases and scaling not inferred from one development fit.')
    with (ROOT/f'experiments/results/market/{a.tag}.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    if a.mode=='fit':torch.save(dict(state_dict=critic.state_dict(),scale=scale.tolist()),ROOT/f'experiments/results/market/{a.tag}_critic.pt')
    print(source['accepted'],source['sign_reversals'],dev['nll'],reference['nll'],result['dev_gain'],flush=True)
if __name__=='__main__':main()
