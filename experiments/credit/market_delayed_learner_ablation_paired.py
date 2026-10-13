"""B10 isolate horizon, subsequent learning and optimizer-memory effects."""
import argparse,copy,json,time
import market_delayed_action_teacher as source
native,teacher,engine=source.native,source.teacher,source.engine
ROOT,torch,np=source.ROOT,source.torch,source.np
token,consume,refresh=source.token,source.consume,source.refresh

def intervened_world(model,opt,critic,carry,scale,old,grads,credit,t,m,k,future_learning=True,moment_cut=None):
    # Clone the issued learner, preserving the optimizer's parameter bindings.
    model,opt=copy.deepcopy((model,opt));critic=copy.deepcopy(critic);critic.core.train(False)
    carry=copy.deepcopy(carry.detach()) if carry is not None else None
    engine.step(model,opt,teacher.guided(model,grads,credit.detach()))
    if moment_cut is not None:
        for name,p in model.named_parameters():
            opt.state[p]['exp_avg']=moment_cut[name][0].clone()
            opt.state[p]['exp_avg_sq']=moment_cut[name][1].clone()
    state=refresh(model,copy.deepcopy(old),t,m,k)
    state,early=consume(model,state,t,m,k+8,8)
    # Subsequent policy decisions and own gradients respond to the alternative.
    incoming=engine.acquired.control.detach_state(state)
    end,later_grads,producer=engine.window(model,state,t,m,k+16,8,'own')
    features,events,operator=token(model,opt,incoming,end,later_grads,producer,t,m,k+16)
    with torch.no_grad():output,carry=critic(features,events,carry,operator)
    if future_learning:
        engine.step(model,opt,teacher.guided(model,later_grads,output[0]*torch.as_tensor(scale)))
        state=refresh(model,incoming,t,m,k+16)
    else:state=engine.acquired.control.detach_state(end)
    state,late=consume(model,state,t,m,k+24,8)
    return model,opt,state,carry,dict(early=early,later_producer=producer,late=late,nll=float(np.mean(early+producer+late)),early_nll=float(np.mean(early)),late_nll=float(np.mean(producer+late)))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--tag',required=True);a=parser.parse_args()
    torch.set_num_threads(1);torch.set_default_dtype(torch.float64);torch.manual_seed(19131);begin=time.monotonic()
    model,inputs=engine.acquired.restored();opt=torch.optim.Adam(model.parameters(),lr=.0003)
    path=ROOT/'experiments/results/market/curie_b10_future_updates_20261013_v1_fit_critic.pt';packet=torch.load(path,map_location='cpu',weights_only=False);inputs[str(path.relative_to(ROOT))]=engine.acquired.sha(path)
    base=native.NativeMarketCredit(264).double();base.core.head=torch.nn.Linear(base.core.total_payload,4).double();base.load_state_dict(packet['state_dict']);critic=native.UtilityCredit(base);critic.core.train(False);scale=np.array(packet['scale'])
    t,m,path=engine.acquired.load('2026-08-24',128+32*8);inputs[str(path.relative_to(ROOT))]=engine.acquired.sha(path)
    state=engine.acquired.warm(model,t,m,128);carry=None;rng=np.random.default_rng(19140);rows=[]
    for block in range(8):
        k=127+32*block;old=engine.acquired.control.detach_state(state);end,grads,scores=engine.window(model,state,t,m,k,8,'own')
        features,events,operator=token(model,opt,old,end,grads,scores,t,m,k)
        with torch.no_grad():output,issued=critic(features,events,carry,operator)
        direction=rng.normal(size=4);direction=.5*direction/np.linalg.norm(direction)
        current=output[0]*torch.as_tensor(scale);candidate=(output[0]+torch.as_tensor(direction))*torch.as_tensor(scale)
        issue_rng=torch.get_rng_state().clone()
        factual=intervened_world(model,opt,critic,issued,scale,old,grads,current,t,m,k)
        factual_rng=torch.get_rng_state().clone()
        torch.set_rng_state(issue_rng)
        alternate=intervened_world(model,opt,critic,issued,scale,old,grads,candidate,t,m,k)
        torch.set_rng_state(issue_rng)
        held=intervened_world(model,opt,critic,issued,scale,old,grads,current,t,m,k,False)
        torch.set_rng_state(issue_rng)
        held_alt=intervened_world(model,opt,critic,issued,scale,old,grads,candidate,t,m,k,False)
        cut_actor,cut_opt=copy.deepcopy((model,opt));engine.step(cut_actor,cut_opt,teacher.guided(cut_actor,grads,current))
        cut={name:(cut_opt.state[p]['exp_avg'].clone(),cut_opt.state[p]['exp_avg_sq'].clone()) for name,p in cut_actor.named_parameters()}
        torch.set_rng_state(issue_rng)
        swapped=intervened_world(model,opt,critic,issued,scale,old,grads,candidate,t,m,k,True,cut)
        assert torch.equal(torch.get_rng_state(),factual_rng)
        torch.set_rng_state(factual_rng)
        q=factual[4]['nll']-alternate[4]['nll'];f=held[4]['nll']-held_alt[4]['nll'];s=factual[4]['nll']-swapped[4]['nll']
        parity=abs(alternate[4]['early_nll']-swapped[4]['early_nll']);assert parity<1e-12
        rows.append(dict(block=block,full_gain=q,held_future_learning_gain=f,swapped_moment_gain=s,moment_effect=s-q,early_moment_swap_parity=parity,learning_preference_reversal=(q>1e-9)!=(f>1e-9),moment_preference_reversal=(q>1e-9)!=(s>1e-9)))
        model,opt,state,carry=factual[:4]
    names=['experiments/credit/market_delayed_learner_ablation_paired.py','experiments/credit/market_delayed_action_teacher.py','experiments/credit/market_actual_utility_fit_v2.py']
    result=dict(status='completed',battle='B10',rows=rows,learning_preference_reversals=sum(r['learning_preference_reversal'] for r in rows),moment_preference_reversals=sum(r['moment_preference_reversal'] for r in rows),input_sha256=inputs,source_sha256={n:engine.acquired.sha(ROOT/n) for n in names},wall_s=time.monotonic()-begin,scope='EightTRAINreceipts, same24targethorizon/sharedfuture observations and identical future native race tape across all interventions; actualglobal tape advances only factualworld. Freeze only next actor update to isolate futurelearning; swap alternative postfirstupdateAdam m/v to factual cut while retaining alternativeparameters/state to isolate optimizer-memory effects. Firstfuture forecasts must be identical undermoment swap. Native parentweightsfrozen; no fittedcritic/DEVquality/publicclaim. Fiveworlds/receipt incl factual, allnative/ownlearning/refresh work additional.')
    with (ROOT/f'experiments/results/market/{a.tag}.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(rows,flush=True)
if __name__=='__main__':main()
