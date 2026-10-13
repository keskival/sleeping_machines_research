"""B10 two-world random-ray forward teaching of native future credit."""
import argparse,copy,json,time
import market_future_interface_scaled as base
ROOT,torch,np=base.ROOT,base.torch,base.np


def ray_label(model,state,sensitivity,signs,scale,t,m,start,group):
    coefficients=signs/scale
    epsilon=1e-4/float(np.linalg.norm(coefficients))
    worlds=[]
    with torch.no_grad():
        for sign in (1.,-1.):
            shifted=copy.deepcopy(base.control.detach_state(state))
            for z,e in zip(base.control.tensors(shifted),sensitivity):
                z.add_(sign*epsilon*torch.as_tensor(e@coefficients).reshape_as(z))
            loss,_=base.future(model,shifted,t,m,start,group);worlds.append(float(loss))
    return (worlds[0]-worlds[1])/(2*epsilon),epsilon


def collect(model,critic,optimizer,t,m,state,scale):
    carry=None;records=[];rng=np.random.default_rng(19128)
    for block in range(128):
        k=127+16*block;old=base.flat(state)
        end,sensitivity,scores=base.producer(model,state,t,m,k,8)
        current=base.flat(end);features=torch.cat((old,current,current-old))[None]
        feedback=current.new_zeros(8)
        feedback[0]=torch.log1p((t[k+8]-t[k])/model.scale)
        feedback[1:7]=torch.nn.functional.one_hot(m[k+1:k+9],6).double().mean(0)
        feedback[7]=float(np.mean(scores))
        signs=rng.choice([-1.,1.],size=4) # issued before future observations
        critic.core.train(True)
        prediction,carry=critic(features,torch.cat((features[0],feedback))[None,None],carry)
        target,epsilon=ray_label(model,end,sensitivity,signs,scale,t,m,k+8,8)
        predicted=(prediction*prediction.new_tensor(signs)).sum()
        records.append(dict(target_ray=target,predicted_ray=float(predicted.detach()),epsilon=epsilon,signs=signs.tolist()))
        optimizer.zero_grad(set_to_none=True)
        ((predicted-target).square()/4).backward()
        torch.nn.utils.clip_grad_norm_(critic.parameters(),1.,error_if_nonfinite=True);optimizer.step()
        with torch.no_grad():_,state=base.future(model,end,t,m,k+8,8)
        state=base.control.detach_state(state)
    return dict(blocks=128,records=records,normalized_ray_risk=float(np.mean([(r['predicted_ray']-r['target_ray'])**2/4 for r in records])),
                normalized_zero_ray_risk=float(np.mean([r['target_ray']**2/4 for r in records])),
                forward_teacher_worlds=256,forward_teacher_targets=2048)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--tag',required=True);args=parser.parse_args()
    torch.set_num_threads(1);torch.set_default_dtype(torch.float64);torch.manual_seed(19124);started=time.monotonic()
    model,inputs=base.restored();t,m,path=base.load('2026-08-24',128+16*128);inputs[str(path.relative_to(ROOT))]=base.sha(path)
    state=base.warm(model,t,m,128);checked=base.contract(model,t,m,state,127)
    scale=np.maximum(1e-4,np.abs(np.array(checked['exact_directional_credit'])))
    end,sensitivity,_=base.producer(model,copy.deepcopy(state),t,m,127,8)
    # Paid independent state adjoint audits the mixed ray before admission.
    boundary=copy.deepcopy(end)
    for z in base.control.tensors(boundary):z.requires_grad_()
    loss,_=base.future(model,boundary,t,m,135,8)
    cot=torch.autograd.grad(loss,tuple(base.control.tensors(boundary)))
    b=sum((e.T@c.detach().numpy().reshape(-1) for e,c in zip(sensitivity,cot)),np.zeros(4))
    signs=np.array([1.,-1.,1.,-1.]);q,epsilon=ray_label(model,end,sensitivity,signs,scale,t,m,135,8)
    ray_error=abs(q-float(signs@(b/scale)));assert ray_error<.01
    critic=base.NativeMarketCredit(264).double();critic.core.head=torch.nn.Linear(critic.core.total_payload,4).double()
    torch.nn.init.zeros_(critic.core.head.weight);torch.nn.init.zeros_(critic.core.head.bias)
    optimizer=torch.optim.Adam(critic.parameters(),lr=.001)
    train=collect(model,critic,optimizer,t,m,state,scale)
    td,md,path=base.load('2026-08-25',128+16*32);inputs[str(path.relative_to(ROOT))]=base.sha(path)
    dev=base.collect(model,critic,optimizer,td,md,base.warm(model,td,md,128),32,False,scale)
    checkpoint=ROOT/f'experiments/results/market/{args.tag}_critic.pt'
    assert not checkpoint.exists();torch.save(dict(state_dict=critic.state_dict(),scale=scale.tolist()),checkpoint)
    inputs[str(checkpoint.relative_to(ROOT))]=base.sha(checkpoint)
    result=dict(status='completed',battle='B10',train=train,dev=dev,scale=scale.tolist(),contract=checked,
                normalized_ray_adjoint_error=ray_error,ray_epsilon=epsilon,critic_parameters=sum(p.numel() for p in critic.parameters()),
                wall_s=time.monotonic()-started,input_sha256=inputs,
                source_sha256={p:base.sha(ROOT/p) for p in ('experiments/credit/market_future_interface_rays.py',
                    'experiments/credit/market_future_interface_scaled.py','experiments/market/projected_clock_state.py',
                    'experiments/credit/native_market_credit.py')},
                scope='Fixed saved actor/native31,804parameter critic;128TRAINframes/32DEVframes same intervals as coordinate teacher. Independent issued Rademacher rays, finite differences along E D^-1 a,scaled parameter-direction perturbation norm1e-4. Two TRAIN teacher worlds/frame,75%fewer worlds/targets than coordinate teaching; full four-coordinate DEV labels evaluation only. BP audits initial contract, never fit labels. Unbiased squared-error gradient identity in the exact directional limit; finite-difference error and nonlinearAdam paths remain. No actor-update or total-work/energy win from this acquisition experiment.')
    with (ROOT/f'experiments/results/market/{args.tag}.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('train','dev','source_sha256','input_sha256')}),flush=True)
    print('DEV',dev['mse'],dev['zero_credit_mse'],flush=True)


if __name__=='__main__':main()
