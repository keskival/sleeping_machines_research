"""B1 future-taught updates reuse the next actual forward/backward, eliminating replay."""
import time
import numpy as np
import torch
from torch.utils.flop_counter import FlopCounterMode
import train_native_future_update as base


def fit(args,depth,train,dev,scale,quantiles,K):
    torch.manual_seed(args.seed+3000+depth)
    original=base.Actor(K,16,8,depth,2,4,4,0.,scale,quantiles).double()
    initial={k:v.detach().clone() for k,v in original.state_dict().items()}
    rows=[];weights={}
    for arm in ['fixed_adam','future_taught']:
        actor=base.Actor(K,16,8,depth,2,4,4,0.,scale,quantiles).double();actor.load_state_dict(initial)
        torch.manual_seed(args.seed+3100+depth);controller=base.Controller().double();meta=torch.optim.Adam(controller.parameters(),lr=.001)
        names=list(dict(actor.named_parameters()));groups=[base.group(n) for n in names]
        moments={n:(torch.zeros_like(p),torch.zeros_like(p)) for n,p in actor.named_parameters()}
        history=[];updates=0;presentations=0;feedback_targets=0;fit_s=eval_s=0.;supported_flops=0;meta_steps=0
        for epoch in range(args.epochs):
            order=np.random.default_rng(args.seed+epoch).permutation(len(train))
            ordered=[train[i] for i in order];blocks=[base.batch(ordered[i:i+8]) for i in range(0,len(ordered),8)]
            issued=None;tick=time.monotonic()
            with FlopCounterMode(display=False) as counter:
                for i,data in enumerate(blocks):
                    parameters=dict(actor.named_parameters());actor.zero_grad()
                    loss,_,_,count=actor(*data);assert torch.isfinite(loss)
                    raw_gradients=[g.detach() for g in torch.autograd.grad(loss,list(parameters.values()))]
                    if issued is not None:
                        previous_factors,previous_directions=issued
                        # Exact local chain rule for the previously issued one-step update.
                        # Actual next loss supplies its derivative, once, through normal actor learning.
                        teacher=sum(previous_factors[base.group(n)]*(g*previous_directions[n]).sum()
                                    for n,g in zip(names,raw_gradients))
                        objective=-.001*teacher+.001*torch.log(previous_factors).square().mean()
                        meta.zero_grad();objective.backward();torch.nn.utils.clip_grad_norm_(controller.parameters(),1.);meta.step()
                        meta_steps+=1;feedback_targets+=int(count);issued=None
                    norm=torch.sqrt(sum(g.square().sum() for g in raw_gradients));clip=(10/norm.clamp_min(1e-30)).clamp_max(1)
                    gradients=[g*clip for g in raw_gradients];updates+=1;presentations+=int(count)
                    directions={};group_norm=[]
                    for n,g in zip(names,gradients):
                        m,v=moments[n];m=.9*m+.1*g;v=.999*v+.001*g.square();moments[n]=(m,v)
                        directions[n]=(m/(1-.9**updates))/(torch.sqrt(v/(1-.999**updates))+1e-8)
                    for index in range(4):
                        values=[g.square().mean() for g,k in zip(gradients,groups) if k==index]
                        group_norm.append(torch.log1p(torch.stack(values).mean().sqrt()))
                    features=torch.stack([loss.detach().clamp(-100,100)/10,torch.tensor(epoch/args.epochs),*group_norm]).detach()
                    factors=controller(features) if arm=='future_taught' else torch.ones(4)
                    with torch.no_grad():
                        for n,p in parameters.items():p.sub_(.001*factors[base.group(n)]*directions[n])
                    if arm=='future_taught' and i+1<len(blocks):issued=(factors,directions)
            fit_s+=time.monotonic()-tick;supported_flops+=counter.get_total_flops()
            tick=time.monotonic();quality=base.score(actor,dev);eval_s+=time.monotonic()-tick
            history.append(dict(epoch=epoch+1,**quality,fit_wall_s=fit_s,supported_fit_flops=supported_flops,factors=factors.detach().tolist()))
        rows.append(dict(depth=depth,arm=arm,history=history,fit_wall_s=fit_s,evaluation_wall_s=eval_s,
                         supported_fit_flops=supported_flops,updates=updates,train_target_presentations=presentations,
                         extra_lookahead_targets=0,controller_feedback_targets=feedback_targets,controller_updates=meta_steps,
                         actor_parameters=sum(p.numel() for p in actor.parameters()),controller_parameters=sum(p.numel() for p in controller.parameters())))
        weights[arm]=dict(actor=actor.state_dict(),controller=controller.state_dict(),meta=meta.state_dict(),moments=moments)
    if args.stage=='smoke':
        path=base.ROOT/'experiments/results/credit/aws_native_future_update_20261011_smoke.pt'
        reference=torch.load(path,weights_only=True)[str(depth)]
        errors={}
        for arm in weights:
            errors[arm]=max(float((weights[arm][part][name]-reference[arm][part][name]).abs().max())
                            for part in ['actor','controller'] for name in weights[arm][part])
        assert max(errors.values())<1e-10,errors
        for row in rows:row['retained_replay_parameter_parity']=errors[row['arm']]
    return rows,weights


if __name__=='__main__':
    base.SOURCES.append('experiments/credit/train_native_future_update_reuse.py')
    base.fit=fit
    base.main()
