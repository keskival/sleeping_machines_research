"""B10 native credit trained by actual refreshed-Adam future utility."""
import argparse,copy,json,time
import market_actual_update_teacher as teacher
from native_market_credit import NativeMarketCredit
ROOT,torch,np=teacher.ROOT,teacher.torch,teacher.np
engine=teacher.engine


class UtilityCredit(torch.nn.Module):
    def __init__(self,base):
        super().__init__();self.base=base;self.core=base.core
        self.operator=torch.nn.Linear(19,264,bias=False).double()
        torch.nn.init.zeros_(self.operator.weight)
    def forward(self,features,events,carry,operator):
        addition=self.operator(operator)
        events=torch.cat((events[:,:,:264]+addition[:,None],events[:,:,264:]),-1)
        return self.base(features+addition,events,carry)


def descriptor(model,opt,grads):
    by=dict(zip((name for name,_ in model.named_parameters()),grads));values=[]
    leaves=dict(model.named_parameters())
    for kind in ('gradient','parameter','first','second'):
        for j,l in enumerate(model.layers):
            for key in ('log_rate','freq'):
                name=f'layers.{j}.{key}';p=leaves[name];st=opt.state[p]
                if kind=='gradient':value=by[name].sum()/np.sqrt(l.n)
                elif kind=='parameter':value=p.detach().mean()
                elif kind=='first':value=st.get('exp_avg',torch.zeros_like(p)).sum()/np.sqrt(l.n)
                else:value=st.get('exp_avg_sq',torch.zeros_like(p)).mean().sqrt()
                values.append(value.detach().asinh())
    norm=torch.linalg.vector_norm(torch.stack([g.norm() for g in grads]))
    step=max([float(st.get('step',0)) for st in opt.state.values()] or [0.])
    values.extend([norm.detach().log1p(),norm.new_tensor(np.log1p(step)),(1/(norm+1e-6)).clamp(max=1.).detach()])
    return torch.stack(values)[None]


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
        with torch.no_grad():reference,anchor_carry=anchor(features,events,anchor_carry)
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


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--tag',required=True);parser.add_argument('--mode',choices=('smoke','fit'),required=True);args=parser.parse_args()
    torch.set_num_threads(1);torch.set_default_dtype(torch.float64);torch.manual_seed(19131);started=time.monotonic()
    model,inputs=engine.acquired.restored()
    path=ROOT/'experiments/results/market/curie_b10_future_updates_20261013_v1_fit_critic.pt'
    packet=torch.load(path,map_location='cpu',weights_only=False);inputs[str(path.relative_to(ROOT))]=engine.acquired.sha(path)
    anchor=NativeMarketCredit(264).double();anchor.core.head=torch.nn.Linear(anchor.core.total_payload,4).double();anchor.load_state_dict(packet['state_dict'])
    critic=UtilityCredit(copy.deepcopy(anchor));scale=np.array(packet['scale'])
    source_blocks=3 if args.mode=='smoke' else 128;dev_blocks=3 if args.mode=='smoke' else 128
    t,m,path=engine.acquired.load('2026-08-24',128+16*source_blocks);inputs[str(path.relative_to(ROOT))]=engine.acquired.sha(path)
    critic,source=stream(model,critic,anchor,scale,t,m,source_blocks,True)
    td,md,path=engine.acquired.load('2026-08-25',128+16*dev_blocks);inputs[str(path.relative_to(ROOT))]=engine.acquired.sha(path)
    torch.manual_seed(19125)
    _,dev=stream(model,critic,anchor,scale,td,md,dev_blocks,False)
    torch.manual_seed(19125)
    reference=engine.stream(model,anchor,scale,td,md,dev_blocks,'learned_refresh')
    result=dict(status='completed',battle='B10',args=vars(args),acquisition=source,dev=dev,reference=reference,
                dev_gain=reference['nll']-dev['nll'],parameters=sum(p.numel() for p in critic.parameters()),
                input_sha256=inputs,source_sha256={p:engine.acquired.sha(ROOT/p) for p in ('experiments/credit/market_actual_utility_fit_v2.py','experiments/credit/market_actual_update_teacher.py')},
                wall_s=time.monotonic()-started,
                scope='Warm-started nativecritic plus zero-initial19to264operator embedding; truefutureloss through actualclippedAdam and8eventrefresh, own gradients/context treated as issued constants. TRAIN utility learning after factualfuture forecasts; DEV frozencritic/causalprivatebanks, same originalactor reset for both arms. Trust penalty.001 on normalized actualparameterstep deviation; coplr.0001 fixed. Full Adam/directparameter/state teacher paid16replay targets/frame. No imitation labels/newTEST/longsequence/depth/hardware claim; bounded developer fit, strongest saved rules retained.')
    with (ROOT/f'experiments/results/market/{args.tag}.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    if args.mode=='fit':torch.save(dict(state_dict=critic.state_dict(),scale=scale.tolist()),ROOT/f'experiments/results/market/{args.tag}_critic.pt')
    print(source['teacher_future_loss_max_error'],dev['nll'],reference['nll'],result['dev_gain'],flush=True)


if __name__=='__main__':main()
