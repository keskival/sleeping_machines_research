"""B10 differentiable actual clipped-Adam/refresh future-loss teacher."""
import argparse,copy,json,time
import market_future_credit_updates as engine
ROOT,torch,np=engine.ROOT,engine.torch,engine.np


def guided(model,grads,credit):
    names=[name for name,_ in model.named_parameters()];out=[g.detach() for g in grads]
    for j,layer in enumerate(model.layers):
        for offset,key in enumerate(('log_rate','freq')):
            i=names.index(f'layers.{j}.{key}')
            out[i]=out[i]+credit[2*j+offset]/np.sqrt(layer.n)
    return out


def adam_parameters(model,optimizer,grads):
    """Audit target: ordinary Adam, wd0, shared clip1, no AMSGrad."""
    group=optimizer.param_groups[0]
    assert len(optimizer.param_groups)==1 and group['weight_decay']==0 and not group['amsgrad']
    beta1,beta2=group['betas'];norm=torch.linalg.vector_norm(torch.stack([torch.linalg.vector_norm(g) for g in grads]))
    coefficient=(1/(norm+1e-6)).clamp(max=1.)
    updated={};moments={}
    for (name,p),gradient in zip(model.named_parameters(),grads):
        state=optimizer.state[p];step=int(state.get('step',0))+1
        gradient=gradient*coefficient
        # lerp matches PyTorch's first-moment operation ordering.
        first=state.get('exp_avg',torch.zeros_like(p)).detach().lerp(gradient,1-beta1)
        second=state.get('exp_avg_sq',torch.zeros_like(p)).detach()*beta2+(1-beta2)*gradient.square()
        # Inactive constant-zero gradient coordinates have zero variance.
        # Mask the sqrt input before differentiating, avoiding inf*0 adjoints.
        positive=second>0
        root=torch.where(positive,torch.where(positive,second,torch.ones_like(second)).sqrt(),torch.zeros_like(second))
        denominator=root/np.sqrt(1-beta2**step)+group['eps']
        updated['actor.'+name]=p.detach()-(group['lr']/(1-beta1**step))*first/denominator
        moments[name]=(first,second)
    return updated,moments


class RefreshedFuture(torch.nn.Module):
    def __init__(self,actor):super().__init__();self.actor=actor
    def forward(self,state,t,m,start,group):
        for index in range(start,start+group):
            _,state=engine.acquired.control.market_consume(self.actor,t[index:index+1],m[index:index+1],state)
        losses=[]
        for index in range(start+group,start+2*group):
            forecast,state=engine.acquired.control.market_consume(self.actor,t[index:index+1],m[index:index+1],state)
            losses.append(-engine.acquired.control.revealed_log_probability(self.actor,forecast,t[index+1:index+2],m[index+1:index+2]).mean())
        return torch.stack(losses).mean()


def objective(model,optimizer,grads,credit,state,t,m,start,group):
    parameters,moments=adam_parameters(model,optimizer,guided(model,grads,credit))
    loss=torch.func.functional_call(RefreshedFuture(model),parameters,(state,t,m,start,group))
    return loss,parameters,moments


def actual(model,optimizer,grads,credit,state,t,m,start,group):
    actor,opt=copy.deepcopy((model,optimizer))
    engine.step(actor,opt,guided(actor,grads,credit.detach()))
    with torch.no_grad():loss=RefreshedFuture(actor)(copy.deepcopy(state),t,m,start,group)
    return float(loss),actor,opt


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--tag',required=True);args=parser.parse_args()
    torch.set_num_threads(1);torch.set_default_dtype(torch.float64);started=time.monotonic()
    model,inputs=engine.acquired.restored();optimizer=torch.optim.Adam(model.parameters(),lr=.0003)
    t,m,path=engine.acquired.load('2026-08-24',128+16*3);inputs[str(path.relative_to(ROOT))]=engine.acquired.sha(path)
    state=engine.acquired.warm(model,t,m,128);rows=[]
    for block in range(3):
        start=127+16*block;old=engine.acquired.control.detach_state(state)
        _,grads,_=engine.window(model,state,t,m,start,8,'own')
        credit=torch.tensor([.003,-.008,.002,-.004],requires_grad=True)
        loss,parameters,moments=objective(model,optimizer,grads,credit,old,t,m,start,8)
        derivative=torch.autograd.grad(loss,credit)[0]
        real,actor,opt=actual(model,optimizer,grads,credit,old,t,m,start,8)
        parameter_error=max(float((parameters['actor.'+name].detach()-p).abs().max()) for name,p in actor.named_parameters())
        moment_error=max(float((moments[name][i].detach()-opt.state[p][key]).abs().max()) for name,p in actor.named_parameters() for i,key in enumerate(('exp_avg','exp_avg_sq')))
        assert max(abs(float(loss.detach())-real),parameter_error,moment_error)<1e-10
        differences=[]
        direction=credit.new_tensor([.5,-.25,.4,-.2]);expected=float(derivative@direction)
        for epsilon in (1e-3,1e-4,1e-5):
            plus,_,_=actual(model,optimizer,grads,credit+epsilon*direction,old,t,m,start,8)
            minus,_,_=actual(model,optimizer,grads,credit-epsilon*direction,old,t,m,start,8)
            differences.append(dict(epsilon=epsilon,derivative=(plus-minus)/(2*epsilon),error=abs((plus-minus)/(2*epsilon)-expected)))
        print('derivative audit',block,expected,differences,flush=True)
        assert min(r['error'] for r in differences)<1e-5
        rows.append(dict(block=block,parameter_error=parameter_error,moment_error=moment_error,future_loss_error=abs(float(loss.detach())-real),credit_derivative=derivative.tolist(),directional_derivative=expected,finite_difference=differences))
        # Advance the ordinary admitted policy so later witnesses have moments.
        engine.step(model,optimizer,guided(model,grads,credit.detach()))
        with torch.no_grad():
            state=old
            for index in range(start,start+16):_,state=engine.acquired.control.market_consume(model,t[index:index+1],m[index:index+1],state)
        state=engine.acquired.control.detach_state(state)
    result=dict(status='completed',battle='B10',rows=rows,input_sha256=inputs,
                source_sha256={p:engine.acquired.sha(ROOT/p) for p in ('experiments/credit/market_actual_update_teacher.py','experiments/credit/market_future_credit_updates.py','experiments/credit/market_native_remote_admission_selected.py')},
                wall_s=time.monotonic()-started,
                scope='Three real actorAdamversions, own gradients treated as issued constants; exact functional clip1/Adam.0003, refreshed8observedproducer events and8futurelosses. Full teacher differentiates future-direct parameters, refreshed state and optimizer. Finite differences use actual ordinaryAdam worlds, including moments. Factual differentiable action only; native hard-route alternative estimator remains separate. Paid contract, no fitted critic/quality/data-efficiency/hardware claim.')
    with (ROOT/f'experiments/results/market/{args.tag}.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
