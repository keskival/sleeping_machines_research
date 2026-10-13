"""B10 real actor-state integration of AWS matrix-free optimizer credit."""
import argparse,copy,json,sys,time
import market_actual_update_teacher as teacher
ROOT,torch,np=teacher.ROOT,teacher.torch,teacher.np
sys.path.insert(0,str(ROOT))
from sleeping_machines.learning.adam_credit_transport import adam_gradient_cotangent
engine=teacher.engine


def geometry(model,optimizer,grads,credit,state,t,m,start,group):
    """Actual future demand plus parameter-step metric on four clock actions."""
    loss,parameters,moments=teacher.objective(model,optimizer,grads,credit,state,t,m,start,group)
    values=list(parameters.values())
    endpoint=torch.autograd.grad(loss,values,retain_graph=True,allow_unused=True)
    endpoint=[torch.zeros_like(v) if g is None else g for v,g in zip(values,endpoint)]
    issued=teacher.guided(model,grads,credit)
    flat=torch.cat([g.detach().reshape(-1) for g in issued])
    leaves=list(model.named_parameters());first=[];second=[];steps=[]
    for name,p in leaves:
        old=optimizer.state[p];first.append(old.get('exp_avg',torch.zeros_like(p)).reshape(-1))
        second.append(old.get('exp_avg_sq',torch.zeros_like(p)).reshape(-1));steps.append(int(old.get('step',0))+1)
    assert len(set(steps))==1
    group_opt=optimizer.param_groups[0];beta1,beta2=group_opt['betas']
    kwargs=dict(step=steps[0],lr=group_opt['lr'],beta1=beta1,beta2=beta2,adam_eps=group_opt['eps'])
    # Four sparse-support directions, complete-vector clipping retained.
    directions=[]
    for j,layer in enumerate(model.layers):
        for key in ('log_rate','freq'):
            directions.append(torch.cat([torch.full_like(p,1/np.sqrt(layer.n)).reshape(-1) if name==f'layers.{j}.{key}' else torch.zeros_like(p).reshape(-1) for name,p in leaves]))
    basis=torch.stack(directions,1)
    transported=adam_gradient_cotangent(flat,torch.cat(first),torch.cat(second),torch.cat([g.detach().reshape(-1) for g in endpoint]),**kwargs)
    predicted=basis.T@transported
    exact=torch.autograd.grad(loss,credit,retain_graph=True)[0]
    # The product of clip and Adam Jacobians is not generally symmetric.
    # Compute the parameter-step pullback with four exact proposal JVPs;
    # cotangent products in the opposite order would give the wrong metric.
    def proposal(action):
        ps,_=teacher.adam_parameters(model,optimizer,teacher.guided(model,grads,action))
        return torch.cat([p.reshape(-1) for p in ps.values()])
    jvp=[]
    for unit in torch.eye(4,dtype=credit.dtype):
        _,column=torch.func.jvp(proposal,(credit.detach(),),(unit,));jvp.append(column.detach())
    actual_jac=torch.stack(jvp,1);metric=actual_jac.T@actual_jac
    return loss,exact,predicted,metric,parameters,moments,kwargs,flat,basis,first,second


def main():
    p=argparse.ArgumentParser();p.add_argument('--tag',required=True);args=p.parse_args()
    torch.set_num_threads(1);torch.set_default_dtype(torch.float64);started=time.monotonic()
    model,inputs=engine.acquired.restored();optimizer=torch.optim.Adam(model.parameters(),lr=.0003)
    t,m,path=engine.acquired.load('2026-08-24',128+16*3);inputs[str(path.relative_to(ROOT))]=engine.acquired.sha(path)
    state=engine.acquired.warm(model,t,m,128);rows=[]
    for block in range(3):
        start=127+16*block;old=engine.acquired.control.detach_state(state)
        _,grads,_=engine.window(model,state,t,m,start,8,'own')
        credit=torch.tensor([.003,-.008,.002,-.004],requires_grad=True)
        loss,exact,predicted,metric,parameters,moments,kwargs,flat,basis,first,second=geometry(model,optimizer,grads,credit,old,t,m,start,8)
        error=float((exact-predicted).abs().max());assert error<1e-10
        eigen=torch.linalg.eigvalsh(metric);assert eigen.min()>-1e-15
        proposals=[]
        # Analytic parameter-step trust metric, local damped descent.
        damping=max(float(eigen.max())*.01,1e-12)
        direction=-torch.linalg.solve(metric+damping*torch.eye(4),exact.detach())
        radius=.001;direction=direction*(radius/direction.norm().clamp(min=radius))
        for scale in (1.,.25):
            candidate=credit.detach()+scale*direction
            real,_,_=teacher.actual(model,optimizer,grads,candidate,old,t,m,start,8)
            proposals.append(dict(radius=radius*scale,future_loss=real,gain=float(loss.detach())-real,linear_predicted_gain=-float(exact.detach()@(scale*direction))))
        rows.append(dict(block=block,transport_error=error,credit_derivative=exact.detach().tolist(),parameter_step_metric=metric.tolist(),metric_eigenvalues=eigen.tolist(),proposals=proposals,gradient_coordinates=flat.numel(),clipped=bool(flat.norm()+1e-6>1)))
        print(rows[-1],flush=True)
        engine.step(model,optimizer,teacher.guided(model,grads,credit.detach()))
        with torch.no_grad():
            state=old
            for index in range(start,start+16):_,state=engine.acquired.control.market_consume(model,t[index:index+1],m[index:index+1],state)
        state=engine.acquired.control.detach_state(state)
    names=['experiments/credit/market_optimizer_transport.py','experiments/credit/market_actual_update_teacher.py','sleeping_machines/learning/adam_credit_transport.py']
    result=dict(status='completed',battle='B10',rows=rows,input_sha256=inputs,source_sha256={name:engine.acquired.sha(ROOT/name) for name in names},wall_s=time.monotonic()-started,scope='Three TRAIN learner versions; exact native future parameter/state demand transported through AWS actual clip/Adam to four clock actions, compared with functional derivative. Parameter-step pullback uses four exact JVPs, not forecast curvature. Damped proposals inspect already revealed TRAIN future labels: oracle teacher diagnostic only, no deployable quality win. Own gradients/incoming state/old moments issued constants. O(n) transport and global contraction paid; no clockless/full-work claim.')
    with (ROOT/f'experiments/results/market/{args.tag}.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
if __name__=='__main__':main()
