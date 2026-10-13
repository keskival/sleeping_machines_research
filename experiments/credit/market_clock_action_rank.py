"""B10 measure future clock demand omitted by uniform four-action credit."""
import argparse,copy,json,time
import market_actual_update_teacher as teacher
ROOT,torch,np,engine=teacher.ROOT,teacher.torch,teacher.np,teacher.engine

def modes(model,grads,credit):
    names=[n for n,_ in model.named_parameters()];out=[g.detach() for g in grads];offset=0
    for j,layer in enumerate(model.layers):
        for key in ('log_rate','freq'):
            i=names.index(f'layers.{j}.{key}');out[i]=out[i]+credit[offset:offset+layer.n];offset+=layer.n
    assert offset==credit.numel()
    return out

def objective(model,opt,grads,credit,old,t,m,k):
    params,_=teacher.adam_parameters(model,opt,modes(model,grads,credit))
    return torch.func.functional_call(teacher.RefreshedFuture(model),params,(old,t,m,k,8))

def actual(model,opt,grads,credit,old,t,m,k):
    actor,optimizer=copy.deepcopy((model,opt));engine.step(actor,optimizer,modes(actor,grads,credit))
    with torch.no_grad():return float(teacher.RefreshedFuture(actor)(copy.deepcopy(old),t,m,k,8))

def main():
    p=argparse.ArgumentParser();p.add_argument('--tag',required=True);a=p.parse_args();torch.set_num_threads(1);torch.set_default_dtype(torch.float64);begin=time.monotonic()
    model,inputs=engine.acquired.restored();opt=torch.optim.Adam(model.parameters(),lr=.0003)
    t,m,path=engine.acquired.load('2026-08-24',128+16*3);inputs[str(path.relative_to(ROOT))]=engine.acquired.sha(path);state=engine.acquired.warm(model,t,m,128)
    basis=torch.zeros(64,4)
    for j in range(4):basis[16*j:16*(j+1),j]=.25
    assert torch.equal(basis.T@basis,torch.eye(4));rows=[]
    for block in range(3):
        k=127+16*block;old=engine.acquired.control.detach_state(state);_,grads,_=engine.window(model,state,t,m,k,8,'own')
        credit=torch.tensor([.003,-.008,.002,-.004],requires_grad=True);c64=(basis@credit.detach()).requires_grad_()
        loss=objective(model,opt,grads,c64,old,t,m,k);q=torch.autograd.grad(loss,c64)[0]
        parent,_,_=teacher.objective(model,opt,grads,credit,old,t,m,k,8);q4=torch.autograd.grad(parent,credit)[0]
        projection_error=float((basis.T@q-q4).abs().max());assert projection_error<1e-10
        real=actual(model,opt,grads,c64.detach(),old,t,m,k);assert abs(real-float(loss.detach()))<1e-9
        parallel=basis@(basis.T@q);perpendicular=q-parallel;norm=float(q.norm());missed=float(perpendicular.norm())
        direction=-perpendicular/perpendicular.norm().clamp_min(1e-30);delta=1e-5
        candidate=actual(model,opt,grads,c64.detach()+delta*direction,old,t,m,k)
        rows.append(dict(block=block,clock_coordinates=64,action_rank=4,loss=float(loss.detach()),projection_error=projection_error,actual_loss_error=abs(real-float(loss.detach())),demand_norm=norm,omitted_norm=missed,omitted_squared_fraction=missed**2/max(norm**2,1e-60),null_action_projection=float((basis.T@direction).abs().max()),null_direction_step=delta,observed_null_direction_gain=real-candidate,predicted_first_order_gain=delta*missed))
        engine.step(model,opt,teacher.guided(model,grads,credit.detach()))
        with torch.no_grad():
            state=old
            for index in range(k,k+16):_,state=engine.acquired.control.market_consume(model,t[index:index+1],m[index:index+1],state)
        state=engine.acquired.control.detach_state(state)
    names=['experiments/credit/market_clock_action_rank.py','experiments/credit/market_actual_update_teacher.py','experiments/credit/market_future_credit_updates.py']
    result=dict(status='completed',battle='B10',rows=rows,input_sha256=inputs,source_sha256={n:engine.acquired.sha(ROOT/n) for n in names},wall_s=time.monotonic()-begin,scope='Three TRAIN learner cuts; differentiable factual8future targets after actual clippedAdam plus8producer refresh. Measures clock-correction action restriction only, not full learner gradient, delayed policy value, population frequency or forecast win. Exact actor/Adam retained. Probes and future backward/actual worlds paid; no new critic fit/TEST/hardwareclaim.')
    with (ROOT/f'experiments/results/market/{a.tag}.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(rows),flush=True)
if __name__=='__main__':main()
