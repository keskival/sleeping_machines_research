"""B10 fixed-cut input sufficiency witness for pooled optimizer context."""
import argparse,copy,json,time
import market_actual_utility_fit_v2 as native
teacher,engine=native.teacher,native.engine
ROOT,torch,np=native.ROOT,native.torch,native.np

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--tag',required=True);a=parser.parse_args();torch.set_num_threads(1);torch.set_default_dtype(torch.float64);begin=time.monotonic()
    model,inputs=engine.acquired.restored();opt=torch.optim.Adam(model.parameters(),lr=.0003)
    t,m,path=engine.acquired.load('2026-08-24',128+64);inputs[str(path.relative_to(ROOT))]=engine.acquired.sha(path)
    state=engine.acquired.warm(model,t,m,128);rows=[]
    for block in range(4):
        k=127+16*block;old=engine.acquired.control.detach_state(state);end,grads,_=engine.window(model,state,t,m,k,8,'own')
        current=torch.tensor([.003,-.008,.002,-.004],requires_grad=True)
        if block:
            actor,other=copy.deepcopy((model,opt))
            for name,p in actor.named_parameters():
                if name.endswith(('log_rate','freq')):
                    for key in ('exp_avg','exp_avg_sq'):other.state[p][key]=other.state[p][key].flip(0).clone()
            original_context=native.descriptor(model,opt,grads);alias_context=native.descriptor(actor,other,grads)
            context_error=float((original_context-alias_context).abs().max());assert context_error<1e-12
            q=current.detach().clone().requires_grad_();loss,_,_=teacher.objective(model,opt,grads,current,old,t,m,k,8);alternate,_,_=teacher.objective(actor,other,grads,q,old,t,m,k,8)
            gradient=torch.autograd.grad(loss,current)[0];other_gradient=torch.autograd.grad(alternate,q)[0]
            rows.append(dict(block=block,context_error=context_error,future_loss=float(loss.detach()),alias_future_loss=float(alternate.detach()),demand=gradient.tolist(),alias_demand=other_gradient.tolist(),demand_max_difference=float((gradient-other_gradient).abs().max()),loss_difference=float(alternate.detach()-loss.detach()),pooled_clock_coordinates=64))
        engine.step(model,opt,teacher.guided(model,grads,current.detach()))
        with torch.no_grad():
            state=old
            for index in range(k,k+16):_,state=engine.acquired.control.market_consume(model,t[index:index+1],m[index:index+1],state)
        state=engine.acquired.control.detach_state(state)
    names=['experiments/credit/market_optimizer_context_alias.py','experiments/credit/market_actual_utility_fit_v2.py','experiments/credit/market_actual_update_teacher.py']
    result=dict(status='completed',battle='B10',rows=rows,input_sha256=inputs,source_sha256={n:engine.acquired.sha(ROOT/n) for n in names},wall_s=time.monotonic()-begin,scope='Fixed native actor/issued gradients/incoming state/current observations, two optimizer cuts differing only by a common permutation of per-mode clock first/second moments. Pooled19context equal; future functional actualAdam/refresh demand may differ. Permuted cuts valid moment tensors but not proven reachable under identical observed history. Current critic carry held conceptualconstant; no claim of observed TRAIN/DEV alias frequency or cause of earlier losses. Diagnostic identifies descriptor insufficiency over allowed optimizer cuts, not impossibility of memory-based inference or trained quality.')
    with (ROOT/f'experiments/results/market/{a.tag}.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(rows,flush=True)
if __name__=='__main__':main()
