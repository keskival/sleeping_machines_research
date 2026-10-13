"""B10 analytic action-tangent integration gate on real actor versions."""
import argparse,json,time
import market_optimizer_transport as audit
from market_adam_action_tangent import adam_action_tangent
teacher=audit.teacher;engine=audit.engine;ROOT,torch=audit.ROOT,audit.torch

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--tag',required=True);args=parser.parse_args()
    torch.set_num_threads(1);torch.set_default_dtype(torch.float64);begin=time.monotonic()
    model,inputs=engine.acquired.restored();opt=torch.optim.Adam(model.parameters(),lr=.0003)
    t,m,path=engine.acquired.load('2026-08-24',128+48);inputs[str(path.relative_to(ROOT))]=engine.acquired.sha(path)
    state=engine.acquired.warm(model,t,m,128);rows=[]
    for block in range(3):
        k=127+16*block;old=engine.acquired.control.detach_state(state);_,grads,_=engine.window(model,state,t,m,k,8,'own')
        credit=torch.tensor([.003,-.008,.002,-.004],requires_grad=True)
        loss,exact,predicted,metric,params,moments,kwargs,g,basis,first,second=audit.geometry(model,opt,grads,credit,old,t,m,k,8)
        jac=adam_action_tangent(g,torch.cat(first),torch.cat(second),basis,**kwargs)
        error=float((jac.T@jac-metric).abs().max());assert error<1e-13
        rows.append(dict(block=block,parameter_metric_error=error,gradient_coordinates=g.numel(),action_columns=4))
        engine.step(model,opt,teacher.guided(model,grads,credit.detach()))
        with torch.no_grad():
            state=old
            for index in range(k,k+16):_,state=engine.acquired.control.market_consume(model,t[index:index+1],m[index:index+1],state)
        state=engine.acquired.control.detach_state(state)
    names=['experiments/credit/check_market_adam_action_tangent.py','experiments/credit/market_adam_action_tangent.py','experiments/credit/market_optimizer_transport.py','experiments/credit/market_actual_update_teacher.py']
    packet=dict(status='completed',battle='B10',rows=rows,input_sha256=inputs,source_sha256={p:engine.acquired.sha(ROOT/p) for p in names},wall_s=time.monotonic()-begin,scope='Analytic O(n*4) action tangent Gram matches four functional actual-proposal JVPs on three real native actorAdam states. Global clipping and moment versions retained; removes need for four dual proposal traversals, no measured whole-fit FLOP/quality claim.')
    with (ROOT/f'experiments/results/market/{args.tag}.json').open('x') as f:json.dump(packet,f,indent=2);f.write('\n')
    print(rows,flush=True)
if __name__=='__main__':main()
