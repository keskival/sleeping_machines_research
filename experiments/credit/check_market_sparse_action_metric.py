"""B10 analytic action-tangent integration gate on real actor versions."""
import argparse,json,time
import market_optimizer_transport as audit
from market_sparse_action_metric import sparse_adam_metric
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
        supports=[]
        for column in basis.T:
            indices=torch.nonzero(column,as_tuple=False).flatten();supports.append((indices,column[indices]))
        actual=sparse_adam_metric(g,torch.cat(first),torch.cat(second),supports,**kwargs)
        error=float((actual-metric).abs().max());assert error<1e-13
        rows.append(dict(block=block,parameter_metric_error=error,gradient_coordinates=g.numel(),action_columns=4,sparse_support_entries=sum(x.numel() for x,_ in supports),clipped=bool(g.norm()+1e-6>1)))
        engine.step(model,opt,teacher.guided(model,grads,credit.detach()))
        with torch.no_grad():
            state=old
            for index in range(k,k+16):_,state=engine.acquired.control.market_consume(model,t[index:index+1],m[index:index+1],state)
        state=engine.acquired.control.detach_state(state)
    names=['experiments/credit/check_market_sparse_action_metric.py','experiments/credit/market_sparse_action_metric.py','experiments/credit/market_optimizer_transport.py','experiments/credit/market_actual_update_teacher.py']
    packet=dict(status='completed',battle='B10',rows=rows,input_sha256=inputs,source_sha256={p:engine.acquired.sha(ROOT/p) for p in names},wall_s=time.monotonic()-begin,scope='Sparse diagonal-plus-rank-one metric matches four functional proposal JVPs on three real native actorAdam states.64action-supportentries, no18164-by4tangent allocation needed by sparse helper. Existing globalclip plus additional scalar norm/contractions retained; no fullyclockless/full-FLOP/quality claim.')
    with (ROOT/f'experiments/results/market/{args.tag}.json').open('x') as f:json.dump(packet,f,indent=2);f.write('\n')
    print(rows,flush=True)
if __name__=='__main__':main()
