"""B10 own-model exact factual gradient references for the admitted schedule."""
import argparse,json,time
import market_future_credit_updates as engine
from market_future_credit_controls import ConstantCredit


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--tag',required=True);args=parser.parse_args()
    engine.torch.set_num_threads(1);engine.torch.set_default_dtype(engine.torch.float64)
    begin=time.monotonic();model,inputs=engine.acquired.restored()
    t,m,path=engine.acquired.load('2026-08-25',128+16*1024);inputs[str(path.relative_to(engine.ROOT))]=engine.acquired.sha(path)
    original=engine.window
    def exact(model,state,t,m,start,group,arm):return original(model,state,t,m,start,group,'bptt')
    engine.window=exact;rows=[]
    try:
        for arm in ('own_carry','own_refresh'):
            engine.torch.manual_seed(19125)
            row=engine.stream(model,ConstantCredit([0.]*4),engine.np.ones(4),t,m,1024,arm)
            row['arm']=arm.replace('own','bptt');rows.append(row)
    finally:engine.window=original
    result=dict(status='completed',battle='B10',rows=rows,input_sha256=inputs,
                source_sha256={p:engine.acquired.sha(engine.ROOT/p) for p in (
                    'experiments/credit/market_future_credit_bptt_controls.py','experiments/credit/market_future_credit_updates.py',
                    'experiments/credit/market_streaming_clock_updates.py')},wall_s=time.monotonic()-begin,
                scope='Own saved native temporal actor with exact factual8event BPTT gradients, matching producer8/future8 scoring, clip1/Adam.0003, initialweights/moments and causal refresh/no-refresh. Numerical state persists; history differentiation truncates at8event producer boundary. No critic/source-guide fitting, external dense model, sealedTEST or altered dataset. Purpose: distinguish native forward-taught credit quality from ordinary exact factual gradients under the identical schedule.')
    with (engine.ROOT/f'experiments/results/market/{args.tag}.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    for row in rows:print(json.dumps({k:v for k,v in row.items() if k!='rows'}),flush=True)


if __name__=='__main__':main()
