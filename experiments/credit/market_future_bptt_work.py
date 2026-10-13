"""B10 exact own-model BPTT deployment work on the same512target prefix."""
import argparse,json
import market_future_credit_updates as engine
from market_future_credit_controls import ConstantCredit
import market_complete_work_probe as probe
from market_complete_work_probe_v2 import LikelihoodAudit


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--tag',required=True);args=parser.parse_args()
    engine.torch.set_num_threads(1);engine.torch.set_default_dtype(engine.torch.float64)
    model,inputs=engine.acquired.restored();root=engine.ROOT
    parent=root/'experiments/results/market/curie_b10_future_updates_20261013_v1_bptt.json'
    saved=json.loads(parent.read_text());inputs[str(parent.relative_to(root))]=engine.acquired.sha(parent)
    t,m,path=engine.acquired.load('2026-08-25',128+16*32);inputs[str(path.relative_to(root))]=engine.acquired.sha(path)
    original=engine.window
    engine.window=lambda model,state,t,m,start,group,arm:original(model,state,t,m,start,group,'bptt')
    probe.TorchAudit=LikelihoodAudit;rows=[]
    try:
        for arm in ('own_carry','own_refresh'):
            engine.torch.manual_seed(19125)
            with probe.instrumentation() as (ta,na):
                row=engine.stream(model,ConstantCredit([0.]*4),engine.np.ones(4),t,m,32,arm)
            row['arm']=arm.replace('own','bptt')
            reference=next(r for r in saved['rows'] if r['arm']==row['arm'])
            error=max(abs(a[key]-b[key]) for a,b in zip(row['rows'],reference['rows'][:32]) for key in ('producer_nll','future_nll'))
            assert error==0
            row.update(torch_work=ta.result(),numpy_work=na.result(),saved512target_parity_error=error)
            assert not row['torch_work']['unsupported_floating_operators'] and not row['numpy_work']['unsupported']
            row['covered_arithmetic_flops']=row['torch_work']['arithmetic_flops']+row['numpy_work']['arithmetic_flops']
            rows.append(row)
    finally:engine.window=original
    result=dict(status='completed',battle='B10',rows=rows,input_sha256=inputs,
                source_sha256={p:engine.acquired.sha(root/p) for p in (
                    'experiments/credit/market_future_bptt_work.py','experiments/credit/market_future_credit_updates.py',
                    'experiments/credit/market_complete_work_probe.py','experiments/credit/market_complete_work_probe_v2.py',
                    'experiments/credit/numpy_work_ledger.py','sleeping_machines/operation_audit.py')},
                scope='Exact factual8event BPTT in our own actor; same512DEVtargets/32actualAdamupdates and127event warmup boundary as prospective-credit ledger. Full gradient/history, refresh and optimizer execution audited. Stable formula arithmetic, special/Pythonmath/random units separate; logical exposure is not physicalASIC traffic. No extra critic acquisition. Prefix only, no16384work extrapolation or chip-energy measurement.')
    with (root/f'experiments/results/market/{args.tag}.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    for row in rows:print(row['arm'],row['covered_arithmetic_flops'],row['saved512target_parity_error'],flush=True)


if __name__=='__main__':main()
