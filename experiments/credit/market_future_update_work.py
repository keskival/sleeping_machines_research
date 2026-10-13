"""Executed work/parity for the admitted prospective-credit update pipeline."""
import argparse,json
import market_future_credit_updates as engine
import market_complete_work_probe as probe
from market_complete_work_probe_v2 import LikelihoodAudit


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--tag',required=True);args=parser.parse_args()
    engine.torch.set_num_threads(1);engine.torch.set_default_dtype(engine.torch.float64)
    root=engine.ROOT;model,inputs=engine.acquired.restored()
    parent=root/'experiments/results/market/curie_b10_future_updates_20261013_v1_fit.json'
    saved=json.loads(parent.read_text());inputs[str(parent.relative_to(root))]=engine.acquired.sha(parent)
    checkpoint=root/'experiments/results/market/curie_b10_future_updates_20261013_v1_fit_critic.pt'
    assert saved['input_sha256'][str(checkpoint.relative_to(root))]==engine.acquired.sha(checkpoint)
    packet=engine.torch.load(checkpoint,map_location='cpu',weights_only=False)
    critic=engine.acquired.NativeMarketCredit(264).double()
    critic.core.head=engine.torch.nn.Linear(critic.core.total_payload,4).double()
    critic.load_state_dict(packet['state_dict']);scale=engine.np.array(packet['scale'])
    inputs[str(checkpoint.relative_to(root))]=engine.acquired.sha(checkpoint)
    t,m,path=engine.acquired.load('2026-08-25',128+16*32);inputs[str(path.relative_to(root))]=engine.acquired.sha(path)
    probe.TorchAudit=LikelihoodAudit;rows=[]
    for arm in ('own_carry','own_refresh','learned_refresh'):
        engine.torch.manual_seed(19125)
        with probe.instrumentation() as (ta,na):
            row=engine.stream(model,critic,scale,t,m,32,arm)
        previous=next(r for r in saved['rows'] if r['arm']==arm)
        error=max(abs(a[key]-b[key]) for a,b in zip(row['rows'],previous['rows'][:32]) for key in ('producer_nll','future_nll'))
        assert error<1e-12
        row.update(torch_work=ta.result(),numpy_work=na.result(),saved512target_parity_error=error)
        assert not row['torch_work']['unsupported_floating_operators']
        assert not row['numpy_work']['unsupported']
        row['covered_arithmetic_flops']=row['torch_work']['arithmetic_flops']+row['numpy_work']['arithmetic_flops']
        rows.append(row)
    result=dict(status='completed',battle='B10',rows=rows,input_sha256=inputs,
                source_sha256={p:engine.acquired.sha(root/p) for p in (
                    'experiments/credit/market_future_update_work.py','experiments/credit/market_future_credit_updates.py',
                    'experiments/credit/market_complete_work_probe.py','experiments/credit/market_complete_work_probe_v2.py',
                    'experiments/credit/numpy_work_ledger.py','sleeping_machines/operation_audit.py')},
                scope='Executed512DEVtargets/32actualAdamupdates per arm, includes127daywarmup events, own trace work, refresh, native prediction and optimizer. Initialization/checkpoint restore and source guide acquisition outside:128TRAINframes/8192teacher targets plus factual/critic learning and original actor fitting are additionally charged, not measured here. Stable mathematical arithmetic conventions; special/Pythonmath/random internals separate, logical payload is not physicalASIC traffic. No16384work extrapolation or total fitting win. Saved512prefix forecasts must reproduce below1e-12.')
    with (root/f'experiments/results/market/{args.tag}.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    for row in rows:print(json.dumps({k:row[k] for k in ('arm','nll','covered_arithmetic_flops','saved512target_parity_error')}),flush=True)


if __name__=='__main__':main()
