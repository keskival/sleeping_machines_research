"""B10 standalone actor acquisition cost, without unused critic preparation."""
import argparse,json,copy
import market_complete_work_probe as probe
from market_complete_work_probe_v2 import LikelihoodAudit
from market_streaming_clock_updates import stream


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--tag',required=True);args=ap.parse_args()
    e=probe.engine;probe.TorchAudit=LikelihoodAudit
    e.torch.set_num_threads(1);e.torch.set_default_dtype(e.torch.float64)
    model,inputs=e.restored();opt=e.torch.optim.Adam(model.parameters(),lr=.0003)
    t,m,path=e.load('2026-08-24',128+2048);inputs[str(path.relative_to(e.ROOT))]=e.sha(path)
    with probe.instrumentation() as (ta,na):
        bundle,source=stream((model,opt,e.warm(model,t,m,128)),t,m,127,64,32,'bptt')
    source_work=dict(torch=ta.result(),numpy=na.result())
    t,m,path=e.load('2026-08-25',128+512);inputs[str(path.relative_to(e.ROOT))]=e.sha(path)
    bundle=(bundle[0],bundle[1],e.warm(bundle[0],t,m,128))
    _,own=stream(bundle,t,m,127,16,32,'own')
    parent=e.ROOT/'experiments/results/market/curie_b10_complete_work_20261013_v2_probe.json'
    prior=json.loads(parent.read_text());saved=next(r for r in prior['rows'] if r['arm']=='own')
    error=max(abs(a-b) for a,b in zip(own['block_nll'],saved['block_nll']));assert error<1e-12
    actor_cost=source_work['torch']['arithmetic_flops']+source_work['numpy']['arithmetic_flops']
    native_cost=prior['acquisition_work']['torch']['arithmetic_flops']+prior['acquisition_work']['numpy']['arithmetic_flops']
    totals=[]
    for row in prior['rows']:
        prep=native_cost if row['arm'] in ['frozen','online'] else actor_cost
        totals.append(dict(arm=row['arm'],nll=row['nll'],scored_targets=row['targets'],
            acquisition_arithmetic_flops=prep,deployment_arithmetic_flops=row['covered_arithmetic_flops'],
            complete_continuation_arithmetic_flops=prep+row['covered_arithmetic_flops'],
            continuation_arithmetic_flops_per_dev_target=(prep+row['covered_arithmetic_flops'])/row['targets']))
    files=['experiments/credit/market_actor_acquisition_work.py',
           'experiments/credit/market_complete_work_probe.py','experiments/credit/market_complete_work_probe_v2.py',
           'experiments/credit/numpy_work_ledger.py','experiments/credit/market_streaming_clock_updates.py']
    result=dict(status='completed',battle='B10',acquisition=source,acquisition_work=source_work,
        actor_acquisition_arithmetic_flops=actor_cost,guidance_acquisition_arithmetic_flops=native_cost,
        guidance_preparation_extra_arithmetic_flops=native_cost-actor_cost,
        standalone_own_score_max_error=error,rows=totals,
        source_sha256={p:e.sha(e.ROOT/p) for p in files},
        input_sha256={**inputs,str(parent.relative_to(e.ROOT)):e.sha(parent)},
        scope='Actor-only exact2048TRAIN preparation charged to standalone own/always-remote controls; native actor-plus-guidance preparation charged to guided arms. Same512target scored prefix, identical standalone own forecasts within1e-12. Arithmetic conventions and missing Python/math/random/special-function internals inherited from probe. Original predictor prefit outside continuation boundary; not full-lifetime fitting arithmetic, physical traffic or energy. No needless critic-training charge to controls.')
    out=e.ROOT/f'experiments/results/market/{args.tag}.json';assert not out.exists()
    with out.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['acquisition','acquisition_work','source_sha256','input_sha256']}),flush=True)


if __name__=='__main__':main()
