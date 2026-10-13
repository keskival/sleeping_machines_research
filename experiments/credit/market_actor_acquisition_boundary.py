"""Normalize actor-only acquisition to the same pre-warmed boundary as guidance."""
import argparse,json,copy
import market_complete_work_probe as probe
from market_complete_work_probe_v2 import LikelihoodAudit


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--tag',required=True);args=ap.parse_args()
    e=probe.engine;probe.TorchAudit=LikelihoodAudit
    e.torch.set_num_threads(1);e.torch.set_default_dtype(e.torch.float64)
    parent=e.ROOT/'experiments/results/market/curie_b10_actor_acquisition_20261013_v1.json'
    prior=json.loads(parent.read_text());model,_=e.restored();t,m,_=e.load('2026-08-24',128+2048)
    with probe.instrumentation() as (ta,na):e.warm(model,t,m,128)
    warm=dict(torch=ta.result(),numpy=na.result())
    warm_flops=warm['torch']['arithmetic_flops']+warm['numpy']['arithmetic_flops']
    actor_cost=prior['actor_acquisition_arithmetic_flops']-warm_flops
    corrected=copy.deepcopy(prior);corrected['actor_acquisition_arithmetic_flops']=actor_cost
    corrected['guidance_preparation_extra_arithmetic_flops']=prior['guidance_acquisition_arithmetic_flops']-actor_cost
    for row in corrected['rows']:
        row['acquisition_arithmetic_flops']=prior['guidance_acquisition_arithmetic_flops'] if row['arm'] in ['frozen','online'] else actor_cost
        row['complete_continuation_arithmetic_flops']=row['acquisition_arithmetic_flops']+row['deployment_arithmetic_flops']
        row['continuation_arithmetic_flops_per_dev_target']=row['complete_continuation_arithmetic_flops']/row['scored_targets']
    corrected['raw_actor_acquisition_work_includes_warmup']=corrected.pop('acquisition_work')
    corrected['measured_train_warmup_work']=warm
    corrected['source_sha256']['experiments/credit/market_actor_acquisition_boundary.py']=e.sha(e.ROOT/'experiments/credit/market_actor_acquisition_boundary.py')
    corrected['input_sha256'][str(parent.relative_to(e.ROOT))]=e.sha(parent)
    corrected['scope']='Actor-only raw v1 probe includes127event TRAIN warmup in its instrumented acquisition despite its outside-boundary scope text. This correction subtracts independently executed identical warmup arithmetic to match guided acquisition pre-warmed boundary. Raw ledger is preserved and labelled; original prefit/day warmup outside corrected continuation arithmetic. Complete continuation means acquisition plus actual512DEVprobe under declared arithmetic conventions, not full lifetime arithmetic or energy. Standalone controls omit critic preparation; native arms pay it.'
    out=e.ROOT/f'experiments/results/market/{args.tag}.json';assert not out.exists()
    with out.open('x') as f:json.dump(corrected,f,indent=2);f.write('\n')
    print(json.dumps(dict(actor_acquisition_flops=actor_cost,warmup_flops=warm_flops,guidance_extra=corrected['guidance_preparation_extra_arithmetic_flops'],rows=corrected['rows'])),flush=True)


if __name__=='__main__':main()
