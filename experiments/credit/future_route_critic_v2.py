"""R1/B1 larger-data diagnostic with a context-free learned-action reference.
Reuses unchanged v1 native interventions/critic/proposer mathematics. Captures
already-generated branch labels; no additional teacher fits or TEST access.
"""
import argparse
import hashlib
import json
import resource
import time

import numpy as np
import torch
from torch.utils.flop_counter import FlopCounterMode
import future_route_critic_v1 as base

SOURCES=['experiments/credit/future_route_critic_v2.py',*base.SOURCES]


def run(a):
    collected=[];original=base.collect
    def capture(model,start,count,pairs):
        # Disjoint history seeds between teacher-seed replicates at larger N.
        offset=100000 if not collected else 200000
        actual=offset+a.seed*1000
        result=original(model,actual,count,pairs)
        collected.append((actual,result));return result
    base.collect=capture
    try:result=base.fit(base.actor(a.depth,a.seed),a.seed,a)
    finally:base.collect=original
    train=collected[0][1][1];dev=collected[1][1][1]
    average=train.mean(0);selected=int(average.argmin())
    raw_errors=(dev-average[None,:]).square().mean(-1)
    energy=dev.square().sum().clamp_min(1e-30)
    reference=dict(selected_action=selected,train_mean_values=average.tolist(),
        dev_relative_mse=float((dev-average[None,:]).square().sum()/energy),
        selected_future_advantage=float(dev[:,selected].mean()),
        chosen_advantages=dev[:,selected].tolist(),
        scope='One TRAIN-fitted mean advantage vector, same fixed action for every DEV context. No DEV tuning.')
    rng=np.random.default_rng(a.seed);indices=rng.integers(0,len(dev),size=(2000,len(dev)))
    for row in result['arms']:
        errors=np.asarray(row['dev']['squared_error_by_packet'])
        advantage=np.asarray(row['dev']['chosen_advantages'])
        improvement=raw_errors.numpy()-errors
        gains=dev[:,selected].numpy()-advantage
        row['dev']['versus_context_free']=dict(
            mean_squared_error_reduction=float(improvement.mean()),
            mse_reduction_bootstrap95=np.quantile(improvement[indices].mean(1),[.025,.975]).tolist(),
            mean_future_loss_reduction=float(gains.mean()),
            future_loss_reduction_bootstrap95=np.quantile(gains[indices].mean(1),[.025,.975]).tolist(),
            scope='Paired DEV-context bootstrap conditional on this fixed teacher, fitted models and TRAIN reference; not seed uncertainty.')
    result.update(context_free_reference=reference,train_seed_start=collected[0][0],
        dev_seed_start=collected[1][0],dev_branch_advantages=dev.tolist(),
        registered_gate='Report each seed: critic DEV MSE below TRAIN-mean reference and selected future loss below its fixed action. Three seeded replicates; bootstrap is context uncertainty only. No automatic scaling from TRAIN fit.')
    return result


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--tag',required=True)
    ap.add_argument('--mode',choices=['smoke','pilot'],required=True)
    ap.add_argument('--seed',type=int,default=170);ap.add_argument('--depth',type=int,default=2)
    ap.add_argument('--pairs',type=int,default=2);ap.add_argument('--steps',type=int,default=256)
    ap.add_argument('--proposal-steps',type=int,default=32);ap.add_argument('--train-packets',type=int,default=128)
    ap.add_argument('--dev-packets',type=int,default=64);a=ap.parse_args()
    out=base.ROOT/'experiments/results/credit'/(a.tag+'.json')
    if out.exists():raise FileExistsError(out)
    if a.mode=='smoke':a.steps=2;a.proposal_steps=2;a.train_packets=2;a.dev_packets=2
    torch.set_num_threads(1);torch.set_default_dtype(torch.float64);start=time.monotonic()
    with FlopCounterMode(display=False) as counter:result=run(a)
    record=dict(status='completed',tag=a.tag,battle='R1/B1',args=vars(a),training=True,
        decision='Does increased TRAIN diversity yield conditional future credit beyond a TRAIN-fitted constant action, before deep coupled learning?',
        metrics=result,wall_s=time.monotonic()-start,peak_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        supported_flops=counter.get_total_flops(),flop_scope='Supported PyTorch formulas; excludes special/unsupported operations. All branch, critic and proposal work included; wall includes profiling.',
        source_sha256={s:hashlib.sha256((base.ROOT/s).read_bytes()).hexdigest() for s in SOURCES})
    temporary=out.with_suffix('.tmp');temporary.write_text(json.dumps(record,indent=2)+'\n');temporary.replace(out)
    print('RESULT',json.dumps({k:v for k,v in record.items() if k!='metrics'}),flush=True)


if __name__=='__main__':main()
