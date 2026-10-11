"""B1/R1 read-only completed-state error analysis; never fits or scores TEST."""
import argparse
import hashlib
import json
from pathlib import Path
import resource
import time

import numpy as np
import torch
from torch.utils.flop_counter import FlopCounterMode

from sparse_paired_actor_critic import ROOT, TAG, initial, unpack, contrasts


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--tag',required=True)
    ap.add_argument('--reuse-result',default=f'experiments/results/credit/{TAG}_reuse.json'); a=ap.parse_args()
    torch.set_num_threads(1); torch.set_default_dtype(torch.float64)
    out=ROOT/f'experiments/results/credit/{a.tag}.json'; assert not out.exists()
    start=time.monotonic(); _,baseline,_,_,_=initial(); stages=[]; sources={}
    with torch.no_grad(), FlopCounterMode(display=False) as counter:
        for stage in ['acquire','reuse']:
            path=ROOT/a.reuse_result if stage=='reuse' else ROOT/f'experiments/results/credit/{TAG}_{stage}.json'
            d=json.loads(path.read_text()); assert d['status']=='completed'
            for name,digest in d['source_sha256'].items():
                assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,name
            cp=ROOT/d['checkpoint']; assert hashlib.sha256(cp.read_bytes()).hexdigest()==d['checkpoint_sha256']
            payload=torch.load(cp,weights_only=False,map_location='cpu')
            rows=[]; histories={}
            for name,packed in payload['states'].items():
                state=unpack(packed,baseline); history=state['history']; histories[name]=history
                assert len(history)==state['cursor'],f'{name}: aliased history or cursor mismatch'
                blocks=[]
                for first in range(0,state['cursor'],128):
                    h=history[first:first+128]; paid=[v for v in h if v['extra_world']]
                    blocks.append(dict(first_prefix=first,episodes=len(h),
                        factual_loss=float(np.mean([v['factual_loss'] for v in h])),
                        extra_worlds=len(paid),
                        issued_paid_mse=float(np.mean([(v['issued_contrast']-v['observed_contrast'])**2 for v in paid])) if paid else None,
                        paid_zero_mse=float(np.mean([v['observed_contrast']**2 for v in paid])) if paid else None))
                row=dict(arm=name,blocks=blocks,history_factual_mean=float(np.mean([h['factual_loss'] for h in history])),
                         parameter_distance_from_prior=float(sum((u-v).square().sum() for u,v in zip(state['critic'].parameters(),baseline.parameters())).sqrt()))
                data=state['buffer']; predictions=[]; priors=[]
                for first in range(0,len(data),16):
                    batch=data[first:first+16]; x=torch.cat([v['x'] for v in batch])
                    acts=[v['action'] for v in batch]; refs=[v['reference'] for v in batch]
                    predictions.extend(contrasts(state['critic'],x,acts,refs,state['critic_version']).tolist())
                    priors.extend(contrasts(baseline,x,acts,refs,0).tolist())
                if data:
                    y=np.array([v['delta'] for v in data]); current=np.array(predictions); prior=np.array(priors)
                    row['retained_training_packets']=dict(count=len(data),
                        current_mse=float(np.mean((current-y)**2)),prior_mse=float(np.mean((prior-y)**2)),
                        zero_mse=float(np.mean(y**2)),target_mean=float(y.mean()),target_std=float(y.std()),
                        current_rms=float(np.sqrt(np.mean(current**2))),prior_rms=float(np.sqrt(np.mean(prior**2))))
                rows.append(row)
            pairs=[]
            for first,second in [('sparse_learned','sparse_fixed'),('sparse_reuse','sparse_reset')]:
                if first not in histories:continue
                h1,h2=histories[first],histories[second]
                assert len(h1)==len(h2)
                for begin in range(0,len(h1),128):
                    delta=np.array([v['factual_loss']-u['factual_loss'] for u,v in zip(h1[begin:begin+128],h2[begin:begin+128])])
                    pairs.append(dict(comparison=f'{first}_over_{second}',first_prefix=begin,episodes=len(delta),
                                      sampled_factual_gain=float(delta.mean()),
                                      same_sampled_actions=sum(u['action']==v['action'] for u,v in zip(h1[begin:begin+128],h2[begin:begin+128]))))
            stages.append(dict(stage=stage,rows=rows,paired_factual_blocks=pairs))
            sources[str(path.relative_to(ROOT))]=hashlib.sha256(path.read_bytes()).hexdigest()
    result=dict(status='completed',battle='B1/R1',tag=a.tag,stages=stages,diagnostic_supported_flops=counter.get_total_flops(),
        wall_s=time.monotonic()-start,peak_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        source_result_sha256=sources,
        source_sha256={str(Path(__file__).relative_to(ROOT)):hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
        scope='Read-only completed acquisition/reuse checkpoints. Retained-buffer error is in-sample on noisy single continuations; shared held-out probe uses eight-continuation averages and is not a like-for-like error level. Factual blocks use training-time sampled actions and common prefix/future streams, not greedy DEV policy evaluation. Cumulative factual means recomputed from histories because cumulative_fits loss fields retain continuation-stage means. No fitting, parameter change or TEST scoring.')
    out.write_text(json.dumps(result,indent=2)+'\n'); print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
