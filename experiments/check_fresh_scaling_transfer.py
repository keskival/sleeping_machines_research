"""Frozen final-weight acquisition transfer on a new development interval."""
import argparse
import hashlib
import json
import statistics
import sys
import time
from pathlib import Path
from types import SimpleNamespace
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'experiments'))
import window_replay_token_language as driver


def main():
    p=argparse.ArgumentParser();p.add_argument('--tag',required=True);a=p.parse_args();torch.set_num_threads(1)
    sources,inputs,rows={},{},[];offset=25165824;panels=16;tokens=2048
    for seed in (6,7,8):
        values=[]
        for size in ('512k','8m'):
            tag=(f'aws_integrated_window_replay_512k_20261012_s{seed}_v1' if size=='512k' else
                f'aws_integrated_fresh_scaling_8m_pool4_{"20261012" if seed==6 else "20261013"}_s{seed}_v1')
            base=ROOT/f'experiments/results/token_language/{tag}'
            blob=base.with_suffix('.json').read_bytes();r=json.loads(blob);assert r['status']=='completed'
            inputs[str(base.with_suffix('.json').relative_to(ROOT))]=hashlib.sha256(blob).hexdigest()
            for name,digest in r['source_sha256'].items():
                assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,name
                assert name not in sources or sources[name]==digest
                sources[name]=digest
            args=SimpleNamespace(**r['args']);assert args.dev_offset==20971520 and args.dev_tokens==2048
            assert offset>=10485760 and offset>=args.dev_offset+args.dev_tokens
            train=driver.load_tokens(args.train_file);dev=driver.load_tokens(args.dev_file)
            assert offset+panels*tokens<=len(dev)
            counts=driver.LaneTokens(train,0,args.train_tokens,args.lanes).counts()
            order=torch.argsort(counts,descending=True,stable=True)
            torch.manual_seed(seed);model=driver.Model(args,order,counts)
            checkpoint=base.with_suffix('.pt');receipt=driver.sha(checkpoint)
            acquired=torch.load(checkpoint,map_location='cpu',weights_only=False)
            assert acquired['step']==args.steps
            model.load_state_dict(acquired['model']);del acquired
            assert driver.sha(checkpoint)==receipt
            inputs[str(checkpoint.relative_to(ROOT))]=receipt
            original=driver.evaluate(model,driver.interval_tensor(dev,args.dev_offset,args.dev_tokens,args.eval_lanes),args)
            assert original==r['curve'][-1]['dev_nll'],'Saved final-weight DEV does not reproduce'
            start=time.monotonic()
            scores=[driver.evaluate(model,driver.interval_tensor(dev,offset+i*tokens,tokens,args.eval_lanes),args) for i in range(panels)]
            values.append(scores)
            rows.append(dict(seed=seed,acquisition=size,targets=r['presentations_total'],final_original_dev_nll=original,
                transfer_nll=statistics.mean(scores),panel_nll=scores,inference_s=time.monotonic()-start,
                scored_targets=panels*(tokens-args.eval_lanes),training_s=r['train_wall_s_total']))
        gains=[x-y for x,y in zip(*values)]
        print(json.dumps(dict(seed=seed,paired_transfer_gain=statistics.mean(gains))),flush=True)
    paired=[dict(seed=seed,gain=next(r['transfer_nll'] for r in rows if r['seed']==seed and r['acquisition']=='512k')-
        next(r['transfer_nll'] for r in rows if r['seed']==seed and r['acquisition']=='8m')) for seed in (6,7,8)]
    source=str(Path(__file__).relative_to(ROOT));sources[source]=driver.sha(Path(__file__))
    result=dict(status='completed',battle='R1',rows=rows,paired_gains=paired,
        paired_gain_mean=statistics.mean(r['gain'] for r in paired),paired_gain_sd=statistics.stdev(r['gain'] for r in paired),
        interval=dict(offset=offset,tokens=panels*tokens,panels=panels,scored_targets=panels*(tokens-8)),
        source_sha256=sources,input_sha256=inputs,
        scope='Frozen final weights, three seeds, unseen development interval at25165824,16contiguous2048-token panels. Each panel uses eight lanes and the original inference protocol/reset/RNG policy. Same panels within each acquired-state pair; original final DEV must reproduce exactly. Panels are contiguous, not independent replicates; sample SD is across seeds. No fitting or checkpoint selection on the new interval. Public validation and reserved scaling protocol remain untouched; within-family acquired-data transfer, no external-model claim.')
    with (ROOT/f'experiments/results/diagnostics/{a.tag}.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')


if __name__=='__main__':main()
