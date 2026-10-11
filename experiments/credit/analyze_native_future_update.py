"""B1 assess actual native quality/work and admit bounded unchanged-version replication."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time
ROOT=Path(__file__).resolve().parents[2]


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--tag',required=True);ap.add_argument('--parent',required=True)
    ap.add_argument('--continue-if-useful',action='store_true');a=ap.parse_args();start=time.monotonic()
    path=ROOT/a.parent;record=json.loads(path.read_text());assert record['status']=='completed' and record['stage']=='fit'
    sources={'experiments/credit/analyze_native_future_update.py'}
    for s,digest in record['source_sha256'].items():
        assert hashlib.sha256((ROOT/s).read_bytes()).hexdigest()==digest;sources.add(s)
    rows=[]
    for depth in [2,4,8]:
        fixed=next(r for r in record['rows'] if r['depth']==depth and r['arm']=='fixed_adam')
        learned=next(r for r in record['rows'] if r['depth']==depth and r['arm']=='future_taught')
        target=fixed['history'][-1]['nll'];cross=next((r for r in learned['history'] if r['nll']<=target),None)
        # All scheduled validation work is conservatively retained after selection.
        fixed_work=fixed['fit_wall_s']+fixed['evaluation_wall_s']
        matched_work=(cross['fit_wall_s']+learned['evaluation_wall_s']) if cross else None
        rows.append(dict(depth=depth,final_fixed_nll=target,final_learned_nll=learned['history'][-1]['nll'],
                         final_quality_gain=target-learned['history'][-1]['nll'],
                         fixed_full_fit_and_validation_s=fixed_work,
                         learned_full_fit_and_validation_s=learned['fit_wall_s']+learned['evaluation_wall_s'],
                         first_matched_quality_epoch=cross['epoch'] if cross else None,
                         matched_quality_fit_and_validation_s=matched_work,
                         fixed_over_learned_matched_work=fixed_work/matched_work if matched_work else None,
                         supported_flops_ratio_at_matched_quality=fixed['supported_fit_flops']/cross['supported_fit_flops'] if cross else None,
                         final_group_scales=learned['history'][-1]['factors']))
    useful=sum(r['final_quality_gain']>0 for r in rows)>=2 and sum(r['final_quality_gain'] for r in rows)>0
    continuation=[];seed=record['args']['seed']
    if a.continue_if_useful and useful and seed<2:
        tag=f'aws_native_future_update_reuse_20261011_s{seed+1}'
        command=['.venv-docker/bin/python','experiments/tpp/submit.py','--slot','3','--battle','B1',
                 '--source','experiments/credit/train_native_future_update_reuse_full.py',
                 '--result-dir','experiments/results/credit','--timeout','3600','--rss-kb','1500000',
                 '--purpose','B1 unchanged-version replication of positive whole-native future-taught learning quality acrossdepth2/4/8; charge complete work',
                 '--job',f'{tag}|--seed {seed+1} --gate experiments/results/credit/aws_native_future_update_reuse_20261011_smoke.json']
        for source in record['source_sha256']:
            if source!='experiments/credit/train_native_future_update_reuse_full.py':command.extend(['--dep',source])
        continuation.append(subprocess.check_output(command,cwd=ROOT,text=True).strip())
        command=['.venv-docker/bin/python','experiments/tpp/submit.py','--slot','3','--battle','B1',
                 '--source','experiments/credit/analyze_native_future_update.py','--result-dir','experiments/results/credit',
                 '--timeout','180','--rss-kb','500000','--purpose','B1 source-bound native prediction-quality/work decision after replication',
                 '--job',f'{tag}_decision|--parent experiments/results/credit/{tag}.json --continue-if-useful']
        for source in record['source_sha256']:command.extend(['--dep',source])
        # Timestamp filenames are unique per slot; separate the two admissions by one second.
        time.sleep(1.1)
        continuation.append(subprocess.check_output(command,cwd=ROOT,text=True).strip())
    result=dict(status='completed',tag=a.tag,battle='B1 native learning quality/work decision',rows=rows,
                positive_development_quality_gate=useful,continuation=continuation,
                paid_parent_wall_s=record['wall_s'],wall_s=time.monotonic()-start,
                source_sha256={s:hashlib.sha256((ROOT/s).read_bytes()).hexdigest() for s in sorted(sources)},
                input_sha256={a.parent:hashlib.sha256(path.read_bytes()).hexdigest()},
                scope='Scheduled DEV checkpoints,one-seed development gate; fixed final quality target registered before fit. Complete measured fitting wall and all validation retained; paid smoke/failed admission additionally recorded in protocol, hardware energy unmeasured. Supported FLOPs are partial counters. Conditional unchanged-version replication through seed2; no test/Transformer/supremacy claim.')
    with (ROOT/'experiments/results/credit'/f'{a.tag}.json').open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps(result))

if __name__=='__main__':main()
