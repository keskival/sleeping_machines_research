"""B1 acquired policy transfer decision, complete work and bounded confirmation."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time
ROOT=Path(__file__).resolve().parents[2]


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--tag',required=True);parser.add_argument('--parent',required=True)
    parser.add_argument('--continue-if-useful',action='store_true');a=parser.parse_args();start=time.monotonic()
    path=ROOT/a.parent;record=json.loads(path.read_text());assert record['status']=='completed' and record['stage']=='fit'
    sources={'experiments/credit/analyze_native_update_transfer.py'}
    for source,digest in record['source_sha256'].items():
        assert hashlib.sha256((ROOT/source).read_bytes()).hexdigest()==digest;sources.add(source)
    rows=[]
    for depth in [2,4,8]:
        frozen=next(r for r in record['decisions'] if r['depth']==depth and r['arm']=='frozen_acquired_policy')
        static=next(r for r in record['decisions'] if r['depth']==depth and r['arm']=='static_acquired_groups')
        work=frozen['matched_fit_and_all_validation_s'];baseline=frozen['baseline_fit_and_validation_s']
        savings=baseline-work if work is not None else None
        acquisition=record['inherited_acquisition_wall_s']
        rows.append(dict(**frozen,quality_gain_vs_static=static['final_nll']-frozen['final_nll'],
                         incremental_matched_work_saving=savings,conservative_acquisition_wall_s=acquisition,
                         conservative_break_even_fit_count=int(acquisition/savings)+1 if savings and savings>0 else None,
                         static_final_quality_gain=static['final_quality_gain'],static_work_ratio=static['baseline_over_matched_work']))
    gate=sum(r['quality_gain_vs_static']>0 and (r['baseline_over_matched_work'] or 0)>1 and r['final_quality_gain']>0 for r in rows)>=2
    continuation=[];seed=record['seed']
    if a.continue_if_useful and gate and seed<5:
        tag=f'aws_native_update_transfer_20261012_s{seed+1}'
        cmd=['.venv-docker/bin/python','experiments/tpp/submit.py','--slot','3','--battle','B1',
             '--source','experiments/credit/train_native_update_transfer.py','--result-dir','experiments/results/credit',
             '--timeout','1800','--rss-kb','1500000','--purpose','B1 unchanged acquired-policy transfer confirmation against static scales and complete matched-quality work',
             '--job',f'{tag}|--stage fit --seed {seed+1} --gate experiments/results/credit/aws_native_update_transfer_20261012_smoke.json']
        for source in record['source_sha256']:
            if source!='experiments/credit/train_native_update_transfer.py':cmd.extend(['--dep',source])
        continuation.append(subprocess.check_output(cmd,cwd=ROOT,text=True).strip());time.sleep(1.1)
        cmd=['.venv-docker/bin/python','experiments/tpp/submit.py','--slot','3','--battle','B1',
             '--source','experiments/credit/analyze_native_update_transfer.py','--result-dir','experiments/results/credit',
             '--timeout','180','--rss-kb','500000','--purpose','B1 completed acquired-policy quality/work and amortization decision',
             '--job',f'{tag}_decision|--parent experiments/results/credit/{tag}.json --continue-if-useful']
        for source in record['source_sha256']:cmd.extend(['--dep',source])
        continuation.append(subprocess.check_output(cmd,cwd=ROOT,text=True).strip())
    result=dict(status='completed',tag=a.tag,battle='B1 acquired native update transfer',seed=seed,rows=rows,
                useful_conditional_quality_and_incremental_work_gate=gate,continuation=continuation,
                source_sha256={s:hashlib.sha256((ROOT/s).read_bytes()).hexdigest() for s in sorted(sources)},
                input_sha256={a.parent:hashlib.sha256(path.read_bytes()).hexdigest()},wall_s=time.monotonic()-start,
                scope='Whole-native fresh-initialization development comparison on same corpus/DEV. Conditional policy vs TRAIN-acquired static scales; work includes complete fitting and all scheduled validation. Acquisition charged separately, conservative break-even model explicitly repeated equivalent fits, not measured deployment saving. Confirmation requires conditional quality over static and positive incremental work saving at>=2depths; no TEST/Transformer claim.')
    with (ROOT/'experiments/results/credit'/f'{a.tag}.json').open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps(result))

if __name__=='__main__':main()
