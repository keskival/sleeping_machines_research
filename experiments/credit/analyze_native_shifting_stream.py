"""B1/G4 same-target changing-stream learning decision and bounded replication."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time
ROOT=Path(__file__).resolve().parents[2]


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--tag',required=True);ap.add_argument('--parent',required=True);ap.add_argument('--continue-if-useful',action='store_true');a=ap.parse_args();start=time.monotonic()
    path=ROOT/a.parent;record=json.loads(path.read_text());assert record['status']=='completed' and record['stage']=='fit'
    sources={'experiments/credit/analyze_native_shifting_stream.py'}
    for s,digest in record['source_sha256'].items():assert hashlib.sha256((ROOT/s).read_bytes()).hexdigest()==digest;sources.add(s)
    rows=[]
    for depth in [4,8]:
        online=next(r for r in record['rows'] if r['depth']==depth and r['arm']=='online_acquired_policy')
        comparisons=[]
        for reference in [r for r in record['rows'] if r['depth']==depth and r['arm']!='online_acquired_policy']:
            post_gain=sum(reference['regimes'][i]['first_window_nll']-online['regimes'][i]['first_window_nll'] for i in [1,2,3])/3
            comparisons.append(dict(reference=reference['arm'],cumulative_quality_gain=reference['cumulative_nll']-online['cumulative_nll'],
                                    first_postchange_window_gain=post_gain,reference_over_online_fit_work=reference['fit_wall_s']/online['fit_wall_s'],
                                    online_total_with_source_acquisition_s=online['fit_wall_s']+record['inherited_acquisition_wall_s'],
                                    per_regime_quality_gain=[r['mean_nll']-o['mean_nll'] for r,o in zip(reference['regimes'],online['regimes'])]))
        rows.append(dict(depth=depth,targets=online['targets'],online_cumulative_nll=online['cumulative_nll'],comparisons=comparisons))
    gate=all(c['cumulative_quality_gain']>0 and c['first_postchange_window_gain']>0 for r in rows for c in r['comparisons'] if c['reference'] in ['static_acquired_groups','frozen_acquired_policy'])
    continuation=[];seed=record['args']['seed']
    if a.continue_if_useful and gate and seed<11:
        tag=f'aws_native_shifting_stream_20261012_s{seed+1}'
        cmd=['.venv-docker/bin/python','experiments/tpp/submit.py','--slot','2','--battle','B1','--source','experiments/credit/train_native_shifting_stream.py','--result-dir','experiments/results/credit','--timeout','10800','--rss-kb','1500000','--purpose','B1/G4 unchanged-seed confirmation of positive cumulative and post-change native learning on396800unique targets','--job',f'{tag}|--stage fit --seed {seed+1} --gate experiments/results/credit/aws_native_shifting_stream_20261012_smoke.json']
        for source in record['source_sha256']:
            if source!='experiments/credit/train_native_shifting_stream.py':cmd.extend(['--dep',source])
        continuation.append(subprocess.check_output(cmd,cwd=ROOT,text=True).strip());time.sleep(1.1)
        cmd=['.venv-docker/bin/python','experiments/tpp/submit.py','--slot','2','--battle','B1','--source','experiments/credit/analyze_native_shifting_stream.py','--result-dir','experiments/results/credit','--timeout','180','--rss-kb','500000','--purpose','B1/G4 source-bound same-target changing-stream quality/work decision','--job',f'{tag}_decision|--parent experiments/results/credit/{tag}.json --continue-if-useful']
        for source in record['source_sha256']:cmd.extend(['--dep',source])
        continuation.append(subprocess.check_output(cmd,cwd=ROOT,text=True).strip())
    result=dict(status='completed',tag=a.tag,battle='B1/G4 nonstationary native learning',rows=rows,seed=seed,useful_cumulative_and_adaptation_gate=gate,continuation=continuation,
                source_sha256={s:hashlib.sha256((ROOT/s).read_bytes()).hexdigest() for s in sorted(sources)},input_sha256={a.parent:hashlib.sha256(path.read_bytes()).hexdigest()},wall_s=time.monotonic()-start,
                scope='Identical unique target stream/order and constructed shifts, online vs frozen/static/exactnativeAdam. Positive gains favor online; wall ratios>1 favor online. Acquisition separately charged. Never compare cumulative losses across unequal regime prefixes as matched-quality work. Actual actor exact backprop retained; no classical architecture/Transformer scaling or natural-drift claim.')
    with (ROOT/'experiments/results/credit'/f'{a.tag}.json').open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps(result))

if __name__=='__main__':main()
