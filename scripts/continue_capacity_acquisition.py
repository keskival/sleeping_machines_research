"""Await replay cohort, then admit one bounded capacity development pipeline."""
import json
import subprocess
from pathlib import Path
from continue_window_replay import ROOT, admit, sha, wait_result


def main():
    gates=['aws_window_replay_fit_gate_20261012_v1']+[f'aws_window_replay_{size}_gate_20261012_s{seed}_v1'
        for seed in (6,7,8) for size in ('128k','512k') if (seed,size)!=(6,'128k')]
    for name in gates:
        record=wait_result(ROOT/f'experiments/results/diagnostics/{name}.json')
        assert record['gate']=='PASS',record
        for source,digest in record['source_sha256'].items():assert sha(ROOT/source)==digest,source
    original=wait_result(ROOT/'experiments/results/diagnostics/aws_window_replay_contract_20261012_v2.json')
    sources=dict(original['source_sha256']);sources.pop('experiments/credit/check_window_replay.py',None)
    checker='experiments/credit/check_capacity_window_replay.py';sources[checker]=sha(ROOT/checker)
    tag='aws_capacity8_replay_contract_20261012_v1';result=f'experiments/results/diagnostics/{tag}.json'
    admit(tag,checker,f'--tag {tag}',result,sources,[],True)
    contract=wait_result(ROOT/result)
    assert len(contract['rows'])==4 and all(r['feature_teacher_gradient_parameter_error']==0 for r in contract['rows'])
    driver='experiments/window_replay_token_language.py'
    saved=wait_result(ROOT/'experiments/results/token_language/aws_integrated_window_replay_128k_20261012_s6_v1.json')
    sources=dict(saved['source_sha256'])
    smoke='aws_integrated_capacity8_smoke_20261012_s6_v1'
    template=(ROOT/'experiments/queue/r1/aws_integrated_window_replay_128k_20261012_s6_v1/aws_integrated_window_replay_128k_20261012_s6_v1.txt').read_text().splitlines()[1]
    arguments=template.split(driver+' ',1)[1].replace('aws_integrated_window_replay_128k_20261012_s6_v1',smoke).replace('--pool 4','--pool 8').replace('--steps 256','--steps 16').replace('--eval-every 64','--eval-every 8')
    result=f'experiments/results/token_language/{smoke}.json'
    admit(smoke,driver,arguments,result,sources,[dict(result=f'experiments/results/diagnostics/{tag}.json')],True)
    small=wait_result(ROOT/result)
    assert small['max_rss_kb']*1.5<1500000 and small['presentations_total']==8192
    for size in ('128k','512k'):
        reference=f'aws_integrated_window_replay_{size}_20261012_s6_v1'
        template=(ROOT/f'experiments/queue/r1/{reference}/{reference}.txt').read_text().splitlines()[1]
        tag=f'aws_integrated_capacity8_{size}_20261012_s6_v1'
        arguments=template.split(driver+' ',1)[1].replace(reference,tag).replace('--pool 4','--pool 8')
        result=f'experiments/results/token_language/{tag}.json'
        admit(tag,driver,arguments,result,sources,[dict(result=f'experiments/results/token_language/{smoke}.json')],True)
        wait_result(ROOT/result)  # No small-quality veto of the larger registered point.
    analysis='experiments/analyze_capacity_acquisition.py';sources[analysis]=sha(ROOT/analysis)
    tag='aws_capacity_acquisition_20261012_v1';result=f'experiments/results/diagnostics/{tag}.json'
    admit(tag,analysis,f'--tag {tag}',result,sources,[],False)
    wait_result(ROOT/result)
    subprocess.run([str(ROOT/'.venv-docker/bin/python'),str(ROOT/'scripts/publish_capacity_acquisition.py')],cwd=ROOT,check=True)
    print('capacity development complete; source-bound decision '+result,flush=True)


if __name__=='__main__':main()
