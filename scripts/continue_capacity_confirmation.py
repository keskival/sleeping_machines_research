"""Confirm a useful larger capacity member; no first-point quality veto."""
import json
import subprocess
from continue_window_replay import ROOT, admit, sha, wait_result


def main():
    decision=wait_result(ROOT/'experiments/results/diagnostics/aws_capacity_acquisition_20261012_v1.json')
    for source,digest in decision['source_sha256'].items():assert sha(ROOT/source)==digest,source
    large=decision['levels'][-1]
    if large['pool8_selected_gain']<=0 or large['training_wall_ratio']>1.05:
        print('No extra confirmation fits: larger-point quality/5% training-premium gate fails. Acquired-state transport contract continues separately.',flush=True)
        return
    base='aws_integrated_capacity8_512k_20261012_s6_v1'
    record=wait_result(ROOT/f'experiments/results/token_language/{base}.json')
    sources=dict(record['source_sha256']);driver='experiments/window_replay_token_language.py'
    template=(ROOT/f'experiments/queue/r1/{base}/{base}.txt').read_text().splitlines()[1]
    for seed in (7,8):
        tag=f'aws_integrated_capacity8_512k_20261012_s{seed}_v1'
        arguments=template.split(driver+' ',1)[1]
        assert '--seed 6' in arguments
        arguments=arguments.replace(base,tag).replace('--seed 6',f'--seed {seed}')
        result=f'experiments/results/token_language/{tag}.json'
        admit(tag,driver,arguments,result,sources,[dict(result='experiments/results/diagnostics/aws_capacity_acquisition_20261012_v1.json')],True)
        wait_result(ROOT/result)
    analysis='experiments/analyze_capacity_confirmation.py';sources[analysis]=sha(ROOT/analysis)
    tag='aws_capacity_confirmation_20261012_v1';result=f'experiments/results/diagnostics/{tag}.json'
    admit(tag,analysis,f'--tag {tag}',result,sources,[],False)
    wait_result(ROOT/result)
    subprocess.run([str(ROOT/'.venv-docker/bin/python'),str(ROOT/'scripts/publish_capacity_acquisition.py')],cwd=ROOT,check=True)


if __name__=='__main__':main()
