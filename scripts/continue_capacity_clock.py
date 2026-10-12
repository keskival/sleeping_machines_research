"""Contract -> resource fit -> two-size integrated clock development."""
import subprocess
from continue_window_replay import ROOT, admit, sha, wait_result


def main():
    gate=wait_result(ROOT/'experiments/results/diagnostics/aws_capacity_clock_contract_20261012_v2.json')
    assert len(gate['rows'])==8 and all(r['feature_teacher_gradient_parameter_error']==0 for r in gate['rows'])
    for source,digest in gate['source_sha256'].items():assert sha(ROOT/source)==digest,source
    driver='experiments/capacity_clock_token_language.py'
    base='aws_integrated_capacity8_128k_20261012_s6_v1'
    saved=wait_result(ROOT/f'experiments/results/token_language/{base}.json')
    sources=dict(saved['source_sha256']);sources.pop('experiments/window_replay_token_language.py')
    for source in (driver,'sleeping_machines/capacity_clock_episodes.py'):sources[source]=gate['source_sha256'][source]
    template=(ROOT/f'experiments/queue/r1/{base}/{base}.txt').read_text().splitlines()[1]
    tag='aws_integrated_clock_capacity8_smoke_20261012_s6_v1'
    args=template.split('experiments/window_replay_token_language.py ',1)[1].replace(base,tag).replace('--steps 256','--steps 16').replace('--eval-every 64','--eval-every 8')
    result=f'experiments/results/token_language/{tag}.json'
    admit(tag,driver,args,result,sources,[dict(result='experiments/results/diagnostics/aws_capacity_clock_contract_20261012_v2.json')],True)
    small=wait_result(ROOT/result);assert small['max_rss_kb']*1.5<1500000 and small['presentations_total']==8192
    for size in ('128k','512k'):
        base=f'aws_integrated_capacity8_{size}_20261012_s6_v1'
        template=(ROOT/f'experiments/queue/r1/{base}/{base}.txt').read_text().splitlines()[1]
        tag=f'aws_integrated_clock_capacity8_{size}_20261012_s6_v1'
        args=template.split('experiments/window_replay_token_language.py ',1)[1].replace(base,tag)
        result=f'experiments/results/token_language/{tag}.json'
        admit(tag,driver,args,result,sources,[dict(result='experiments/results/token_language/'+small['args']['tag']+'.json')],True)
        wait_result(ROOT/result)
    # The analysis binds both saved drivers as well as the calibrated one.
    sources.update(saved['source_sha256'])
    analysis='experiments/analyze_capacity_clock.py';sources[analysis]=sha(ROOT/analysis)
    tag='aws_capacity_clock_decision_20261012_v1';result=f'experiments/results/diagnostics/{tag}.json'
    admit(tag,analysis,f'--tag {tag}',result,sources,[],False)
    wait_result(ROOT/result)
    subprocess.run([str(ROOT/'.venv-docker/bin/python'),str(ROOT/'scripts/publish_capacity_clock.py')],cwd=ROOT,check=True)


if __name__=='__main__':main()
