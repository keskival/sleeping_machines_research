"""Confirm the completed 8M integrated learning frontier with two more seeds."""
import subprocess
import time
import continue_window_replay as queue
from continue_fresh_scaling import admit
ROOT=queue.ROOT
queue.DEADLINE=time.monotonic()+10*3600


def main():
    gate=queue.wait_result(ROOT/'experiments/results/diagnostics/aws_fresh_scaling_8m_decision_20261012_v1.json')
    row=gate['rows'][0]
    if not (row['pool']==4 and row['final_learning_gain']>.05 and row['ms_per_target']<1.5 and row['exact_recorded_acquisition_prefix']):
        print('Confirmation stopped: completed 8M quality/work gate failed',flush=True);return
    for seed in (7,8):
        reference=f'aws_integrated_window_replay_512k_20261012_s{seed}_v1'
        saved=queue.wait_result(ROOT/f'experiments/results/token_language/{reference}.json')
        sources=saved['source_sha256']
        for name,digest in sources.items():assert queue.sha(ROOT/name)==digest,name
        driver='experiments/window_replay_token_language.py'
        command=(ROOT/f'experiments/queue/r1/{reference}/{reference}.txt').read_text().splitlines()[1]
        tag=f'aws_integrated_fresh_scaling_8m_pool4_20261013_s{seed}_v1'
        args=command.split(driver+' ',1)[1].replace(reference,tag).replace('--steps 1024','--steps 16384')
        result=f'experiments/results/token_language/{tag}.json'
        admit(tag,driver,args,result,sources,[dict(result='experiments/results/diagnostics/aws_fresh_scaling_8m_decision_20261012_v1.json')],True)
        queue.wait_result(ROOT/result)
        driver='experiments/analyze_fresh_scaling_confirmation.py';sources=dict(sources);sources[driver]=queue.sha(ROOT/driver)
        tag=f'aws_fresh_scaling_8m_confirmation_20261013_s{seed}_v1';result=f'experiments/results/diagnostics/{tag}.json'
        admit(tag,driver,f'--tag {tag} --through-seed {seed}',result,sources,[],False)
        record=queue.wait_result(ROOT/result)
        subprocess.run([str(ROOT/'.venv-docker/bin/python'),str(ROOT/'scripts/publish_fresh_scaling_confirmation.py'),'--through-seed',str(seed)],cwd=ROOT)
        if not record['continue_confirmation']:
            print('Further confirmation stopped: acquired-learning/resource gate failed',flush=True);return


if __name__=='__main__':main()
