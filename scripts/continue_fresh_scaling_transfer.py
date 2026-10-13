"""After completed three-seed acquisition, score frozen final weights once."""
import json
import subprocess
import time
import continue_window_replay as queue
from continue_fresh_scaling import admit
ROOT=queue.ROOT
queue.DEADLINE=time.monotonic()+12*3600


def main():
    gate=queue.wait_result(ROOT/'experiments/results/diagnostics/aws_fresh_scaling_8m_confirmation_20261013_s8_v1.json')
    assert len(gate['rows'])==3
    sources={};requires=[]
    for seed in (6,7,8):
        for size in ('512k','8m'):
            tag=(f'aws_integrated_window_replay_512k_20261012_s{seed}_v1' if size=='512k' else
                f'aws_integrated_fresh_scaling_8m_pool4_{"20261012" if seed==6 else "20261013"}_s{seed}_v1')
            result=f'experiments/results/token_language/{tag}.json'
            sources.update(queue.wait_result(ROOT/result)['source_sha256']);requires.append(dict(result=result))
    driver='experiments/check_fresh_scaling_transfer.py';sources[driver]=queue.sha(ROOT/driver)
    tag='aws_fresh_scaling_transfer_20261013_v1';result=f'experiments/results/diagnostics/{tag}.json'
    admit(tag,driver,f'--tag {tag}',result,sources,requires,True)
    queue.wait_result(ROOT/result)
    subprocess.run([str(ROOT/'.venv-docker/bin/python'),str(ROOT/'scripts/publish_fresh_scaling_transfer.py')],cwd=ROOT,check=True)
    print('Frozen checkpoint transfer audit completed; source-bound packet ready for reporting',flush=True)


if __name__=='__main__':main()
