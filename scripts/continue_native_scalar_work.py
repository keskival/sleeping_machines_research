"""Bounded native arithmetic-coverage refinement and exact learning gate."""
import time
import continue_window_replay as queue
from continue_fresh_scaling import admit
ROOT=queue.ROOT
queue.DEADLINE=time.monotonic()+6*3600


def main():
    old=queue.wait_result(ROOT/'experiments/results/token_language/aws_native_work_probe_20261013_v1.json')
    sources=dict(old['source_sha256'])
    driver='experiments/native_training_work_v2.py'
    for name in (driver,'experiments/credit/native_scalar_work.py'):sources[name]=queue.sha(ROOT/name)
    tag='aws_native_work_scalar_probe_20261013_v1';result=f'experiments/results/token_language/{tag}.json'
    command=(ROOT/'experiments/queue/r1/aws_native_work_probe_20261013_v1/aws_native_work_probe_20261013_v1.txt').read_text().splitlines()[1]
    args=command.split('experiments/native_training_work.py ',1)[1].replace('aws_native_work_probe_20261013_v1',tag)
    admit(tag,driver,args,result,sources,[],True);d=queue.wait_result(ROOT/result)
    sources.update(d['source_sha256']);driver='experiments/check_native_work_scalar_probe.py';sources[driver]=queue.sha(ROOT/driver)
    tag='aws_native_work_scalar_accounting_20261013_v1';result=f'experiments/results/diagnostics/{tag}.json'
    admit(tag,driver,f'--tag {tag}',result,sources,[],False);queue.wait_result(ROOT/result)
    print('Scalar/optimizer coverage refined; exact-learning gate passes',flush=True)


if __name__=='__main__':main()
