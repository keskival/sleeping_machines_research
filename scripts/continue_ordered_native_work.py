"""Admit native-work comparison only after exact-learning whole-fit PASS."""
import time
import continue_window_replay as queue
from continue_fresh_scaling import admit
ROOT=queue.ROOT
queue.DEADLINE=time.monotonic()+3*3600


def main():
    gate=queue.wait_result(ROOT/'experiments/results/diagnostics/aws_ordered_token_lookup_fit_gate_20261013_v2.json');assert gate['gate']=='PASS'
    saved=queue.wait_result(ROOT/'experiments/results/token_language/aws_integrated_ordered_lookup_128k_20261013_s6_v2.json')
    old=queue.wait_result(ROOT/'experiments/results/token_language/aws_native_work_scalar_probe_20261013_v1.json')
    sources=dict(saved['source_sha256']);sources.update(old['source_sha256'])
    driver='experiments/profile_ordered_native_work.py';sources[driver]=queue.sha(ROOT/driver)
    tag='aws_ordered_native_work_probe_20261013_v1';result=f'experiments/results/token_language/{tag}.json'
    command=(ROOT/'experiments/queue/r1/aws_native_work_scalar_probe_20261013_v1/aws_native_work_scalar_probe_20261013_v1.txt').read_text().splitlines()[1]
    args=command.split('experiments/native_training_work_v2.py ',1)[1].replace('aws_native_work_scalar_probe_20261013_v1',tag)
    admit(tag,driver,args,result,sources,[],True);queue.wait_result(ROOT/result)
    driver='experiments/analyze_ordered_native_work.py';sources[driver]=queue.sha(ROOT/driver)
    tag='aws_ordered_native_work_decision_20261013_v1';result=f'experiments/results/diagnostics/{tag}.json'
    admit(tag,driver,f'--tag {tag}',result,sources,[],False);queue.wait_result(ROOT/result)
    print('Ordered-gradient native-work comparison completed',flush=True)


if __name__=='__main__':main()
