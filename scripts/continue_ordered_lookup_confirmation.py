"""Exact-learning confirmation of retained ordered lookup at saved larger data."""
import subprocess,time
import continue_window_replay as queue
from continue_fresh_scaling import admit
ROOT=queue.ROOT
queue.DEADLINE=time.monotonic()+4*3600


def main():
    gate=queue.wait_result(ROOT/'experiments/results/diagnostics/aws_ordered_token_lookup_fit_gate_20261013_v2.json');assert gate['gate']=='PASS'
    native=queue.wait_result(ROOT/'experiments/results/diagnostics/aws_ordered_native_work_decision_20261013_v1.json')
    assert native['exact_recorded_learning'] and native['removed_work']['declared_fit_scalar_arithmetic']>0
    for size,seed in (('512k',6),('128k',7),('512k',7),('128k',8),('512k',8)):
        reference=f'aws_integrated_window_replay_{size}_20261012_s{seed}_v1'
        old=queue.wait_result(ROOT/f'experiments/results/token_language/{reference}.json')
        sources=dict(old['source_sha256']);sources.update(gate['source_sha256'])
        for name,digest in sources.items():assert queue.sha(ROOT/name)==digest,name
        driver='experiments/ordered_token_language.py'
        tag=f'aws_integrated_ordered_lookup_{size}_20261013_s{seed}_v3'
        command=(ROOT/f'experiments/queue/r1/{reference}/{reference}.txt').read_text().splitlines()[1]
        args=command.split('experiments/window_replay_token_language.py ',1)[1].replace(reference,tag)
        result=f'experiments/results/token_language/{tag}.json'
        admit(tag,driver,args,result,sources,[],True);d=queue.wait_result(ROOT/result);sources.update(d['source_sha256'])
        driver='experiments/analyze_window_replay.py';sources[driver]=queue.sha(ROOT/driver)
        check=f'aws_ordered_lookup_{size}_gate_20261013_s{seed}_v3';result=f'experiments/results/diagnostics/{check}.json'
        admit(check,driver,f'--tag {check} --reference {reference} --candidate {tag}',result,sources,[],False)
        proof=queue.wait_result(ROOT/result)
        if proof['gate']!='PASS':print('STOP: exact-learning/lower-time confirmation failed '+check,flush=True);return
        print('PASS '+check,flush=True)
    print('Ordered lookup confirmations complete; all saved quality trajectories exact',flush=True)


if __name__=='__main__':main()
