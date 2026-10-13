"""R1 bounded frozen-state history diagnostic after exact ordered confirmations."""
import time,subprocess
import continue_window_replay as queue
from continue_fresh_scaling import admit
ROOT=queue.ROOT
queue.DEADLINE=time.monotonic()+3*3600

def main():
    for size,seed in (('512k',6),('128k',7),('512k',7),('128k',8),('512k',8)):
        d=queue.wait_result(ROOT/f'experiments/results/diagnostics/aws_ordered_lookup_{size}_gate_20261013_s{seed}_v3.json')
        assert d['gate']=='PASS' and d['exact_learning_curve']
    history=queue.wait_result(ROOT/'experiments/results/diagnostics/aws_acquired_history_transfer_20261013_v1.json')
    assert len(history['paired'])==3
    transfer=queue.wait_result(ROOT/'experiments/results/diagnostics/aws_fresh_scaling_transfer_20261013_v1.json')
    sources=dict(transfer['source_sha256']);driver='experiments/check_acquired_short_history_transfer.py';sources[driver]=queue.sha(ROOT/driver)
    for name,digest in sources.items():assert queue.sha(ROOT/name)==digest,name
    tag='aws_acquired_short_history_transfer_20261013_v1';result=f'experiments/results/diagnostics/{tag}.json'
    admit(tag,driver,f'--tag {tag}',result,sources,[],True)
    queue.wait_result(ROOT/result)
    subprocess.run([str(ROOT/'.venv-docker/bin/python'),str(ROOT/'scripts/publish_acquired_short_history_transfer.py')],cwd=ROOT,check=True)
    print('Frozen history intervention complete and signed results published',flush=True)

if __name__=='__main__':main()
