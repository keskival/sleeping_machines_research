"""R1 hardware-work probe parity and independent local-bank update contract."""
import time
import continue_window_replay as queue
from continue_fresh_scaling import admit
ROOT=queue.ROOT
queue.DEADLINE=time.monotonic()+8*3600


def main():
    base='aws_integrated_window_replay_128k_20261012_s6_v1'
    saved=queue.wait_result(ROOT/f'experiments/results/token_language/{base}.json')
    sources=dict(saved['source_sha256'])
    for name,digest in sources.items():assert queue.sha(ROOT/name)==digest,name
    line=(ROOT/f'experiments/queue/r1/{base}/{base}.txt').read_text().splitlines()[1]
    template=line.split('experiments/window_replay_token_language.py ',1)[1].replace('--steps 256','--steps 16').replace('--eval-every 64','--eval-every 8')
    for kind,driver in [('reference','experiments/window_replay_token_language.py'),('probe','experiments/native_training_work.py')]:
        tag=f'aws_native_work_{kind}_20261013_v1';result=f'experiments/results/token_language/{tag}.json'
        deps=dict(sources)
        if kind=='probe':
            for source in (driver,'experiments/credit/token_flop_accounting.py'):deps[source]=queue.sha(ROOT/source)
        admit(tag,driver,template.replace(base,tag),result,deps,[],True)
        record=queue.wait_result(ROOT/result)
        sources.update(record['source_sha256'])
    driver='experiments/check_native_work_probe.py';sources[driver]=queue.sha(ROOT/driver)
    tag='aws_native_work_accounting_20261013_v1';result=f'experiments/results/diagnostics/{tag}.json'
    admit(tag,driver,f'--tag {tag}',result,sources,[],False);queue.wait_result(ROOT/result)
    driver='experiments/credit/check_local_credit_banks.py'
    for source in (driver,'sleeping_machines/learning/local_credit_banks.py','experiments/credit/check_window_replay.py','experiments/common_prior_token_language.py'):
        sources[source]=queue.sha(ROOT/source)
    tag='aws_local_credit_banks_contract_20261013_v1';result=f'experiments/results/diagnostics/{tag}.json'
    admit(tag,driver,f'--tag {tag}',result,sources,[],False);queue.wait_result(ROOT/result)
    print('Hardware-work and local-bank contracts complete: source-bound receipts ready',flush=True)


if __name__=='__main__':main()
