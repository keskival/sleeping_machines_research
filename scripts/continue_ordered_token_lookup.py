"""After current acquisition evidence, gate ordered lookup on exact learning."""
import time
import continue_window_replay as queue
from continue_fresh_scaling import admit
ROOT=queue.ROOT
queue.DEADLINE=time.monotonic()+14*3600


def main():
    queue.wait_result(ROOT/'experiments/results/diagnostics/aws_fresh_scaling_transfer_20261013_v1.json')
    base='aws_integrated_window_replay_128k_20261012_s6_v1'
    saved=queue.wait_result(ROOT/f'experiments/results/token_language/{base}.json')
    sources=dict(saved['source_sha256'])
    for source in ('experiments/ordered_token_language.py','sleeping_machines/ordered_token_episodes.py',
                   'sleeping_machines/ordered_token_lookup.py','experiments/credit/check_ordered_token_lookup.py'):
        sources[source]=queue.sha(ROOT/source)
    tag='aws_ordered_token_lookup_contract_20261013_v1';driver='experiments/credit/check_ordered_token_lookup.py'
    result=f'experiments/results/diagnostics/{tag}.json'
    admit(tag,driver,f'--tag {tag}',result,sources,[],False)
    gate=queue.wait_result(ROOT/result)
    assert all(r['feature_teacher_gradient_parameter_error']==0 for r in gate['rows'])
    tag='aws_integrated_ordered_lookup_128k_20261013_s6_v1';driver='experiments/ordered_token_language.py'
    command=(ROOT/f'experiments/queue/r1/{base}/{base}.txt').read_text().splitlines()[1]
    args=command.split('experiments/window_replay_token_language.py ',1)[1].replace(base,tag)
    result=f'experiments/results/token_language/{tag}.json'
    sources.update(gate['source_sha256'])
    admit(tag,driver,args,result,sources,[dict(result='experiments/results/diagnostics/aws_ordered_token_lookup_contract_20261013_v1.json')],True)
    queue.wait_result(ROOT/result)
    driver='experiments/analyze_window_replay.py';sources[driver]=queue.sha(ROOT/driver)
    check='aws_ordered_token_lookup_fit_gate_20261013_v1';result=f'experiments/results/diagnostics/{check}.json'
    admit(check,driver,f'--tag {check} --reference {base} --candidate {tag}',result,sources,[],False)
    record=queue.wait_result(ROOT/result)
    print('Ordered-content complete fit gate '+record['gate']+'; retain only exact learning and lower full training time',flush=True)


if __name__=='__main__':main()
