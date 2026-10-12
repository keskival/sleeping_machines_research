"""Guarded acquired-state contract after the registered capacity fits finish."""
import json
from continue_window_replay import ROOT, admit, sha, wait_result


def main():
    completed=wait_result(ROOT/'experiments/results/diagnostics/aws_capacity_acquisition_20261012_v1.json')
    for source,digest in completed['source_sha256'].items():assert sha(ROOT/source)==digest,source
    original=wait_result(ROOT/'experiments/results/diagnostics/aws_window_replay_contract_20261012_v2.json')
    sources=dict(original['source_sha256']);sources.pop('experiments/credit/check_window_replay.py',None)
    checker='experiments/credit/check_capacity_optimizer_reuse.py'
    for source in (checker,'sleeping_machines/expanded_token_core.py','sleeping_machines/optimizer_expansion.py'):
        sources[source]=sha(ROOT/source)
    tag='aws_capacity_optimizer_reuse_20261012_v1';result=f'experiments/results/diagnostics/{tag}.json'
    admit(tag,checker,f'--tag {tag}',result,sources,
        [dict(result='experiments/results/diagnostics/aws_capacity_acquisition_20261012_v1.json')],True)
    receipt=wait_result(ROOT/result)
    assert len(receipt['rows'])==2 and all(r['retained_parameter_error']==0 for r in receipt['rows'])
    print(json.dumps(receipt['rows']),flush=True)
    print('Acquired-learner capacity transport contract complete; no warm-expanded quality fit admitted.',flush=True)


if __name__=='__main__':main()
