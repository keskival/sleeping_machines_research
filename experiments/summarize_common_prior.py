"""Read-only confirmed future-learning contrasts from paired acquired states."""
import hashlib
import json
from pathlib import Path
import statistics
ROOT=Path(__file__).resolve().parents[1]


def summarize():
    records=[];inputs={}
    for seed in (6,7,8):
        p=ROOT/'experiments/results/token_language'/f'aws_integrated_common_prior_decision_20261012_s{seed}_v1.json'
        blob=p.read_bytes();d=json.loads(blob);assert d['status']=='completed'
        assert d['prefix_curve_error']==0 and d['development_gate_passed']
        for name,digest in {**d['source_sha256'],**d['input_sha256']}.items():
            assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,name
        records.append(d);inputs[str(p.relative_to(ROOT))]=hashlib.sha256(blob).hexdigest()
    def stats(xs):return dict(mean=statistics.mean(xs),sample_sd=statistics.stdev(xs))
    return dict(status='completed',battle='R1',seeds=[6,7,8],
        fresh_over_shared_branch_stop=stats([d['rows'][1]['branch_dev_nll']-d['rows'][1]['post_branch_dev_nll'] for d in records]),
        fresh_over_post_branch_replay=stats([d['post_branch_quality_gain'] for d in records]),
        selected_model_gain=stats([d['selected_quality_gain'] for d in records]),
        selected_fresh_nll=stats([d['rows'][1]['selected_dev_nll'] for d in records]),
        training_wall_s={arm:stats([d['rows'][i]['training_wall_s'] for d in records]) for i,arm in enumerate(('replayed','fresh'))},
        acquired_state_sha256=[d['shared_learner_fingerprint'] for d in records],
        input_sha256=inputs,source_sha256={'experiments/summarize_common_prior.py':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
        scope='Three-seed within-family development evidence, mean±sampleSD. Identical prior and acquired-state fingerprint within each pair; same131056targets/256updates and private2040-target DEV. Stop refers to the actual branch state after128updates; selected-model comparison allows all scheduled checkpoints. Extra fresh learning costs128updates relative to stopping; replay/fresh totals have equal targets. No trained allocation policy or external-architecture claim.')


if __name__=='__main__':print(json.dumps(summarize(),indent=2))
