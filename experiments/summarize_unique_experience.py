"""Read-only three-seed R1 development evidence; no model or data scoring."""
import hashlib
import json
from pathlib import Path
import statistics

ROOT=Path(__file__).resolve().parents[1]


def summarize():
    records=[]; inputs={}
    for seed in (6,7,8):
        path=ROOT/'experiments/results/token_language'/f'aws_integrated_p24_unique_stream_decision_20261012_s{seed}_v1.json'
        blob=path.read_bytes();d=json.loads(blob);assert d['status']=='completed'
        for name,digest in d['source_sha256'].items():
            assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,name
        for name,digest in d['input_sha256'].items():
            p=ROOT/name
            if p.suffix=='.pt' and not p.exists():continue
            assert hashlib.sha256(p.read_bytes()).hexdigest()==digest,name
        assert {r['fitting_targets'] for r in d['rows']}=={131056}
        assert {r['updates'] for r in d['rows']}=={256}
        assert [r['unique_tokens'] for r in d['rows']]==[65536,131064]
        records.append(d);inputs[str(path.relative_to(ROOT))]=hashlib.sha256(blob).hexdigest()
    def stats(xs):return dict(mean=statistics.mean(xs),sample_sd=statistics.stdev(xs))
    return dict(status='completed',battle='R1',seeds=[6,7,8],
        selected_nll={arm:stats([d['rows'][i]['selected_dev_nll'] for d in records]) for i,arm in enumerate(('repeated','distinct'))},
        paired_quality_gain=stats([d['quality_gain'] for d in records]),
        paired_learning_gain_change=stats([d['learning_gain_change'] for d in records]),
        paired_context_gain_change=stats([d['context_gain_change'] for d in records]),
        training_wall_s={arm:stats([d['rows'][i]['training_wall_s'] for d in records]) for i,arm in enumerate(('repeated','distinct'))},
        positive_quality_seeds=sum(d['quality_gain']>0 for d in records),
        input_sha256=inputs,source_sha256={'experiments/summarize_unique_experience.py':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
        scope='Three-seed development comparison within the integrated family; identical131056fitting targets/256updates/private2040-target DEV. Distinct data, lane partition, corpus-wrap resets and TRAIN-frequency initialization differ. Mean±sampleSD; contextual comparator fixed from causal TRAIN features. Measured CPU training wall, not audited fitting FLOPs or an incumbent comparison.')


if __name__=='__main__':print(json.dumps(summarize(),indent=2))
