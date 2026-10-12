"""Source-bound exact-learning and complete CPU-work gate for cached replay."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--tag', required=True)
    p.add_argument('--reference', required=True)
    p.add_argument('--candidate', required=True)
    a = p.parse_args()
    inputs, sources, records = {}, {}, []
    for tag in (a.reference, a.candidate):
        path = ROOT / f'experiments/results/token_language/{tag}.json'
        blob = path.read_bytes(); d = json.loads(blob)
        assert d['status'] == 'completed'
        inputs[str(path.relative_to(ROOT))] = hashlib.sha256(blob).hexdigest()
        for source, digest in d['source_sha256'].items():
            assert hashlib.sha256((ROOT/source).read_bytes()).hexdigest() == digest, source
            assert source not in sources or sources[source] == digest
            sources[source] = digest
        records.append(d)
    reference, candidate = records
    assert {k:v for k,v in reference['args'].items() if k != 'tag'} == {k:v for k,v in candidate['args'].items() if k != 'tag'}
    for field in ('train_sha256', 'dev_sha256'):
        assert reference['identity'][field] == candidate['identity'][field]
    assert reference['presentations_total'] == candidate['presentations_total']
    def canonical(value):
        if isinstance(value, dict):
            return {k:canonical(v) for k,v in value.items() if k not in ('wall_s', 'replay_length', 'skipped_prefix')}
        if isinstance(value, list): return [canonical(v) for v in value]
        return value
    exact = canonical(reference['curve']) == canonical(candidate['curve'])
    ratio = candidate['train_wall_s_total'] / reference['train_wall_s_total']
    result = dict(status='completed', battle='R1', input_sha256=inputs,
        exact_learning_curve=exact, gate='PASS' if exact and ratio < 1 else 'FAIL',
        targets=candidate['presentations_total'], reference_training_s=reference['train_wall_s_total'],
        candidate_training_s=candidate['train_wall_s_total'], training_wall_ratio=ratio,
        training_time_reduction_percent=100*(1-ratio),
        reference_nll=reference['best_dev_nll'], candidate_nll=candidate['best_dev_nll'],
        scope='Single-seed same-source-core/data/settings own-model execution comparison. Entire recorded learning/teacher/gradient curves must match exactly except replay-work counters and wall time. Complete training wall includes factual learning, teacher replay, clipping and optimizer; no FLOP, energy, external-model or learned-estimator efficiency claim.')
    source = str(Path(__file__).relative_to(ROOT))
    sources[source] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    result['source_sha256'] = sources
    out = ROOT / f'experiments/results/diagnostics/{a.tag}.json'
    with out.open('x') as f: json.dump(result, f, indent=2); f.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('input_sha256','source_sha256')}), flush=True)


if __name__ == '__main__': main()
