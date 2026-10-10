#!/usr/bin/env python3
"""B5 validation-only decision from completed, source-bound v5/v6 results."""
import argparse
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize(spec):
    records = {}
    inputs = {}
    for label, binding in spec['inputs'].items():
        path = ROOT / binding['result']
        record = json.loads(path.read_text())
        if record.get('status') != 'completed':
            raise ValueError(f'{label}: incomplete result')
        for source, digest in binding['source_sha256'].items():
            if record.get('source_sha256', {}).get(source) != digest:
                raise ValueError(f'{label}: source mismatch: {source}')
        records[label] = record
        inputs[label] = dict(path=binding['result'], sha256=sha(path))

    heuristic = records['heuristic']
    if heuristic['queries'] != 730784 or heuristic['args']['max_val'] != 0:
        raise ValueError('Heuristic must cover full validation')
    baseline = heuristic['scorers']['pop30']['mrr']
    rows = {}
    for label in ('A', 'B', 'C', 'D'):
        record = records[label]
        args = record['args']
        if (record['dataset'] != 'tgbl-review' or record['trained_events'] != 3413837
                or args['max_train'] != 0 or args['max_val'] != 0 or args['epochs'] != 1
                or args['seed'] != 0 or args['neg'] != 20
                or record['model_switches'] != dict(identity='none', pop_residual=True)
                or record['hard_negatives']['on'] != (label in ('B', 'D'))
                or bool(record.get('logq', False)) != (label in ('C', 'D'))):
            raise ValueError(f'{label}: incompatible arm settings')
        selected = record['history'][record['best_epoch']]
        profile = selected['query_profile']
        if (selected['val_queries'] != 730784
                or {k: v['queries'] for k, v in profile.items()} != heuristic['class_counts']):
            raise ValueError(f'{label}: validation coverage mismatch')
        mrr = record['val_mrr']
        if not math.isfinite(mrr) or not 0 <= mrr <= 1 or mrr != selected['val_mrr']:
            raise ValueError(f'{label}: invalid selected score')
        rows[label] = dict(val_mrr=mrr, gap_to_pop30=mrr-baseline,
                           query_profile=profile, wall_s=record['wall_s'],
                           max_rss_kb=record['max_rss_kb'], parameters=record['parameters'])
    # Same registered fit; only log-Q changes within each pair.
    for left, right in (('A', 'C'), ('B', 'D')):
        left_args = {k: v for k, v in records[left]['args'].items() if k != 'tag'}
        right_args = {k: v for k, v in records[right]['args'].items() if k != 'tag'}
        if left_args != right_args:
            raise ValueError(f'{left}/{right}: fit settings differ')
    winner = max(('C', 'D'), key=lambda label: rows[label]['val_mrr'])
    beat = rows[winner]['val_mrr'] > baseline
    return dict(
        status='completed', battle='B5 tgbl-review development', inputs=inputs,
        scope='Full VALIDATION, seed 0, one TRAIN epoch; no TEST; development evidence',
        popularity_val_mrr=baseline, arms=rows, best_logq_arm=winner,
        paired_logq_gain=dict(C_minus_A=rows['C']['val_mrr']-rows['A']['val_mrr'],
                             D_minus_B=rows['D']['val_mrr']-rows['B']['val_mrr']),
        beats_popularity=beat,
        next_decision=('Analyze residual gains by query class and confirm on validation seeds before TEST admission'
                       if beat else
                       'Diagnose sampled objective and residual calibration on TRAIN/VALIDATION; no TEST admission'),
        approximation='v6 uses end-of-batch proposal frequencies and does not correct duplicate/positive rejection; '
                      'scores test this implementation, not exact sampled-softmax equivalence',
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tag', required=True)
    parser.add_argument('--spec', required=True)
    args = parser.parse_args()
    spec_path = ROOT / args.spec
    result = summarize(json.loads(spec_path.read_text()))
    result['tag'] = args.tag
    result['source_sha256'] = {
        str(Path(__file__).resolve().relative_to(ROOT)): sha(Path(__file__).resolve()),
        args.spec: sha(spec_path),
    }
    out = ROOT / 'experiments/results/tgb' / f'{args.tag}.json'
    with out.open('x') as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
