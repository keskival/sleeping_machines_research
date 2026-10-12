"""R1 source-bound future-learning returns at two acquired-state scales."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]


def stats(values):
    return {'mean': statistics.mean(values), 'sample_sd': statistics.stdev(values)}


def analyze():
    inputs = {}
    levels = []
    for larger in (False, True):
        rows = []
        for seed in (6, 7, 8):
            stem = 'aws_integrated_common_prior_512k_decision' if larger else 'aws_integrated_common_prior_decision'
            path = ROOT / f'experiments/results/token_language/{stem}_20261012_s{seed}_v1.json'
            blob = path.read_bytes()
            d = json.loads(blob)
            assert d['status'] == 'completed' and d['prefix_curve_error'] == 0
            for name, digest in {**d['source_sha256'], **d['input_sha256']}.items():
                assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest, name
            inputs[str(path.relative_to(ROOT))] = hashlib.sha256(blob).hexdigest()
            raw = [json.loads((ROOT / name).read_text()) for name in d['input_sha256']]
            replay = next(x for x in raw if x['args']['exposure'] == 'repeated')
            fresh = next(x for x in raw if x['args']['exposure'] == 'fresh')
            branch = fresh['args']['steps'] // 2
            rc = {x['step']: x for x in replay['curve']}
            fc = {x['step']: x for x in fresh['curve']}
            assert rc.keys() == fc.keys()
            assert rc[branch]['learner_fingerprint'] == fc[branch]['learner_fingerprint'] == d['shared_learner_fingerprint']
            horizon = [{'post_branch_updates': step - branch,
                        'fresh_over_replay': rc[step]['dev_nll'] - fc[step]['dev_nll'],
                        'fresh_over_stop': fc[branch]['dev_nll'] - fc[step]['dev_nll']}
                       for step in sorted(fc) if step > branch]
            rows.append({'seed': seed, 'acquired_state_sha256': d['shared_learner_fingerprint'],
                         'horizon_returns': horizon,
                         'replay_training_s': d['rows'][0]['training_wall_s'],
                         'fresh_training_s': d['rows'][1]['training_wall_s']})
        levels.append({'prior_tokens': 262144 if larger else 65536,
                       'fitting_targets': 524272 if larger else 131056,
                       'final_fresh_over_replay': stats([r['horizon_returns'][-1]['fresh_over_replay'] for r in rows]),
                       'final_fresh_over_stop': stats([r['horizon_returns'][-1]['fresh_over_stop'] for r in rows]),
                       'fresh_replay_training_ratio': stats([r['fresh_training_s']/r['replay_training_s'] for r in rows]),
                       'rows': rows})
    return {'status': 'completed', 'battle': 'R1', 'levels': levels,
            'input_sha256': inputs,
            'source_sha256': {'experiments/analyze_common_prior_scaling.py': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
            'scope': 'Three seeds per exposure level, common acquired state within every pair, shared private 2040-target DEV. Fixed final endpoint, no best-checkpoint selection. Intermediate horizons preserve signs of future-learning returns. Seeds are learner replications, not independent policy-training contexts. Replay/fresh targets match; stopping uses fewer updates. Two scales do not identify a scaling exponent or crossover. Full training CPU wall is measured; future policy acquisition and audit costs are not implied by these fits.'}


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--tag', required=True)
    args = p.parse_args()
    result = analyze()
    out = ROOT / f'experiments/results/diagnostics/{args.tag}.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('x') as f:
        json.dump(result, f, indent=2)
        f.write('\n')
    print(json.dumps(result), flush=True)
