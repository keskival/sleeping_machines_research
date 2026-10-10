#!/usr/bin/env python3
"""B1 block-credit numerical diagnosis and registered three-seed quality decision."""
import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PREFIX = 'experiments/results/credit/aws_deep_blocks_real_20261010T180155Z_smoke.json'
LOG = 'experiments/queue/logs/aws_deep_blocks_roundoff_20261010T180851Z.log'


def sha(path):
    return hashlib.sha256((ROOT/path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tag', required=True)
    parser.add_argument('--stage', choices=['diagnosis', 'quality'], required=True)
    parser.add_argument('--fit-prefix')
    args = parser.parse_args()
    path = ROOT/'experiments/results/credit'/f'{args.tag}.json'
    assert not path.exists()
    source = str(Path(__file__).resolve().relative_to(ROOT))
    if args.stage == 'diagnosis':
        prefix = json.loads((ROOT/PREFIX).read_text())
        assert prefix['status'] == 'completed' and prefix['parity_pass'] and prefix['training_speedup'] > 1.5
        rows = [json.loads(line) for line in (ROOT/LOG).read_text().splitlines() if line.startswith('{')]
        audits = [r['roundoff_audit'] for r in rows if 'roundoff_audit' in r]
        assert len(audits) == 1
        audit = audits[0]
        assert audit['controlled_gradient_scaled_error'] < 1e-9
        assert audit['controlled_mean_likelihood_error'] < 1e-9
        assert max(audit['same_incoming_state_errors'].values()) < 1e-9
        last = [r for r in rows if 'timings' in r][-1]
        result = dict(status='completed', tag=args.tag, battle='B1/R1', stage='roundoff_diagnosis',
                      identical_state_audit_pass=True, audit=audit,
                      stopped_independent_trajectory=True, last_logged_independent_batch=last,
                      prefix_training_speedup=prefix['training_speedup'], data_sha256=prefix['data_sha256'],
                      inputs={p:sha(p) for p in (PREFIX, LOG)},
                      quality_rule=dict(mean_final_dev_drop_at_most=.002, every_seed_drop_at_most=.005,
                                        seeds=[0, 1, 2], epochs=20),
                      decision='Exact-state execution passes; independent floating-point trajectory identity fails. '
                               'Compare fixed-recipe 20-pass final DEV quality across three seeds; no TEST.',
                      source_sha256={source:sha(source), LOG:sha(LOG)})
    else:
        assert args.fit_prefix
        fits = []; refs = []; inputs = {}
        for seed in (0, 1, 2):
            file = f'experiments/results/credit/{args.fit_prefix}_s{seed}.json'
            r = json.loads((ROOT/file).read_text())
            assert r['status'] == 'completed' and r['stage'] == 'full_fit' and r['epochs'] == 20 and r['seed'] == seed
            ref = r['reference']['path']; assert sha(ref) == r['reference']['sha256']
            fits.append(r); refs.append(json.loads((ROOT/ref).read_text())); inputs[file] = sha(file); inputs[ref] = sha(ref)
        mean = statistics.mean(r['final_dev_ll'] for r in fits)
        ref_mean = statistics.mean(r['final_dev_ll'] for r in refs)
        deltas = [r['final_dev_ll']-ref['final_dev_ll'] for r, ref in zip(fits, refs)]
        assert all(math.isfinite(v) for v in deltas)
        accepted = mean-ref_mean >= -.002 and min(deltas) >= -.005
        result = dict(status='completed', tag=args.tag, battle='B1/R1', stage='quality_decision',
                      seeds=[0, 1, 2], mean_final_dev_ll=mean, reference_mean_final_dev_ll=ref_mean,
                      seed_sd=statistics.stdev(r['final_dev_ll'] for r in fits), paired_final_dev_gains=deltas,
                      quality_preservation_gate_pass=accepted,
                      quality_rule=dict(mean_final_dev_drop_at_most=.002, every_seed_drop_at_most=.005),
                      mean_whole_fit_wall_s=statistics.mean(r['wall_s'] for r in fits),
                      inputs=inputs, source_sha256={source:sha(source)},
                      decision=('Adopt economical exact reference for further integration' if accepted else
                                'Keep original learner; diagnose precision and stability before adoption'),
                      scope='Same model/recipe, 20 passes, three seeds, DEV only. Numerical trajectories may diverge; '
                            'completed paired prefix is speed evidence; saved full-fit walls are historical. No TEST.')
    with path.open('x') as stream: stream.write(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
