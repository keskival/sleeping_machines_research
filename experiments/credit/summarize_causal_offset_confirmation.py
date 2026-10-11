"""B1/R1 predeclared two-new-history gate; no fitting or implicit extra jobs."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PREFIX = 'curie_causal_offset_confirm_20261011T0650Z'


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--tag', required=True); args = ap.parse_args()
    out = ROOT/'experiments/results/credit'/f'{args.tag}.json'; assert not out.exists()
    names = ['curie_causal_offset_20261011T0630Z_measure', f'{PREFIX}_stream473', f'{PREFIX}_stream474']
    rows, inputs = [], {}
    for name in names:
        p = ROOT/'experiments/results/credit'/f'{name}.json'
        d = json.loads(p.read_text()); assert d['status'] == 'completed' and d['mode'] == 'measure'
        cp = ROOT/d['checkpoint']; assert hashlib.sha256(cp.read_bytes()).hexdigest() == d['checkpoint_sha256']
        m = d['metrics']; arms = {r['arm']: r for r in m['rows']}
        assert m['prefixes'] == 32 and m['continuations'] == 8
        rows.append(dict(tag=name, stream_seed=m['stream_seed'], audit_seed=m.get('audit_seed', 492),
                         causal_ratio=arms['causal_centered']['variance_ratio_vs_zero'],
                         fixed_ratio=arms['train_centered']['variance_ratio_vs_zero'],
                         raw_ratio=arms['raw_task']['variance_ratio_vs_zero'],
                         variance_zero=arms['zero_predictor']['mean_sampling_variance'],
                         variance_causal=arms['causal_centered']['mean_sampling_variance'],
                         realized_mse_zero=arms['zero_predictor']['realized_squared_error'],
                         realized_mse_causal=arms['causal_centered']['realized_squared_error'],
                         changed_prefixes=m['changed_prefixes'], updates=m['calibration_feedback_updates'],
                         max_mean_error=m['max_mean_reconstruction_error'],
                         wall_s=d['whole_wall_s'], policy_supported_flops=m['policy_supported_flops'],
                         teacher_supported_flops=m['teacher_supported_flops'],
                         preparation_supported_flops=m['train_calibration_preparation_supported_flops'],
                         peak_rss_kb=d['peak_rss_kb'], selection_sha=m['selection_checkpoint_sha256'],
                         initial_offset=m['initial_train_offset']))
        inputs[str(p.relative_to(ROOT))] = hashlib.sha256(p.read_bytes()).hexdigest()
    assert len({r['selection_sha'] for r in rows}) == 1
    assert max(r['initial_offset'] for r in rows)-min(r['initial_offset'] for r in rows) < 1e-12
    passed = all(r['causal_ratio'] is not None and r['causal_ratio'] <= .90 and r['max_mean_error'] < 1e-11 for r in rows[1:])
    result = dict(status='completed', tag=args.tag, battle='B1/R1', rows=rows, confirmation_gate=passed,
                  gate='Both new independent episode/audit histories473/474 reduce full measured variance>=10% versus zero predictor; conditional mean error<1e-11. First472 motivates confirmation and is not selection evidence for the gate.',
                  inputs=inputs, source_sha256={str(Path(__file__).relative_to(ROOT)): hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
                  scope='One fixed native actor/critic/proposal with independent synthetic episode and audit histories, not three fitted model seeds. Causal calibration learns from prior sampled feedback. Prefix intervals are descriptive under adaptive history. Variance is conditional on each finite generated continuation table and does not itself establish quality, total training efficiency, population significance, real-sequence nonstationarity or depth scaling. All diagnostic replay, discovery, calibration/preparation and prior fitting costs remain charged.')
    out.write_text(json.dumps(result, indent=2)+'\n'); print(json.dumps(result, indent=2))


if __name__ == '__main__': main()
