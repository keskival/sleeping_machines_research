"""B5/R1 outcome calibration where route changes affect the loss."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main():
    p = argparse.ArgumentParser(); p.add_argument('--tag', required=True); a = p.parse_args()
    rows = []; inputs = {}
    for arm in ('learned', 'prior', 'exact'):
        path = ROOT/f'experiments/results/causal_induction/curie_world_credit_20261012_v2_{arm}_reuse.json'
        blob = path.read_bytes(); d = json.loads(blob)
        assert d['status'] == 'completed' and d['targets'] == 1024
        for name, digest in d['source_sha256'].items():
            assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == digest, name
        inputs[str(path.relative_to(ROOT))] = hashlib.sha256(blob).hexdigest()
        gap = sum(r['route_logit_gap_squared'] for r in d['rows'])
        error = sum(r['route_contrast_residual_squared'] for r in d['rows'])
        assert gap > 0
        cases = [dict(path=r['path'], depth=r['depth'], variant=r['variant'],
                      route_sensitivity_weighted_brier=4*r['route_contrast_residual_squared']/r['route_logit_gap_squared'],
                      route_logit_rms=(r['route_logit_gap_squared']/r['targets'])**.5,
                      actor_nll=r['nll']) for r in d['rows']]
        if arm == 'prior': assert abs(4*error/gap-.25) < 1e-6
        if arm == 'exact': assert error == 0
        rows.append(dict(arm=arm, actor_nll=d['pooled_nll'],
                         outcome_brier=sum(r['world_brier_sum'] for r in d['rows'])/d['targets'],
                         route_sensitivity_weighted_brier=4*error/gap,
                         route_logit_rms=(gap/d['targets'])**.5, cases=cases))
    learned, prior, exact = rows
    result = dict(status='completed', battle='B5/R1', rows=rows,
                  weighted_error_reduction_vs_prior=1-learned['route_sensitivity_weighted_brier']/.25,
                  actor_gain_vs_prior=prior['actor_nll']-learned['actor_nll'],
                  input_sha256=inputs,
                  source_sha256={str(Path(__file__).relative_to(ROOT)):hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
                  derivation='BCE contrast is softplus(z_alt)-softplus(z_factual)-q*(z_alt-z_factual). Halfblend uses .5*y+.5*q, so residual squared is .25*(q-y)^2*(z_alt-z_factual)^2. Therefore 4*sum(residual squared)/sum(logit gap squared) reconstructs sensitivity-weighted Brier. Exact arm uses y, hence zero teaching residual does not imply perfect issued outcome prediction.',
                  scope='Completed exploratory eight-case cohort. Arms share incoming learner but diverge during adaptation; weighted errors use each arm own routes, not paired fixed-route causal effects. Logit-gap weighting is immediate loss sensitivity, not optimizer/future-learning sensitivity. No new fit, task advantage, audit saving or efficiency claim.')
    out = ROOT/f'experiments/results/diagnostics/{a.tag}.json'
    with out.open('x') as f: json.dump(result, f, indent=2); f.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('rows','input_sha256','source_sha256')}), flush=True)


if __name__ == '__main__': main()
