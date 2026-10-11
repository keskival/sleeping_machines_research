"""B1/R1 completed-stage diagnostics and resource admission, no implicit scaling."""
import argparse
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PREFIX = 'curie_sparse_paired_ac_v2_20261011T0930Z'


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--tag', required=True)
    ap.add_argument('--stage', choices=['acquire', 'reuse'], required=True); a = ap.parse_args()
    source = ROOT/f'experiments/results/credit/{PREFIX}_{a.stage}.json'
    d = json.loads(source.read_text()); assert d['status'] == 'completed' and d['mode'] == a.stage
    cp = ROOT/d['checkpoint']; assert hashlib.sha256(cp.read_bytes()).hexdigest() == d['checkpoint_sha256']
    m = d['metrics']; e = m['evaluation']; fits = {r['arm']: r for r in m['cumulative_fits']}
    evaluation = {r['arm']: r for r in e['rows']}
    finite = all(math.isfinite(r['gain_vs_baseline']) and math.isfinite(r['common_probe_mse']) for r in e['rows'])
    admitted = finite and d['peak_rss_kb'] <= 1500000
    full = fits['full_fixed']
    rows = []
    for name, fit in fits.items():
        rows.append(dict(arm=name, factual_targets=fit['unique_factual_targets'], policy_updates=fit['policy_updates'],
                         credit_updates=fit['credit_updates'], extra_worlds=fit['extra_worlds'],
                         whole_fit_gflops=fit['supported_flops']/1e9, fit_mflops_per_target=fit['supported_flops_per_factual_target']/1e6,
                         whole_fit_s=fit['wall_s'], work_ratio_vs_full=fit['supported_flops']/full['supported_flops'],
                         gain_vs_baseline=evaluation[name]['gain_vs_baseline'],
                         heldout_pair_error_over_zero=evaluation[name]['common_probe_error_over_zero']))
    out = ROOT/f'experiments/results/credit/{a.tag}.json'; assert not out.exists()
    result = dict(status='completed', tag=a.tag, battle='B1/R1', stage=a.stage, rows=rows,
                  comparisons=e['comparisons'], continue_registered_ladder=admitted and a.stage == 'acquire',
                  resource_finite_gate=admitted,
                  learned_probe_error_ratio_vs_fixed=evaluation['sparse_learned']['common_probe_mse']/evaluation['sparse_fixed']['common_probe_mse'] if evaluation['sparse_fixed']['common_probe_mse'] else None,
                  source_result=str(source.relative_to(ROOT)), source_result_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                  source_checkpoint_sha256=d['checkpoint_sha256'],
                  source_sha256={str(Path(__file__).relative_to(ROOT)): hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
                  scope='Completed one-seed synthetic connected route-learning development stage. Raw carrier fixed; readout adapts within each episode; controller and critic weights persist across episodes. Reuse admission follows registered finite/resource gate, not immediate utility. Cumulative work includes128-step acquisition for continuation and reset/reuse clones; inherited model pretraining and shared DEV diagnostics remain additional. No benchmark, matched-quality/depth or independent-confirmation claim.')
    out.write_text(json.dumps(result, indent=2)+'\n'); print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__': main()
