"""Completed information-path gate; no implicit further training."""
import argparse
import hashlib
import json
import math
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
SOURCE='experiments/results/credit/curie_future_packet_v2_20261011T0740Z_measure.json'


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--tag',required=True);a=ap.parse_args()
    p=ROOT/SOURCE;d=json.loads(p.read_text());assert d['status']=='completed' and d['mode']=='measure'
    cp=ROOT/d['checkpoint'];assert hashlib.sha256(cp.read_bytes()).hexdigest()==d['checkpoint_sha256']
    m=d['metrics'];r={v['arm']:v for v in m['dev_rows']};f=r['revealed'];z=r['zero_future'];b=r['prefix_prior']
    checks=dict(finite=all(math.isfinite(v['mse']) and math.isfinite(v['policy_logit_residual_second_moment']) for v in r.values()),
        revealed_mse_below_zero_future=f['mse']<z['mse'],revealed_mse_below_prefix_prior=f['mse']<b['mse'],
        positive_paired_prefix_interval=m['prefix_bootstrap95'][0]>0,
        sensitivity_second_moment_below_both=f['policy_logit_residual_second_moment']<min(z['policy_logit_residual_second_moment'],b['policy_logit_residual_second_moment']),
        resource=d['peak_rss_kb']<=1500000)
    out=ROOT/f'experiments/results/credit/{a.tag}.json';assert not out.exists()
    result=dict(status='completed',battle='B1/R1',tag=a.tag,information_path_gate=all(checks.values()),checks=checks,
        dev_rows=m['dev_rows'],chronological_holdout=m['chronological_holdout'],
        revealed_mse_ratio_vs_zero_future=f['mse']/z['mse'],revealed_mse_ratio_vs_prior=f['mse']/b['mse'],
        revealed_logit_residual_ratio_vs_prior=f['policy_logit_residual_second_moment']/b['policy_logit_residual_second_moment'],
        paired_mse_improvement=m['revealed_over_zero_future_mse_gain'],prefix_bootstrap95=m['prefix_bootstrap95'],
        new_fitting_supported_gflops=sum(v['supported_flops'] for v in m['fit_work'])/1e9,
        preparation_supported_gflops=m['preparation']['supported_flops']/1e9,
        dev_verification_supported_gflops=m['evaluation_supported_flops']/1e9,
        whole_wall_s=d['whole_wall_s'],source_result=SOURCE,source_result_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),source_checkpoint_sha256=d['checkpoint_sha256'],
        source_sha256={str(Path(__file__).relative_to(ROOT)):hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
        scope='One-seed connected credit information-path development gate registered before full result. Per-future MSE and policy-logit residual second moment, not native-parameter gradient variance, forward utility or matched-quality work. Raw exogenous observations on learning side only; causal actor unchanged. Original512 acquisition, prior-model pretraining and failed-protocol development costs are additional. Gate admits design of coupled causal acquisition, not an automatic larger run.')
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
