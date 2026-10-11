"""B1/R1 completed coupled utility, residual and full-fit work diagnosis."""
import argparse,hashlib,json
import numpy as np
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--source',required=True);ap.add_argument('--tag',required=True);a=ap.parse_args()
    path=ROOT/a.source;d=json.loads(path.read_text());assert d['status']=='completed'
    for p,digest in d['source_sha256'].items():assert hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==digest,p
    out=ROOT/f'experiments/results/credit/{a.tag}.json';assert not out.exists()
    fits={r['arm']:r for r in d['fits']};cost={r['arm']:r for r in d['cumulative_fits']};rows=[]
    for r in d['evaluation']['rows']:
        n=r['arm'];f=fits[n];c=cost[n]
        rows.append(dict(arm=n,gain_vs_baseline=r['gain_vs_baseline'],bootstrap95=r['bootstrap95'],whole_fit_gflops=c['supported_flops']/1e9,fit_mflops_per_target=c['supported_flops']/c['factual_targets']/1e6,whole_fit_s=c['wall_s'],sampled_residual_mse=f['mean_sampled_residual_square'],actual_worlds=f['actual_worlds'],readout_updates=f['readout_updates'],reference_projection_targets=f['projected_route_targets'],frozen_projection_targets=(f['end_prefix']-f['start_prefix'])*54*4 if n!='prefix' else (f['end_prefix']-f['start_prefix'])*2*4))
    prefix=fits['prefix'];comparisons=d['evaluation']['comparisons'];ref=comparisons['reference_learner_over_prefix']
    verdict='Reference-learning credit improves actual route utility with positive paired interval.' if ref['bootstrap95'][0]>0 else 'Reference-learning credit does not clear the positive paired utility gate.'
    result=dict(status='completed',battle='B1/R1',tag=a.tag,source_result=a.source,source_result_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),rows=rows,comparisons=comparisons,verdict=verdict,residual_mse_ratio_reference_over_prefix=fits['reference_learner']['mean_sampled_residual_square']/prefix['mean_sampled_residual_square'],fit_wall_ratio_reference_over_prefix=cost['reference_learner']['wall_s']/cost['prefix']['wall_s'],gate=ref['bootstrap95'][0]>0,scope='One controller initialization, fresh64-prefix/eight-future synthetic DEV. Sampled residual MSE follows different learned action distributions; it is not a paired fixed-policy variance comparison. Whole-fit includes inherited128 acquisition. Supported FLOPs omit unsupported scalar/special/optimizer operations. The source projected_route_targets counter covers evolving-reference projections only; frozen output projections are counted separately here, including two addressed outputs in prefix replay. No TEST/depth/hardware claim.')
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2),flush=True)
if __name__=='__main__':main()
