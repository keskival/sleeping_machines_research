"""B4 fixed-recipe single-readout three-seed decision; no model execution."""
import argparse,hashlib,json,statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
def main():
 p=argparse.ArgumentParser();p.add_argument('--tag',required=True);a=p.parse_args();rows=[];inputs={}
 for seed in range(3):
  path=ROOT/'experiments/results/tpp_b4'/f'aws_nonlinear_mark_readout_20261011_s{seed}.json';r=json.loads(path.read_text());assert r['status']=='completed' and r['seed']==seed
  for source,h in r['source_sha256'].items():assert hashlib.sha256((ROOT/source).read_bytes()).hexdigest()==h
  inputs[str(path.relative_to(ROOT))]=hashlib.sha256(path.read_bytes()).hexdigest()
  rows.append(dict(seed=seed,baseline=r['baseline_dev'],student=r['student_dev'],teacher=r['teacher_dev'],mark_gain=r['mark_gain'],retention=r['retained_teacher_gain'],gate=r['gate'],parameters=r['parameters'],teacher_parameters=r['teacher_parameters'],student_serving_s=r['warm_single_dev_s'],teacher_serving_s=r['warm_pair_dev_s'],prerequisite_fitting_s=r['prerequisite_fitting_s'],teacher_preparation_s=r['teacher_preparation_s'],linear_fit_s=r['readout_fit_s'],residual_fit_s=r['nonlinear_fit_s']))
 out=dict(status='completed',battle='B4 development',tag=a.tag,rows=rows,gate=all(r['gate'] for r in rows),mean_mark_gain=statistics.mean(r['mark_gain'] for r in rows),sample_sd=statistics.stdev(r['mark_gain'] for r in rows),mean_retention=statistics.mean(r['retention'] for r in rows),rule='Registered after seed0 selection, before seeds1/2: each seed meets unchanged pilot gate gain>=1nat/sequence and retention>=80%; timing fixed. Same Github split1 DEV, not independent datasets.',input_sha256=inputs,source_sha256={str(Path(__file__).resolve().relative_to(ROOT)):hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},next_decision='Publish every seed and full prerequisite/preparation/head/serving costs. If gate passes, measure integrated serving/query contract before further dataset development; if it fails, diagnose causal feature information, no blind wider fitting. No TEST or automatic benchmark claim.')
 with (ROOT/'experiments/results/tpp_b4'/f'{a.tag}.json').open('x') as f:f.write(json.dumps(out,indent=2)+'\n')
 print(json.dumps(out),flush=True)
if __name__=='__main__':main()
