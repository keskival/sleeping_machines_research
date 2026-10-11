"""B4 prospective three-seed calibration gate; no fit or TEST access."""
import argparse, hashlib, json, statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
def main():
 p=argparse.ArgumentParser();p.add_argument('--tag',required=True);a=p.parse_args()
 tags=['aws_b4_mark_confidence_20261011T0619Z']+[f'aws_b4_calibration_confirm_20261011_s{s}' for s in (1,2)]
 inputs={};rows=[]
 for seed,tag in enumerate(tags):
  path=ROOT/'experiments/results/tpp_b4'/f'{tag}.json';r=json.loads(path.read_text());assert r['status']=='completed'
  assert r.get('seed',0)==seed
  for source,sha in r['source_sha256'].items():assert hashlib.sha256((ROOT/source).read_bytes()).hexdigest()==sha
  inputs[str(path.relative_to(ROOT))]=hashlib.sha256(path.read_bytes()).hexdigest()
  rows.append(dict(seed=seed,gain=r['dev_mark_gain'],alpha=r['alpha_train_locked'],baseline=r['baseline_dev'],calibrated=r['dev'],prerequisite_fitting_s=r['prerequisite_fitting_s'],predictor_forward_s=r['predictor_forward_s'],scalar_fit_s=r['scalar_fit_s']))
 gains=[r['gain'] for r in rows];gate=statistics.mean(gains)>=1 and min(gains)>0
 out=dict(status='completed',battle='B4 development',tag=a.tag,rows=rows,mean_mark_gain=statistics.mean(gains),sample_sd=statistics.stdev(gains),gate=gate,rule='Registered before new seeds: mean DEV mark gain>=1 nat/sequence, positive gain in each of seeds0/1/2; TRAIN-only global coefficient, repaired timing fixed. Separate from the failed joint-mark stability gate.',input_sha256=inputs,source_sha256={str(Path(__file__).resolve().relative_to(ROOT)):hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},next_decision='If gate passes, measure both-predictor serving cost before developing a single integrated readout; if it fails, diagnose per-sequence errors without further blind fitting. No TEST, leading-total replacement, or efficiency claim.')
 with (ROOT/'experiments/results/tpp_b4'/f'{a.tag}.json').open('x') as f:f.write(json.dumps(out,indent=2)+'\n')
 print(json.dumps(out),flush=True)
if __name__=='__main__':main()
