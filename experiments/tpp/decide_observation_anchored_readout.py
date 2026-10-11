"""B4 preregistered three-seed observation-anchored readout decision."""
import argparse,json,hashlib,statistics,time,resource
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--tag',required=True);a=ap.parse_args();start=time.monotonic();base=ROOT/'experiments/results/tpp_b4';prior=base/'aws_single_readout_decision_20261011.json';old=json.loads(prior.read_text());assert old['status']=='completed' and old['gate'];inputs={str(prior.relative_to(ROOT)):digest(prior)};rows=[]
 for seed in range(3):
  path=base/f'aws_observation_anchored_readout_20261011_s{seed}.json';r=json.loads(path.read_text());assert r['status']=='completed' and r['observed_train_weight']==.1
  for s,h in r['source_sha256'].items():assert digest(ROOT/s)==h
  inputs[str(path.relative_to(ROOT))]=digest(path);gain=old['rows'][seed]['student']['L_M']-r['student_dev']['L_M'];rows.append(dict(seed=seed,student=r['student_dev'],prior_student=old['rows'][seed]['student'],gain_over_prior_student=gain,baseline_gain=r['mark_gain'],teacher_retention=r['retained_teacher_gain'],original_gate=r['gate'],fit_s=r['readout_fit_s']+r['nonlinear_fit_s'],teacher_preparation_s=r['teacher_preparation_s'],prerequisite_fitting_s=r['prerequisite_fitting_s']))
 mean=statistics.mean(x['gain_over_prior_student'] for x in rows);gate=mean>=.1 and all(x['gain_over_prior_student']>=-.25 and x['original_gate'] for x in rows)
 out=dict(status='completed',tag=a.tag,battle='B4',gate=gate,mean_gain_over_prior_student=mean,sample_sd=statistics.stdev(x['gain_over_prior_student'] for x in rows),rows=rows,source_sha256={'experiments/tpp/decide_observation_anchored_readout.py':digest(Path(__file__))},input_sha256=inputs,wall_s=time.monotonic()-start,max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,rule='Registered before all3scores: mean DEV gain>=0.1 against saved one-core student, each gain>=-0.25 and unchanged baseline/80% teacher-retention gate all pass. Fixed TRAIN label weight0.1, no DEV grid.',scope='Same Github split1 DEV, three fixed training seeds; no TEST or independent-dataset confirmation. Teacher fits and prior/current readout development are paid. No future-intervention or sparse-learning claim.')
 with (base/f'{a.tag}.json').open('x') as f:json.dump(out,f,indent=2)
 print(json.dumps(out))
if __name__=='__main__':main()
