"""Source-bound B4 registered readout work-to-quality endpoints."""
import argparse,json,hashlib,statistics,time,resource
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];BASE=ROOT/'experiments/results/tpp_b4'
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def load(tag,inputs):
 p=BASE/f'{tag}.json';r=json.loads(p.read_text());assert r['status']=='completed'
 for s,h in r['source_sha256'].items():assert digest(ROOT/s)==h
 inputs[str(p.relative_to(ROOT))]=digest(p);return r
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--tag',required=True);a=ap.parse_args();start=time.monotonic();inputs={};rows=[];all_cost=0.
 for seed in range(3):
  arms={name:load(f'aws_readout_curve_20261011_{name}_s{seed}',inputs) for name in ['teacher','anchor']}
  assert arms['teacher']['data_sha256']==arms['anchor']['data_sha256'];assert arms['teacher']['registered_target_mark_loss']==arms['anchor']['registered_target_mark_loss']
  points={};prior_cost=0.
  for name,weight,prior_tag in [('teacher',0.,f'aws_nonlinear_mark_readout_20261011_s{seed}'),('anchor',.1,f'aws_observation_anchored_readout_20261011_s{seed}')]:
   r=arms[name];assert r['observed_train_weight']==weight;prior=load(prior_tag,inputs);assert abs(r['student_dev']['L_M']-prior['student_dev']['L_M'])<1e-9
   prior_cost+=prior['readout_fit_s']+prior['nonlinear_fit_s']+prior['teacher_preparation_s'];all_cost+=r['wall_s']
  for name,r in arms.items():
   reached=next((p for p in r['learning_curve'] if p['reached']),None)
   points[name]=dict(reached=reached is not None,first_scheduled_point=reached,final_mark_loss=r['student_dev']['L_M'],full_curve=r['learning_curve'],diagnostic_wall_s=r['wall_s'],curve_evaluation_s=r['curve_evaluation_s'],complete_supported_training_wall_s_at_target=(r['prerequisite_fitting_s']+prior_cost+reached['cumulative_work_wall_s']) if reached else None)
  t,c=points['teacher']['first_scheduled_point'],points['anchor']['first_scheduled_point'];rows.append(dict(seed=seed,target=arms['teacher']['registered_target_mark_loss'],arms=points,final_gain=arms['teacher']['student_dev']['L_M']-arms['anchor']['student_dev']['L_M'],prior_readout_development_s=prior_cost,incremental_work_ratio=(c['cumulative_work_wall_s']/t['cumulative_work_wall_s']) if t and c else None,complete_supported_training_work_ratio=(points['anchor']['complete_supported_training_wall_s_at_target']/points['teacher']['complete_supported_training_wall_s_at_target']) if t and c else None))
 out=dict(status='completed',tag=a.tag,battle='B4 readout learning economy',rows=rows,mean_final_gain=statistics.mean(x['final_gain'] for x in rows),whole_six_fit_diagnostic_wall_s=all_cost,source_sha256={'experiments/tpp/summarize_readout_learning_curve.py':digest(Path(__file__))},input_sha256=inputs,wall_s=time.monotonic()-start,max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,scope='Fixed Github split1 DEV targets set from saved teacher-only students; scheduled snapshots0/5/10/20/35/50. No early stopping/DEV choice/TEST. Complete supported training wall includes inherited original core fitting and both prior readout-development rounds; core evaluation and earlier failed pilot investigations are additional unallocated costs, not silently zero. No whole-substrate/depth/sparse-credit learning-efficiency win; supported FLOPs unmeasured.')
 with (BASE/f'{a.tag}.json').open('x') as f:json.dump(out,f,indent=2)
 print(json.dumps(out))
if __name__=='__main__':main()
