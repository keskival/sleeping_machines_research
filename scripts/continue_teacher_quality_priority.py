"""User-directed R1 quality/work promotion: tiny time increases do not veto quality gains."""
import json,time,statistics
import continue_window_replay as queue
from continue_bernoulli_teacher import run
ROOT=queue.ROOT
queue.DEADLINE=time.monotonic()+10*3600

def quality_gate(tag):
 path=ROOT/f'experiments/results/diagnostics/{tag}_gate.json';d=queue.wait_result(path)
 # User13Oct: quality improvement matters; seed8 0.03% extra time is accepted.
 # Bound any new promotion to at most1% extra complete time and >=0.01 NLL gain.
 accepted=d['final_nll_delta']<=-.01 and d['training_wall_ratio']<=1.01
 receipt=dict(status='completed',battle='R1',accepted=accepted,quality_gain=-d['final_nll_delta'],training_wall_ratio=d['training_wall_ratio'],original_gate=d['gate'],input_sha256={str(path.relative_to(ROOT)):queue.sha(path)},source_sha256={str(__file__.replace(str(ROOT)+'/','')):queue.sha(ROOT/'scripts/continue_teacher_quality_priority.py')},scope='User-directed13October quality-priority promotion: gain≥0.01nats with≤1% extra complete CPU time. Does not convert slower fits into wall-time wins; original strict gate preserved.')
 out=ROOT/f'experiments/results/diagnostics/{tag}_quality_priority_v2.json'
 with out.open('x') as f:json.dump(receipt,f,indent=2);f.write('\n')
 print(json.dumps(receipt),flush=True)
 return accepted

def main():
 for seed in (6,7,8):
  version='v1' if seed==6 else 'v2'
  for size in ('128k','512k'):
   tag=f'aws_bernoulli_teacher_{size}_20261013_s{seed}_{version}'
   if seed==8 and size=='512k':run(f'aws_integrated_ordered_lookup_512k_20261013_s8_v3',tag,.25)
   if not quality_gate(tag):print('STOP: quality/resource gate requires diagnosis',flush=True);return
 for size in ('2m','8m'):
  tag=f'aws_bernoulli_teacher_{size}_20261013_s6_v2'
  run(f'aws_integrated_fresh_scaling_{size}_pool4_20261012_s6_v1',tag,.25)
  if not quality_gate(tag):print('STOP: larger quality/resource gate requires diagnosis',flush=True);return
 print('Quality-priority acquisition ladder completed',flush=True)
if __name__=='__main__':main()
