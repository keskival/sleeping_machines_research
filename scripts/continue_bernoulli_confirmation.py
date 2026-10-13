"""R1 independent-seed admission confirmation after the larger seed6 gate."""
import json,time
import continue_window_replay as queue
from continue_bernoulli_teacher import run
ROOT=queue.ROOT
queue.DEADLINE=time.monotonic()+4*3600

def main():
 gate=queue.wait_result(ROOT/'experiments/results/diagnostics/aws_bernoulli_teacher_512k_20261013_s6_v1_gate.json')
 if gate['gate']!='PASS':print('STOP: larger seed6 quality/work gate failed',flush=True);return
 for seed in (7,8):
  for size in ('128k','512k'):
   reference=f'aws_integrated_ordered_lookup_{size}_20261013_s{seed}_v3'
   tag=f'aws_bernoulli_teacher_{size}_20261013_s{seed}_v2'
   if not run(reference,tag,.25):print('STOP: independent seed quality/work gate failed',flush=True);return
 print('Three-seed teacher-admission confirmation completed',flush=True)
if __name__=='__main__':main()
