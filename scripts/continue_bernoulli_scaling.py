"""R1 larger acquisition: promote only after independent-seed quality/work confirmation."""
import statistics,time
import continue_window_replay as queue
from continue_bernoulli_teacher import run
ROOT=queue.ROOT
queue.DEADLINE=time.monotonic()+10*3600

def main():
 groups={size:[] for size in ('128k','512k')}
 for seed in (6,7,8):
  for size in groups:
   version='v1' if seed==6 else 'v2'
   path=ROOT/f'experiments/results/diagnostics/aws_bernoulli_teacher_{size}_20261013_s{seed}_{version}_gate.json'
   d=queue.wait_result(path)
   if d['gate']!='PASS':print('STOP: seed confirmation failed',flush=True);return
   groups[size].append(d)
 if any(statistics.mean(d['final_nll_delta'] for d in group)>0 for group in groups.values()):
  print('STOP: mean quality did not improve at both acquired-data sizes',flush=True);return
 for size in ('2m','8m'):
  reference=f'aws_integrated_fresh_scaling_{size}_pool4_20261012_s6_v1'
  candidate=f'aws_bernoulli_teacher_{size}_20261013_s6_v1'
  if not run(reference,candidate,.25):print('STOP: larger acquisition quality/work gate failed',flush=True);return
 print('Larger acquisition ladder complete; inspect completed quality/work before public scoring',flush=True)
if __name__=='__main__':main()
