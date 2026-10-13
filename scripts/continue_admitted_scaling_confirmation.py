"""R1 large-data replication after unchanged seed6 quality/resource and prefix gates."""
import time
import continue_window_replay as queue
from continue_fresh_scaling import admit
from continue_bernoulli_teacher import run
ROOT=queue.ROOT;queue.DEADLINE=time.monotonic()+10*3600

def prefix(seed):
 driver='experiments/analyze_admitted_scaling_prefix.py';sources={driver:queue.sha(ROOT/driver)}
 names=([f'aws_bernoulli_teacher_2m_20261013_s6_v2',f'aws_bernoulli_teacher_8m_20261013_s6_v2'] if seed==6 else [f'aws_bernoulli_teacher_512k_20261013_s{seed}_v2',f'aws_bernoulli_teacher_8m_20261013_s{seed}_v3'])
 for name in names:sources.update(queue.wait_result(ROOT/f'experiments/results/token_language/{name}.json')['source_sha256'])
 tag=f'aws_admitted_scaling_prefix_20261013_s{seed}_v1';result=f'experiments/results/diagnostics/{tag}.json'
 admit(tag,driver,f'--tag {tag} --seed {seed}',result,sources,[],False)
 return queue.wait_result(ROOT/result)['gate']=='PASS'

def main():
 initial=queue.wait_result(ROOT/'experiments/results/diagnostics/aws_bernoulli_teacher_8m_20261013_s6_v2_quality_priority_v2.json')
 if not initial['accepted'] or not prefix(6):print('STOP: seed6 larger-quality/prefix gate failed',flush=True);return
 for seed in (7,8):
  tag=f'aws_bernoulli_teacher_8m_20261013_s{seed}_v3'
  run(f'aws_integrated_fresh_scaling_8m_pool4_20261013_s{seed}_v1',tag,.25)
  gate=queue.wait_result(ROOT/f'experiments/results/diagnostics/{tag}_gate.json')
  if gate['final_nll_delta']>-.01 or gate['training_wall_ratio']>1.01 or not prefix(seed):print('STOP: independent larger quality/resource/prefix gate failed',flush=True);return
 print('Three-seed larger acquisition confirmation complete',flush=True)
if __name__=='__main__':main()
