"""R1 arithmetic/payload audit after completed three-seed quality evidence."""
import time
import continue_window_replay as queue
from continue_fresh_scaling import admit
ROOT=queue.ROOT;queue.DEADLINE=time.monotonic()+10*3600

def main():
 for seed in (6,7,8):
  version='v1' if seed==6 else 'v2'
  d=queue.wait_result(ROOT/f'experiments/results/diagnostics/aws_bernoulli_teacher_512k_20261013_s{seed}_{version}_quality_priority_v2.json');assert d['accepted']
 base=queue.wait_result(ROOT/'experiments/results/token_language/aws_ordered_native_work_probe_20261013_v1.json')
 command=(ROOT/'experiments/queue/r1/aws_ordered_native_work_probe_20261013_v1/aws_ordered_native_work_probe_20261013_v1.txt').read_text().splitlines()[1]
 raw=command.split('.py ',1)[1];sources=dict(base['source_sha256'])
 for driver,tag in (('experiments/bernoulli_teacher_token_language.py','aws_teacher_native_reference_20261013_v1'),('experiments/profile_bernoulli_native_work.py','aws_teacher_native_probe_20261013_v1')):
  for name in ('experiments/bernoulli_teacher_token_language.py',driver):sources[name]=queue.sha(ROOT/name)
  args=raw.replace('aws_ordered_native_work_probe_20261013_v1',tag)+' --teacher-probability 0.25';result=f'experiments/results/token_language/{tag}.json'
  admit(tag,driver,args,result,sources,[],True);d=queue.wait_result(ROOT/result);sources.update(d['source_sha256'])
 driver='experiments/analyze_teacher_native_work.py';sources[driver]=queue.sha(ROOT/driver);tag='aws_teacher_native_work_decision_20261013_v1';result=f'experiments/results/diagnostics/{tag}.json'
 admit(tag,driver,f'--tag {tag}',result,sources,[],False);queue.wait_result(ROOT/result)
 print('Teacher arithmetic/payload audit complete; no CPU timing extrapolation',flush=True)
if __name__=='__main__':main()
