"""R1 error analysis after the reserved large acquisition; no fitting."""
import time
import continue_window_replay as queue
from continue_fresh_scaling import admit
ROOT=queue.ROOT;queue.DEADLINE=time.monotonic()+10*3600

def main():
 gate=queue.wait_result(ROOT/'experiments/results/diagnostics/aws_bernoulli_teacher_8m_20261013_s6_v2_quality_priority_v2.json');selected='8m' if gate['accepted'] else '2m'
 sources={}
 for size,version in (('512k','v1'),(selected,'v2')):
  d=queue.wait_result(ROOT/f'experiments/results/token_language/aws_bernoulli_teacher_{size}_20261013_s6_{version}.json');sources.update(d['source_sha256'])
 for name in ('experiments/check_language_error_buckets.py','experiments/window_replay_token_language.py','experiments/public_token_scoring.py'):sources[name]=queue.sha(ROOT/name)
 tag='aws_language_error_buckets_20261013_v1';result=f'experiments/results/diagnostics/{tag}.json'
 admit(tag,'experiments/check_language_error_buckets.py',f'--tag {tag}',result,sources,[],True);queue.wait_result(ROOT/result)
 print('Frozen long-window frequency-bucket error analysis complete; use for next repair decision',flush=True)
if __name__=='__main__':main()
