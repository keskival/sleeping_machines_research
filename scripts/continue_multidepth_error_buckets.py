"""R1 paired error analysis after the reserved interface repair; DEV only."""
import time
import continue_window_replay as queue
from continue_fresh_scaling import admit
ROOT=queue.ROOT;queue.DEADLINE=time.monotonic()+12*3600

def main():
 sources={}
 for size in ('512k','2m'):
  for variant in ('bernoulli_teacher','multidepth_readout'):
   version='v2' if size=='2m' and variant=='bernoulli_teacher' else 'v1'
   d=queue.wait_result(ROOT/f'experiments/results/token_language/aws_{variant}_{size}_20261013_s6_{version}.json');sources.update(d['source_sha256'])
 for name in ('experiments/check_multidepth_error_buckets.py','experiments/public_token_scoring.py'):
  sources[name]=queue.sha(ROOT/name)
 tag='aws_multidepth_error_buckets_20261013_v1';result=f'experiments/results/diagnostics/{tag}.json'
 admit(tag,'experiments/check_multidepth_error_buckets.py',f'--tag {tag}',result,sources,[],True)
 queue.wait_result(ROOT/result)
 print('Paired frozen1024-history DEV analysis complete; inspect frequent-token and total loss before next repair',flush=True)
if __name__=='__main__':main()
