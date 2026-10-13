"""R1 source-bound long-history quality/work after each native deep fit."""
import json,time
import continue_window_replay as queue
from continue_fresh_scaling import admit
ROOT=queue.ROOT;queue.DEADLINE=time.monotonic()+14*86400

def main():
 for depth,size in ((8,'8m'),(16,'8m'),(8,'32m')):
  run=f'aws_native_deep_d{depth}_{size}_20261013_s6_v1'
  r=queue.wait_result(ROOT/f'experiments/results/token_language/{run}.json');sources=dict(r['source_sha256'])
  for name in ('experiments/check_deep_language_quality.py','experiments/public_token_scoring.py'):sources[name]=queue.sha(ROOT/name)
  tag=f'aws_native_deep_d{depth}_{size}_quality_20261013_s6_v1';result=f'experiments/results/diagnostics/{tag}.json'
  admit(tag,'experiments/check_deep_language_quality.py',f'--tag {tag} --run {run}',result,sources,[],True)
  d=queue.wait_result(ROOT/result)
  print(json.dumps(dict(stage=run,rows=[{k:v for k,v in row.items() if k!='buckets'} for row in d['rows']])),flush=True)
 print('Deep acquisition quality/work record complete; exact public benchmark is a separate selected-checkpoint evaluation',flush=True)
if __name__=='__main__':main()
