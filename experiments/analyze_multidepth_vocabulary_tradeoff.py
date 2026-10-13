"""R1 exact vocabulary decomposition from completed paired DEV evidence."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
 source=ROOT/'experiments/results/diagnostics/aws_multidepth_error_buckets_20261013_v1.json'
 d=json.loads(source.read_text());assert d['status']=='completed';rows=[]
 for size in ('512k','2m'):
  a,b=[next(r for r in d['rows'] if r['acquisition']==size and r['variant']==v) for v in ('baseline','multidepth')]
  assert a['scored']['targets']==b['scored']['targets']
  buckets=[]
  for x,y in zip(a['buckets'],b['buckets']):
   assert (x['rank_low'],x['rank_high'],x['count'])==(y['rank_low'],y['rank_high'],y['count'])
   buckets.append(dict(rank_low=x['rank_low'],rank_high=x['rank_high'],target_fraction=x['target_fraction'],paired_nll_gain=x['nll']-y['nll'],weighted_gain=x['total_nll_contribution']-y['total_nll_contribution']))
  gain=a['scored']['nll']-b['scored']['nll'];assert abs(sum(x['weighted_gain'] for x in buckets)-gain)<1e-10
  rows.append(dict(acquisition=size,paired_gain=gain,buckets=buckets))
 out=ROOT/'experiments/results/diagnostics/aws_multidepth_vocabulary_tradeoff_20261013_v1.json'
 packet=dict(status='completed',battle='R1',rows=rows,input_sha256={str(source.relative_to(ROOT)):hashlib.sha256(source.read_bytes()).hexdigest()},source_sha256={str(Path(__file__).relative_to(ROOT)):hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},decision='Keep reserved8M acquisition. Do not promote at2M: frequent-token benefit is outweighed by other buckets. If the8M loss persists, investigate a baseline-preserving extra-depth interface before further widening.',scope='One seed,8192existing DEV targets,1024history. Exact loss decomposition, not causal attribution to head width, optimization or depth.')
 with out.open('x') as f:json.dump(packet,f,indent=2);f.write('\n')
 print(json.dumps(rows))
if __name__=='__main__':main()
