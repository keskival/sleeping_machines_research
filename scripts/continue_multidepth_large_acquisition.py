"""R1 reserved8M acquisition: no small-data quality veto, no extrapolated win."""
import json,math,time
import continue_window_replay as queue
from continue_fresh_scaling import admit
ROOT=queue.ROOT;queue.DEADLINE=time.monotonic()+16*3600

def main():
 proof=queue.wait_result(ROOT/'experiments/results/diagnostics/aws_multidepth_readout_contract_20261013_v1.json');assert proof['gate']=='PASS'
 candidate=queue.wait_result(ROOT/'experiments/results/token_language/aws_multidepth_readout_2m_20261013_s6_v1.json')
 assert candidate['status']=='completed'
 if candidate['max_rss_kb']>1200000 or candidate['train_wall_s_total']/candidate['presentations_total']>.0025:
  print('STOP: diagnose measured host resource failure before8M',flush=True);return
 sources=dict(candidate['source_sha256'])
 for name,digest in sources.items():assert queue.sha(ROOT/name)==digest,name
 reference='aws_bernoulli_teacher_8m_20261013_s6_v2';tag='aws_multidepth_readout_8m_20261013_s6_v1'
 text=(ROOT/f'experiments/queue/r1/{reference}/{reference}.txt').read_text().splitlines()[1]
 args=text.split('.py ',1)[1].replace(reference,tag);result=f'experiments/results/token_language/{tag}.json'
 admit(tag,'experiments/multidepth_readout_token_language.py',args,result,sources,[],True)
 queue.wait_result(ROOT/result);rows=[]
 for size in ('512k','2m','8m'):
  pair={}
  for variant in ('bernoulli_teacher','multidepth_readout'):
   version='v2' if size!='512k' and variant=='bernoulli_teacher' else 'v1'
   r=queue.wait_result(ROOT/f'experiments/results/token_language/aws_{variant}_{size}_20261013_s6_{version}.json')
   for name,digest in r['source_sha256'].items():assert queue.sha(ROOT/name)==digest,name
   pair[variant]=dict(targets=r['presentations_total'],nll=r['curve'][-1]['dev_nll'],training_s=r['train_wall_s_total'])
  assert pair['bernoulli_teacher']['targets']==pair['multidepth_readout']['targets']
  rows.append(dict(size=size,models=pair,paired_gain=pair['bernoulli_teacher']['nll']-pair['multidepth_readout']['nll']))
 slopes=[]
 for left,right in zip(rows,rows[1:]):
  slopes.append(dict(interval=left['size']+'->'+right['size'],nll_reduction_per_data_doubling={v:(left['models'][v]['nll']-right['models'][v]['nll'])/math.log2(right['models'][v]['targets']/left['models'][v]['targets']) for v in left['models']}))
 path=ROOT/'experiments/results/diagnostics/aws_multidepth_acquisition_curve_20261013_v1.json'
 with path.open('x') as f:json.dump(dict(status='completed',battle='R1',rows=rows,slopes=slopes,scope='Single seed, same fixed recipe and private DEV; fresh independent fits at three acquisition budgets. Empirical finite differences, not an asymptotic scaling law, external win, iso-FLOP result or hardware efficiency measurement.'),f,indent=2);f.write('\n')
 print(json.dumps(dict(rows=rows,slopes=slopes)),flush=True)
if __name__=='__main__':main()
