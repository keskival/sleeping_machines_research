"""R1 concrete output-information repair: contract then512k and reserved2M development."""
import time
import continue_window_replay as queue
from continue_fresh_scaling import admit
ROOT=queue.ROOT;queue.DEADLINE=time.monotonic()+12*3600

def main():
 saved=queue.wait_result(ROOT/'experiments/results/token_language/aws_bernoulli_teacher_512k_20261013_s6_v1.json');sources=dict(saved['source_sha256'])
 for name in ('experiments/multidepth_readout_token_language.py','sleeping_machines/multidepth_counterfactual_episodes.py','sleeping_machines/multidepth_ordered_episodes.py','experiments/credit/check_multidepth_readout.py'):sources[name]=queue.sha(ROOT/name)
 tag='aws_multidepth_readout_contract_20261013_v1';result=f'experiments/results/diagnostics/{tag}.json'
 admit(tag,'experiments/credit/check_multidepth_readout.py',f'--tag {tag}',result,sources,[],True)
 contract=queue.wait_result(ROOT/result);assert contract['gate']=='PASS'
 for size in ('512k','2m'):
  reference=f'aws_bernoulli_teacher_{size}_20261013_s6_'+('v1' if size=='512k' else 'v2')
  text=(ROOT/f'experiments/queue/r1/{reference}/{reference}.txt').read_text().splitlines()[1]
  tag=f'aws_multidepth_readout_{size}_20261013_s6_v1';args=text.split('.py ',1)[1].replace(reference,tag);result=f'experiments/results/token_language/{tag}.json'
  admit(tag,'experiments/multidepth_readout_token_language.py',args,result,sources,[],True);d=queue.wait_result(ROOT/result)
  print('Completed '+tag+'; finalDEV '+str(d['curve'][-1]['dev_nll'])+'; full training '+str(d['train_wall_s_total']),flush=True)
  if d['max_rss_kb']>1200000 or d['train_wall_s_total']/d['presentations_total']>.0025:print('STOP: measured resource envelope exceeded; diagnose execution',flush=True);return
 print('Reserved2M information-path repair complete; evaluate quality against retained recipe and external target',flush=True)
if __name__=='__main__':main()
