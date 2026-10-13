"""Record the external-quality repair's development outcomes, not investor wins."""
import fcntl,time
import continue_window_replay as queue
from continue_unique_language import ROOT,command
from continue_window_replay import sha

def main():
 queue.DEADLINE=time.monotonic()+12*3600
 proof=queue.wait_result(ROOT/'experiments/results/diagnostics/aws_multidepth_readout_contract_20261013_v1.json');assert proof['gate']=='PASS';rows=[]
 for size in ('512k','2m'):
  candidate=queue.wait_result(ROOT/f'experiments/results/token_language/aws_multidepth_readout_{size}_20261013_s6_v1.json')
  version='v1' if size=='512k' else 'v2';base=queue.wait_result(ROOT/f'experiments/results/token_language/aws_bernoulli_teacher_{size}_20261013_s6_{version}.json')
  for r in (candidate,base):
   for name,digest in r['source_sha256'].items():assert sha(ROOT/name)==digest,name
  assert candidate['presentations_total']==base['presentations_total']
  rows.append(f"{size}: finalDEV{candidate['curve'][-1]['dev_nll']:.6f} vs retained{base['curve'][-1]['dev_nll']:.6f}; complete training{candidate['train_wall_s_total']:.3f} vs{base['train_wall_s_total']:.3f}s; readout parameters{candidate['readout_parameters']} vs{base['readout_parameters']}.")
  if candidate['max_rss_kb']>1200000 or candidate['train_wall_s_total']/candidate['presentations_total']>.0025:break
 text=' '.join(rows)+' Single-seed development of temporally aligned all-depth selected-output interface48→96. Same propertokens/acquisition/optimizer budget and probability0.25 teacher,4selectedwrites/16scoredkeys; largerhead and extra transports paid. Core-state/last-leg/prior numerical contract passed. This is an owned quality repair toward an external benchmark, not a public score or an investor competitive-win headline. Next error analysis determines which representation/credit path to improve; no first-pass family verdict.'
 with open('/tmp/aws-language-publication.lock','a') as f:
  fcntl.flock(f,fcntl.LOCK_EX)
  if command('git','status','--porcelain','--untracked-files=no').strip():raise RuntimeError('Preserve tracked edits; defer publication')
  command('git','pull','--rebase')
  for name,prefix in [('experiments/HANDOFF.md','**AWS multi-depth language development — 13 October.** '),('report/I_SCIENCE.md','Current AWS multi-depth language development. ')]:
   p=ROOT/name;parts=[x for x in p.read_text().split('\n\n') if not x.startswith(prefix)]
   p.write_text(prefix+text+'\n\n'+'\n\n'.join(parts) if name.startswith('experiments/') else '\n\n'.join(parts).rstrip()+'\n\n'+prefix+text+'\n')
  command('git','add','experiments/HANDOFF.md','report/I_SCIENCE.md');command('git','commit','-m','Record owned multi-depth language quality development')
  command('.venv-docker/bin/python','report/make_pdf.py');command('.venv-docker/bin/python','report/split_report.py');command('git','add','REPORT.md','report')
  if command('git','diff','--cached','--name-only').strip():command('git','commit','-m','Publish current multi-depth language research evidence')
  command('git','pull','--rebase');command('git','push','origin','main')
if __name__=='__main__':main()
