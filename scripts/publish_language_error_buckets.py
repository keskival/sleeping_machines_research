"""Publish completed frozen language bucket diagnosis without declaring a capacity failure."""
import fcntl,time
import continue_window_replay as queue
from continue_unique_language import ROOT,command
from continue_window_replay import sha

def main():
 queue.DEADLINE=time.monotonic()+10*3600
 d=queue.wait_result(ROOT/'experiments/results/diagnostics/aws_language_error_buckets_20261013_v1.json')
 for name,digest in {**d['source_sha256'],**d['input_sha256']}.items():assert sha(ROOT/name)==digest,name
 lines=[]
 for row in d['rows']:
  pieces=[f"rank{b['rank_low']}–{b['rank_high']}:fraction{b['target_fraction']:.3f},NLL{b['nll']:.4f},featuregain{b['feature_gain']:.4f}" for b in row['buckets'] if b['count']]
  lines.append(f"{row['acquisition']} finalweights:1024-historyDEV NLL{row['scored']['nll']:.6f}; "+'; '.join(pieces)+'.')
 text=' '.join(lines)+' '+d['scope']
 with open('/tmp/aws-language-publication.lock','a') as f:
  fcntl.flock(f,fcntl.LOCK_EX)
  if command('git','status','--porcelain','--untracked-files=no').strip():raise RuntimeError('Preserve tracked edits; defer publication')
  command('git','pull','--rebase')
  for name,prefix in [('experiments/HANDOFF.md','**AWS frozen language error analysis — 13 October.** '),('report/I_SCIENCE.md','Current AWS frozen language error analysis. ')]:
   p=ROOT/name;parts=[x for x in p.read_text().split('\n\n') if not x.startswith((prefix,'**AWS large-fit error-analysis next'))]
   p.write_text(prefix+text+'\n\n'+'\n\n'.join(parts) if name.startswith('experiments/') else '\n\n'.join(parts).rstrip()+'\n\n'+prefix+text+'\n')
  command('git','add','experiments/HANDOFF.md','report/I_SCIENCE.md');command('git','commit','-m','Record frequency-bucket language error diagnosis')
  command('.venv-docker/bin/python','report/make_pdf.py');command('.venv-docker/bin/python','report/split_report.py');command('git','add','REPORT.md','report')
  if command('git','diff','--cached','--name-only').strip():command('git','commit','-m','Publish current language error analysis')
  command('git','pull','--rebase');command('git','push','origin','main')
if __name__=='__main__':main()
