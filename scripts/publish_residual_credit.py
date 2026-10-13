"""Publish bounded R1 admission gates, including a stopped development result."""
import fcntl,json,time
import continue_window_replay as queue
from continue_unique_language import ROOT,command
from continue_window_replay import sha

def main():
 queue.DEADLINE=time.monotonic()+4*3600;rows=[]
 for name in ('off','128k','512k'):
  path=ROOT/f'experiments/results/diagnostics/aws_residual_credit_{name}_20261013_s6_v1_gate.json'
  d=queue.wait_result(path)
  for source,digest in {**d['source_sha256'],**d['input_sha256']}.items():assert sha(ROOT/source)==digest,source
  rows.append((name,d))
  if d['gate']!='PASS':break
 lines=[]
 for name,d in rows:
  w=d['teacher_work'];lines.append(f"{name} gate{d['gate']}: final DEV delta{d['final_nll_delta']:+.6f}nats/target; full training{d['reference_training_s']:.3f}→{d['candidate_training_s']:.3f}s; teacher{w['admitted_calls']}/{w['offered_calls']}updates, replayed{w['replayed_tokens']}tokens, scored{w['scored_targets']}targets.")
 text=' '.join(lines)+' '+rows[-1][1]['scope']+' First-pass linear event-conditioned credit pilot; retained fixed-admission quality evidence stays the leading recipe unless a completed comparable fit improves it.'
 with open('/tmp/aws-language-publication.lock','a') as f:
  fcntl.flock(f,fcntl.LOCK_EX)
  if command('git','status','--porcelain','--untracked-files=no').strip():raise RuntimeError('Preserve tracked edits; defer publication')
  command('git','pull','--rebase')
  for name,prefix in [('experiments/HANDOFF.md','**AWS residual credit integration — 13 October.** '),('report/I_SCIENCE.md','Current AWS residual credit integration. ')]:
   p=ROOT/name;parts=[x for x in p.read_text().split('\n\n') if not x.startswith((prefix,'**AWS next residual-credit integration'))]
   p.write_text(prefix+text+'\n\n'+'\n\n'.join(parts) if name.startswith('experiments/') else '\n\n'.join(parts).rstrip()+'\n\n'+prefix+text+'\n')
  command('git','add','experiments/HANDOFF.md','report/I_SCIENCE.md');command('git','commit','-m','Record bounded learned residual credit quality-work outcome')
  command('.venv-docker/bin/python','report/make_pdf.py');command('.venv-docker/bin/python','report/split_report.py')
  command('git','add','REPORT.md','report')
  if command('git','diff','--cached','--name-only').strip():command('git','commit','-m','Publish current causal teacher admission evidence')
  command('git','pull','--rebase');command('git','push','origin','main')
if __name__=='__main__':main()
