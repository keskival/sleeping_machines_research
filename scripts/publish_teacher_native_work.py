"""Publish completed native teacher arithmetic, with explicit covered-work boundary."""
import fcntl,json,time
import continue_window_replay as queue
from continue_unique_language import ROOT,command
from continue_window_replay import sha

def main():
 queue.DEADLINE=time.monotonic()+10*3600
 path=ROOT/'experiments/results/diagnostics/aws_teacher_native_work_decision_20261013_v1.json';d=queue.wait_result(path)
 for name,digest in {**d['source_sha256'],**d['input_sha256']}.items():assert sha(ROOT/name)==digest,name
 before,after=d['rows'];removed=d['removed_work'];text=(f"8192-target matched-acquisition probe, full/admitted teaching: registered fit subtotal{before['registered_fit_flops_subtotal']/1e9:.6f}→{after['registered_fit_flops_subtotal']/1e9:.6f}GFLOPs; declared scalar arithmetic{before['declared_fit_scalar_arithmetic']}→{after['declared_fit_scalar_arithmetic']}; logical fitting-output payload{before['logical_fit_output_bytes']}→{after['logical_fit_output_bytes']}bytes. "+d['scope'])
 with open('/tmp/aws-language-publication.lock','a') as f:
  fcntl.flock(f,fcntl.LOCK_EX)
  if command('git','status','--porcelain','--untracked-files=no').strip():raise RuntimeError('Preserve tracked edits; defer publication')
  command('git','pull','--rebase')
  for name,prefix in [('experiments/HANDOFF.md','**AWS admitted teacher native work — 13 October.** '),('report/I_SCIENCE.md','Current AWS admitted teacher native work. ')]:
   p=ROOT/name;parts=[x for x in p.read_text().split('\n\n') if not x.startswith(prefix)];p.write_text(prefix+text+'\n\n'+'\n\n'.join(parts) if name.startswith('experiments/') else '\n\n'.join(parts).rstrip()+'\n\n'+prefix+text+'\n')
  command('git','add','experiments/HANDOFF.md','report/I_SCIENCE.md');command('git','commit','-m','Record native arithmetic effect of admitted counterfactual teaching')
  command('.venv-docker/bin/python','report/make_pdf.py');command('.venv-docker/bin/python','report/split_report.py');command('git','add','REPORT.md','report')
  if command('git','diff','--cached','--name-only').strip():command('git','commit','-m','Publish current native teacher work evidence')
  command('git','pull','--rebase');command('git','push','origin','main')
if __name__=='__main__':main()
