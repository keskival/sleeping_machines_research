"""Publish complete or stopped independent-seed R1 teacher-admission cohort."""
import fcntl,json,statistics,time
import continue_window_replay as queue
from continue_unique_language import ROOT,command
from continue_window_replay import sha

def main():
 queue.DEADLINE=time.monotonic()+4*3600;rows=[]
 for seed in (6,7,8):
  for size in ('128k','512k'):
   version='v1' if seed==6 else 'v2'
   path=ROOT/f'experiments/results/diagnostics/aws_bernoulli_teacher_{size}_20261013_s{seed}_{version}_gate.json'
   d=queue.wait_result(path)
   for source,digest in {**d['source_sha256'],**d['input_sha256']}.items():assert sha(ROOT/source)==digest,source
   rows.append(dict(size=size,seed=seed,result=d))
   if d['gate']!='PASS':break
  if rows[-1]['result']['gate']!='PASS':break
 complete=len(rows)==6 and all(x['result']['gate']=='PASS' for x in rows)
 lines=[f"Independent-seed quality/work gates: {sum(x['result']['gate']=='PASS' for x in rows)}/{len(rows)} PASS; three-seed cohort {'complete' if complete else 'stopped'}. "]
 for size in ('128k','512k'):
  r=[x['result'] for x in rows if x['size']==size]
  if not r:continue
  nll=[x['final_nll_delta'] for x in r];reductions=[100*(1-x['training_wall_ratio']) for x in r]
  lines.append(f"{size}, {len(r)}seeds: paired final DEV delta mean{statistics.mean(nll):+.6f}"+(f"±{statistics.stdev(nll):.6f}" if len(r)>1 else '')+f"nats/target; mean complete training-time reduction{statistics.mean(reductions):.2f}%; teacher admission{sum(x['teacher_work']['admitted_calls'] for x in r)}/{sum(x['teacher_work']['offered_calls'] for x in r)}updates.")
 text=' '.join(lines)+' Same acquired data, initial member, optimizer/evaluation budget; fixed probability0.25 and inverse-probability teacher weighting. SampleSD across seeds. Temporal races, persistent state, actual alternative writes, windowBP and denseAdam retained. Complete CPU time; full arithmetic/ASIC energy not inferred from admitted teacher count. Development evidence, public validation sealed.'
 with open('/tmp/aws-language-publication.lock','a') as f:
  fcntl.flock(f,fcntl.LOCK_EX)
  if command('git','status','--porcelain','--untracked-files=no').strip():raise RuntimeError('Preserve tracked edits; defer publication')
  command('git','pull','--rebase')
  for name,prefix in [('experiments/HANDOFF.md','**AWS teacher admission confirmation — 13 October.** '),('report/I_SCIENCE.md','Current AWS teacher admission confirmation. ')]:
   p=ROOT/name;parts=[x for x in p.read_text().split('\n\n') if not x.startswith((prefix,'**AWS next teacher confirmation'))]
   p.write_text(prefix+text+'\n\n'+'\n\n'.join(parts) if name.startswith('experiments/') else '\n\n'.join(parts).rstrip()+'\n\n'+prefix+text+'\n')
  command('git','add','experiments/HANDOFF.md','report/I_SCIENCE.md');command('git','commit','-m','Record independent-seed teacher admission outcome')
  command('.venv-docker/bin/python','report/make_pdf.py');command('.venv-docker/bin/python','report/split_report.py')
  command('git','add','REPORT.md','report')
  if command('git','diff','--cached','--name-only').strip():command('git','commit','-m','Publish current teacher admission confirmation PDFs')
  command('git','pull','--rebase');command('git','push','origin','main')
if __name__=='__main__':main()
