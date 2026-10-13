"""Publish quality/resource decisions after user-revised promotion, preserving strict receipts."""
import fcntl,json,statistics,time
import continue_window_replay as queue
from continue_unique_language import ROOT,command
from continue_window_replay import sha

def publish(rows,stage):
 lines=[]
 for size in ('128k','512k','2m','8m'):
  values=[x['gate'] for x in rows if x['size']==size]
  if not values:continue
  gains=[-x['final_nll_delta'] for x in values];times=[100*(1-x['training_wall_ratio']) for x in values]
  lines.append(f"{size},{len(values)}seeds: mean finalDEV gain{statistics.mean(gains):+.6f}"+(f"±{statistics.stdev(gains):.6f}" if len(values)>1 else '')+f"nats/target; mean complete time reduction{statistics.mean(times):+.2f}%, range{min(times):+.2f}–{max(times):+.2f}%; teacher{sum(x['teacher_work']['admitted_calls'] for x in values)}/{sum(x['teacher_work']['offered_calls'] for x in values)}updates.")
 text=' '.join(lines)+' User13October quality-priority rule: accept≥0.01nats improvement with≤1% extra complete time. Negative time reduction means a measured time increase, not an efficiency win. Original strict-wall receipts preserved. Same acquisition/optimizer/evaluation budgets, fixedprobability0.25 importance-weighted actual alternative teaching, native persistent/race model, sampleSD acrossseeds. CPU time, not ASIC energy or full-FLOP claim; public validation sealed. Current completed stage: '+stage+'.'
 with open('/tmp/aws-language-publication.lock','a') as f:
  fcntl.flock(f,fcntl.LOCK_EX)
  if command('git','status','--porcelain','--untracked-files=no').strip():raise RuntimeError('Preserve tracked edits; defer publication')
  command('git','pull','--rebase')
  for name,prefix in [('experiments/HANDOFF.md','**AWS quality-priority acquisition — 13 October.** '),('report/I_SCIENCE.md','Current AWS quality-priority acquisition. ')]:
   p=ROOT/name;parts=[x for x in p.read_text().split('\n\n') if not x.startswith((prefix,'**AWS teacher admission confirmation','Current AWS teacher admission confirmation.'))]
   p.write_text(prefix+text+'\n\n'+'\n\n'.join(parts) if name.startswith('experiments/') else '\n\n'.join(parts).rstrip()+'\n\n'+prefix+text+'\n')
  command('git','add','experiments/HANDOFF.md','report/I_SCIENCE.md');command('git','commit','-m','Record quality-priority acquisition '+stage)
  command('.venv-docker/bin/python','report/make_pdf.py');command('.venv-docker/bin/python','report/split_report.py')
  command('git','add','REPORT.md','report')
  if command('git','diff','--cached','--name-only').strip():command('git','commit','-m','Publish current quality-priority learning evidence')
  command('git','pull','--rebase');command('git','push','origin','main')

def main():
 queue.DEADLINE=time.monotonic()+10*3600;rows=[]
 order=[(size,seed,'v1' if seed==6 else 'v2') for seed in (6,7,8) for size in ('128k','512k')]+[('2m',6,'v2'),('8m',6,'v2')]
 for size,seed,version in order:
  tag=f'aws_bernoulli_teacher_{size}_20261013_s{seed}_{version}'
  receipt=queue.wait_result(ROOT/f'experiments/results/diagnostics/{tag}_quality_priority_v2.json')
  gate_path=ROOT/f'experiments/results/diagnostics/{tag}_gate.json';gate=json.loads(gate_path.read_text())
  assert sha(gate_path)==receipt['input_sha256'][str(gate_path.relative_to(ROOT))]
  for source,digest in {**gate['source_sha256'],**gate['input_sha256']}.items():assert sha(ROOT/source)==digest,source
  rows.append(dict(size=size,seed=seed,gate=gate))
  if not receipt['accepted']:publish(rows,'stopped');return
  if (size=='512k' and seed==8) or size in ('2m','8m'):publish(rows,size)
if __name__=='__main__':main()
