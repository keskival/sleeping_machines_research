"""Publish source-bound independent-seed large-acquisition confirmation or its stop."""
import fcntl,json,statistics,time
import continue_window_replay as queue
from continue_unique_language import ROOT,command
from continue_window_replay import sha

def main():
 queue.DEADLINE=time.monotonic()+10*3600;rows=[];stop=None
 initial=queue.wait_result(ROOT/'experiments/results/diagnostics/aws_bernoulli_teacher_8m_20261013_s6_v2_quality_priority_v2.json')
 if not initial['accepted']:return
 for seed in (6,7,8):
  version='v2' if seed==6 else 'v3';tag=f'aws_bernoulli_teacher_8m_20261013_s{seed}_{version}'
  gate=queue.wait_result(ROOT/f'experiments/results/diagnostics/{tag}_gate.json')
  for name,digest in {**gate['source_sha256'],**gate['input_sha256']}.items():assert sha(ROOT/name)==digest,name
  if gate['final_nll_delta']>-.01 or gate['training_wall_ratio']>1.01:stop=f'seed{seed} quality/resource gate';break
  prefix=queue.wait_result(ROOT/f'experiments/results/diagnostics/aws_admitted_scaling_prefix_20261013_s{seed}_v1.json')
  for name,digest in {**prefix['source_sha256'],**prefix['input_sha256']}.items():assert sha(ROOT/name)==digest,name
  if prefix['gate']!='PASS':stop=f'seed{seed} recorded-prefix gate';break
  candidate=queue.wait_result(ROOT/f'experiments/results/token_language/{tag}.json')
  rows.append(dict(seed=seed,gain=-gate['final_nll_delta'],nll=candidate['curve'][-1]['dev_nll'],training_s=candidate['train_wall_s_total'],time_reduction=100*(1-gate['training_wall_ratio']),teacher_work=candidate['teacher_work']))
 if stop:text=f'Large-data confirmation stopped at {stop}; {len(rows)} fully validated seeds. Retain the completed parent recipe; diagnose this stage before further acquisition.'
 else:
  text=(f"Three8M seeds complete: finalDEV NLL{statistics.mean(x['nll'] for x in rows):.6f}±{statistics.stdev(x['nll'] for x in rows):.6f}; paired gain vs same-seed cachedfullteaching{statistics.mean(x['gain'] for x in rows):.6f}±{statistics.stdev(x['gain'] for x in rows):.6f}. Mean complete training{statistics.mean(x['training_s'] for x in rows):.3f}s, mean time reduction{statistics.mean(x['time_reduction'] for x in rows):.2f}%. Recorded smaller-acquisition prefixes exact in allthree seeds.")
 text+=' Fixed262144-token TRAIN prior,8388352fresh targets, GPT2 FineWeb, same optimizer/evaluation budgets and finalcheckpoint rule. Orderedaggregation plus probability0.25 actualalternative teaching; time savings combine both repairs. SampleSD across independentseeds. Globalclip/windowBP/denseAdam retained; public validation sealed, no external-model or ASICenergy claim.'
 with open('/tmp/aws-language-publication.lock','a') as f:
  fcntl.flock(f,fcntl.LOCK_EX)
  if command('git','status','--porcelain','--untracked-files=no').strip():raise RuntimeError('Preserve tracked edits; defer publication')
  command('git','pull','--rebase')
  for name,prefix in [('experiments/HANDOFF.md','**AWS admitted large-data confirmation — 13 October.** '),('report/I_SCIENCE.md','Current AWS admitted large-data confirmation. ')]:
   p=ROOT/name;parts=[x for x in p.read_text().split('\n\n') if not x.startswith((prefix,'**AWS reserved large-data confirmation'))]
   p.write_text(prefix+text+'\n\n'+'\n\n'.join(parts) if name.startswith('experiments/') else '\n\n'.join(parts).rstrip()+'\n\n'+prefix+text+'\n')
  command('git','add','experiments/HANDOFF.md','report/I_SCIENCE.md');command('git','commit','-m','Record large-data admitted-teacher confirmation outcome')
  command('.venv-docker/bin/python','report/make_pdf.py');command('.venv-docker/bin/python','report/split_report.py');command('git','add','REPORT.md','report')
  if command('git','diff','--cached','--name-only').strip():command('git','commit','-m','Publish current large-data confirmation evidence')
  command('git','pull','--rebase');command('git','push','origin','main')
if __name__=='__main__':main()
