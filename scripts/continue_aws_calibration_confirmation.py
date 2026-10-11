"""Finite AWS continuation: await admitted fits, queue gate, then serving/error diagnostics.
All model work goes through immutable coordinator packets and run_safe. No source mutation.
"""
import hashlib,json,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
RESULT=ROOT/'experiments/results/tpp_b4'
TAG='aws_b4_calibration_confirmation_20261011'
STATE=ROOT/'.git'/f'{TAG}.continuation.json'
DEPS=['race_tpp_logspread.py','race_tpp_logspread_marks.py','race_tpp_v19.py','race_tpp_v20.py','race_tpp_b4g.py']
def wait(tags):
 deadline=time.monotonic()+24000
 while time.monotonic()<deadline:
  pending=[]
  for tag in tags:
   p=RESULT/f'{tag}.json'
   if not p.exists():pending.append(tag);continue
   if json.loads(p.read_text())['status']!='completed':raise RuntimeError(f'Failed prerequisite: {tag}')
  if not pending:return
  print(json.dumps(dict(waiting=pending)),flush=True);time.sleep(30)
 raise RuntimeError('bounded continuation deadline reached')
def submit(source,tag,args='',deps=()):
 cmd=[str(ROOT/'.venv-docker/bin/python'),'experiments/tpp/submit.py','--slot','1','--battle','B4','--source',source,'--result-dir','experiments/results/tpp_b4','--timeout','300','--rss-kb','2000000','--purpose','Automatically assess completed fixed-seed calibration confirmation, then measure serving cost and sequence errors; no new fit or TEST','--job',f'{tag}|{args}']
 for dep in deps:cmd+=['--dep','experiments/tpp/'+dep]
 subprocess.run(cmd,cwd=ROOT,check=True);time.sleep(1.1)
def state(stage):STATE.write_text(json.dumps(dict(stage=stage,time_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())),indent=2)+'\n')
def main():
 state('waiting for admitted calibration seeds')
 wait([f'aws_b4_calibration_confirm_20261011_s{s}' for s in (1,2)])
 decision=TAG+'_decision'
 submit('experiments/tpp/decide_calibration_confirmation.py',decision);wait([decision])
 state('running serving and per-sequence diagnosis')
 tags=[]
 for s in range(3):
  base='aws_b4_logspread_develop_20261010T2112Z' if s==0 else f'aws_b4_repair_confirm_20261011_s{s}'
  marks='aws_b4_marks_develop_20261010T2300Z' if s==0 else f'aws_b4_marks_confirm_20261011_s{s}'
  tag=TAG+f'_serving_s{s}';tags.append(tag)
  submit('experiments/tpp/diagnose_calibration_serving.py',tag,f'--baseline {base} --marks {marks}',DEPS)
 wait(tags);state('completed gate and serving/error evidence; next architecture decision requires these results')
if __name__=='__main__':
 try:main()
 except Exception as e:
  state('stopped: '+repr(e));raise
