"""Finite split2 progression: completed contract -> fit -> registered decision."""
import json,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'experiments/results/tpp_b4';PYTHON=str(ROOT/'.venv-docker/bin/python')
def wait(tag,limit):
 start=time.monotonic();p=BASE/f'{tag}.json'
 while not p.exists():
  if time.monotonic()-start>limit:raise TimeoutError(tag)
  time.sleep(20)
 return json.loads(p.read_text())
def submit(source,tag,args,purpose,timeout,rss,deps=()):
 cmd=[PYTHON,'experiments/tpp/submit.py','--slot','1','--battle','B4','--source',source,'--result-dir','experiments/results/tpp_b4','--timeout',str(timeout),'--rss-kb',str(rss),'--purpose',purpose,'--job',tag+'|'+args]
 for dep in deps:cmd+=['--dep','experiments/tpp/'+dep]
 subprocess.run(cmd,cwd=ROOT,check=True)
gate=wait('aws_b4_split2_marks_check_20261011',600);assert gate['status']=='completed' and gate['full_gradient_finite'] and gate['time_gradient_norm']==0 and gate['zero_branch_loss_error']<1e-9
submit('experiments/tpp/b4_logspread_split2_marks_develop.py','aws_b4_split2_marks_develop_20261011','--gate experiments/results/tpp_b4/aws_b4_split2_marks_check_20261011.json','B4 split2: after repaired-core gain8.314 and mark/time isolation contract, unchanged100pass/30patience mark-teacher fit; registered last20 mark gain>=1/zero skip gate; no TEST',9000,4000000,['race_tpp_logspread.py','race_tpp_logspread_marks.py','race_tpp_v20.py','race_tpp_v19.py','race_tpp_b4g.py'])
wait('aws_b4_split2_marks_develop_20261011',10500)
submit('experiments/tpp/summarize_split2_marks.py','aws_b4_split2_marks_decision_20261011','','B4 split2: compare completed original-budget mark/repaired fits using unchanged20-epoch stability gate; selected/common DEV and full fit/inference accounting, no new training/TEST',120,500000)
