"""Bounded B4 prospective partition continuation: contract to five-pass pilot."""
import json,time,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];path=ROOT/'experiments/results/tpp_b4/aws_b4_split2_logspread_check_20261011_v2.json';start=time.monotonic()
while not path.exists():
 if time.monotonic()-start>600:raise TimeoutError('Numerical gate not completed')
 time.sleep(10)
r=json.loads(path.read_text());assert r['status']=='completed' and r['captured']['full_gradients_finite'] and r['captured']['head_gradients_finite']
cmd=[str(ROOT/'.venv-docker/bin/python'),'experiments/tpp/submit.py','--slot','1','--battle','B4','--source','experiments/tpp/b4_logspread_split2_pilot.py','--result-dir','experiments/results/tpp_b4','--timeout','1800','--rss-kb','4000000','--purpose','B4: five fresh Github split2 TRAIN/DEV passes after full backward contract; numerical repair fixed, measure skips/quality/work before admitting longer model development; no TEST','--job','aws_b4_split2_logspread_pilot_20261011_v2|--gate experiments/results/tpp_b4/aws_b4_split2_logspread_check_20261011_v2.json']
for source in ['race_tpp_logspread.py','race_tpp_b4g.py','race_tpp_v19.py']:cmd+=['--dep','experiments/tpp/'+source]
subprocess.run(cmd,cwd=ROOT,check=True)
