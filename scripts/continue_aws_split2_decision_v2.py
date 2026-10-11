"""Continue after the admitted five-pass pilot with a bounded decision job."""
import json,time,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];p=ROOT/'experiments/results/tpp_b4/aws_b4_split2_logspread_pilot_20261011_v2.json';start=time.monotonic()
while not p.exists():
 if time.monotonic()-start>2400:raise TimeoutError('Bounded split2 pilot deadline reached')
 time.sleep(20)
assert json.loads(p.read_text())['status']=='completed'
subprocess.run([str(ROOT/'.venv-docker/bin/python'),'experiments/tpp/submit.py','--slot','1','--battle','B4','--source','experiments/tpp/summarize_split2_logspread_pilot_v2.py','--result-dir','experiments/results/tpp_b4','--timeout','120','--rss-kb','500000','--purpose','B4: after five-pass split2 pilot, compare saved guarded epochs and finite/skip quality gates before longer fitting; no training or TEST','--job','aws_b4_split2_logspread_decision_20261011_v2|'],cwd=ROOT,check=True)
