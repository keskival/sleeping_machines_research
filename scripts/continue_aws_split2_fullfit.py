"""Continue after the admitted full fit with source-bound quality/work decision."""
import json,time,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];path=ROOT/'experiments/results/tpp_b4/aws_b4_split2_logspread_develop_20261011.json';start=time.monotonic()
while not path.exists():
 if time.monotonic()-start>10500:raise TimeoutError('Bounded split2 full-fit deadline reached')
 time.sleep(20)
subprocess.run([str(ROOT/'.venv-docker/bin/python'),'experiments/tpp/submit.py','--slot','1','--battle','B4','--source','experiments/tpp/summarize_split2_development.py','--result-dir','experiments/results/tpp_b4','--timeout','120','--rss-kb','500000','--purpose','B4: after full repaired split2 fit, source-bound selected/common-epoch DEV quality and complete fitting/pilot/inference accounting; registered0.5 improvement/zero skip gate before mark followup, no TEST','--job','aws_b4_split2_development_decision_20261011|'],cwd=ROOT,check=True)
