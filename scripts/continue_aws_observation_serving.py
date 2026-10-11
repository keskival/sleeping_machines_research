"""After the locked quality decision, issue serving checks only on a passing arm."""
import json,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];path=ROOT/'experiments/results/tpp_b4/aws_observation_readout_decision_20261011.json';start=time.monotonic()
while not path.exists():
 if time.monotonic()-start>900:raise TimeoutError('Decision not completed')
 time.sleep(10)
r=json.loads(path.read_text());assert r['status']=='completed'
if r['gate']:
 cmd=[str(ROOT/'.venv-docker/bin/python'),'experiments/tpp/submit.py','--slot','1','--battle','B4','--source','experiments/tpp/check_observation_readout_serving.py','--result-dir','experiments/results/tpp_b4','--timeout','180','--rss-kb','1500000','--purpose','B4: after registered observation-anchor gate passes, validate teacher-free portable bundle scores/time/normalization/suffix causality before adoption; no fitting/TEST','--job','aws_observation_readout_serving_20261011|']
 for source in ['nonlinear_single_mark_readout.py','single_mark_readout.py','race_tpp_logspread.py','race_tpp_v19.py','race_tpp_b4g.py']:cmd+=['--dep','experiments/tpp/'+source]
 subprocess.run(cmd,cwd=ROOT,check=True)
else:print('Registered gate failed: retain previous student; no serving/adoption run')
