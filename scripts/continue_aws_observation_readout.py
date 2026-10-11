"""Finite guarded continuation from three registered fits to their decision."""
import json,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
start=time.monotonic()
while time.monotonic()-start<900:
 paths=[ROOT/'experiments/results/tpp_b4'/f'aws_observation_anchored_readout_20261011_s{s}.json' for s in range(3)]
 if all(p.exists() for p in paths):
  assert all(json.loads(p.read_text())['status']=='completed' for p in paths)
  subprocess.run([str(ROOT/'.venv-docker/bin/python'),'experiments/tpp/submit.py','--slot','1','--battle','B4','--source','experiments/tpp/decide_observation_anchored_readout.py','--result-dir','experiments/results/tpp_b4','--timeout','120','--rss-kb','500000','--purpose','B4: apply preregistered three-seed observation-anchor quality gate before any adoption; no fitting/TEST','--job','aws_observation_readout_decision_20261011|'],cwd=ROOT,check=True);break
 time.sleep(10)
else:raise TimeoutError('Registered fit continuation exceeded900s')
