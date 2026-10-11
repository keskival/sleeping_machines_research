"""Finite continuation: publish the locked six-arm readout learning decision."""
import json,time,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];start=time.monotonic()
while time.monotonic()-start<900:
 paths=[ROOT/'experiments/results/tpp_b4'/f'aws_readout_curve_20261011_{arm}_s{seed}.json' for seed in range(3) for arm in ['teacher','anchor']]
 if all(p.exists() for p in paths):
  assert all(json.loads(p.read_text())['status']=='completed' for p in paths)
  subprocess.run([str(ROOT/'.venv-docker/bin/python'),'experiments/tpp/submit.py','--slot','1','--battle','B4','--source','experiments/tpp/summarize_readout_learning_curve.py','--result-dir','experiments/results/tpp_b4','--timeout','120','--rss-kb','500000','--purpose','B4: source-bound fixed-target readout learning curves, target attainment/final quality/prerequisite costs and exact final-score reproduction; no fit/TEST','--job','aws_readout_curve_decision_20261011|'],cwd=ROOT,check=True);break
 time.sleep(10)
else:raise TimeoutError('Registered curve continuation exceeded900s')
