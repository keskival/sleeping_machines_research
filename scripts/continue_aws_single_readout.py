"""Finite automatic B4 follow-up: await fixed-seed readouts, queue decision."""
import json,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];STATE=ROOT/'.git/aws_single_readout_20261011.continuation.json'
def main():
 STATE.write_text(json.dumps(dict(status='waiting for three completed fixed readouts'))+'\n')
 deadline=time.monotonic()+1200
 while time.monotonic()<deadline:
  paths=[ROOT/'experiments/results/tpp_b4'/f'aws_nonlinear_mark_readout_20261011_s{s}.json' for s in range(3)]
  if not all(p.exists() for p in paths):time.sleep(10);continue
  assert all(json.loads(p.read_text())['status']=='completed' for p in paths)
  subprocess.run([str(ROOT/'.venv-docker/bin/python'),'experiments/tpp/submit.py','--slot','3','--battle','B4','--source','experiments/tpp/decide_single_readout.py','--result-dir','experiments/results/tpp_b4','--timeout','120','--rss-kb','500000','--purpose','Automatically publish strict fixed-recipe three-seed single-core readout decision after completed fits, no TEST or further fitting','--job','aws_single_readout_decision_20261011|'],cwd=ROOT,check=True)
  STATE.write_text(json.dumps(dict(status='decision queued through run_safe'))+'\n');return
 raise TimeoutError('bounded readout continuation deadline')
if __name__=='__main__':
 try:main()
 except Exception as e:STATE.write_text(json.dumps(dict(status='stopped',reason=repr(e)))+'\n');raise
