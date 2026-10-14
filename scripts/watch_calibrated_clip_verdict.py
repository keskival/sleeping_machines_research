"""Pure saved-evidence verdict; no training, inference or additional job slot."""
import subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
 expected=[ROOT/f'experiments/results/diagnostics/aws_calibrated_clip_pair_{p}_20261014_s6_v1_2m_quality.json' for p in ['global','calibrated']]
 deadline=time.monotonic()+30*86400
 while not all(p.exists() for p in expected):
  if time.monotonic()>deadline:raise TimeoutError('Inspect paired pipeline; no replacement fit admitted')
  time.sleep(20)
 subprocess.run([str(ROOT/'.venv-docker/bin/python'),'experiments/analyze_calibrated_clip_pair.py','--tag','aws_calibrated_clip_pair_verdict_20261014_v1'],cwd=ROOT,check=True)
if __name__=='__main__':main()
