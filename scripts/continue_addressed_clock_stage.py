"""Finish a bounded admitted B10 control, then guarded analysis/publication."""
import os,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
name='curie_b10_addressed_clock_20261013_v3_energy_fit'
log=ROOT/f'experiments/queue/market/runner_{name}.out';deadline=time.monotonic()+1200
while True:
    text=log.read_text() if log.exists() else ''
    if f'done {name} (exit 0)' in text:break
    if f'done {name} (exit ' in text:raise SystemExit('Control failed; no analysis or publication admitted.')
    if time.monotonic()>deadline:raise SystemExit('Control completion deadline expired; no publication.')
    time.sleep(10)
env=dict(os.environ,MEM_CAP_KB='8000000',MEM_CAP_RSS_KB='1500000',MIN_AVAIL_MB='9216',CGROUP_RESERVE_MB='2048',JOB_TIMEOUT_S='300',WAIT='1',WAIT_TIMEOUT_S='120')
for queue in ['curie_b10_addressed_clock_20261013_v3_energy_analysis','curie_b10_addressed_clock_20261013_v3_publish']:
    subprocess.run(['bash','experiments/queue/run_safe.sh',f'experiments/queue/market/{queue}.txt'],cwd=ROOT,env=env,check=True)
