"""Bounded sequential B10 parent confirmation; all quality outcomes retained."""
import json,os,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
for member in ['addressed','uniform']:
    p=ROOT/f'experiments/results/market/curie_b10_addressed_clock_20261013_confirm_smoke_v1_{member}.json';d=json.loads(p.read_text());assert d['status']=='completed';assert all(r['paired_zero_action_parity']==0 and r['first_future_parity']<1e-9 for r in d['acquisition']['rows'])
env=dict(os.environ,MEM_CAP_KB='8000000',MEM_CAP_RSS_KB='1500000',MIN_AVAIL_MB='9216',CGROUP_RESERVE_MB='2048',JOB_TIMEOUT_S='900',WAIT='1',WAIT_TIMEOUT_S='120')
for seed in [19129,19130]:
    for member in ['addressed','uniform']:
        tag=f'curie_b10_addressed_clock_20261013_confirm_s{seed}_v1_{member}'
        subprocess.run(['bash','experiments/queue/run_safe.sh',f'experiments/queue/market/{tag}.txt'],cwd=ROOT,env=env,check=True)
        d=json.loads((ROOT/f'experiments/results/market/{tag}.json').read_text());assert d['status']=='completed' and d['reference_saved_error']<1e-12
# No quality-dependent admission: the complete cohort is the decision unit.
env['JOB_TIMEOUT_S']='300'
for tag in ['curie_b10_addressed_clock_20261013_confirmation_v1','curie_b10_addressed_clock_20261013_confirmation_v1_publish']:
    subprocess.run(['bash','experiments/queue/run_safe.sh',f'experiments/queue/market/{tag}.txt'],cwd=ROOT,env=env,check=True)
