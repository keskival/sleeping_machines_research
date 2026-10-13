"""Wait for fixed cohort publication; run exactly one B10 numerical smoke."""
import json
import os
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
session = 'curie_b10_credit_confirmation_20261013_v1'
cohort = ROOT / 'experiments/results/market/curie_b10_addressed_clock_20261013_confirmation_v1.json'
while subprocess.run(['tmux', 'has-session', '-t', session], capture_output=True).returncode == 0:
    time.sleep(20)
if not cohort.exists() or json.loads(cohort.read_text())['status'] != 'completed':
    raise SystemExit('Fixed cohort did not complete; query contract not launched')
publication = ROOT / '.git/credit-report-preview/curie_b10_addressed_clock_20261013_confirmation_v1_publish/publication.json'
if not publication.exists():
    raise SystemExit('Fixed cohort publication did not complete; query contract not launched')
tag = 'curie_b10_query_clock_20261013_v1_contract'
env = dict(os.environ, MEM_CAP_KB='8000000', MEM_CAP_RSS_KB='1500000',
           MIN_AVAIL_MB='9216', CGROUP_RESERVE_MB='2048', JOB_TIMEOUT_S='300',
           WAIT='1', WAIT_TIMEOUT_S='120')
subprocess.run(['bash', 'experiments/queue/run_safe.sh', f'experiments/queue/market/{tag}.txt'],
               cwd=ROOT, env=env, check=True)
result_path = ROOT / f'experiments/results/market/{tag}.json'
result = json.loads(result_path.read_text())
assert result['status'] == 'completed' and result['null_forecast_error'] == 0
handoff = ROOT / 'experiments/HANDOFF.md'
text = handoff.read_text()
prefix = '**Curie query-conditioned native credit — 13 October, contract admitted after cohort.**'
start = text.find(prefix)
if start >= 0:
    end = text.find('\n\n', start)
    summary = ('**Curie query-conditioned native credit — 13 October, numerical contract completed.** '
               'Exact48forecast/globalRNG null parity, three TRAIN actual-Adam/paired-future smoke receipts. '
               'Measured native events6/key scores48/selected writes24 per decision; shared four-leg head, '
               'all-mode preprocessing still paid. Completed cohort quarter/teacher diagnostics stored in '
               f'{tag}.json. No quality fit admitted or TEST access; next decision is consistency repair '
               'versus economical target-conditioned acquisition based on completed cohort. '
               'Theory QUERY_CONDITIONED_NATIVE_CLOCK_CREDIT.md. All Curie jobs complete.')
    handoff.write_text(text[:start] + summary + ('' if end < 0 else text[end:]))
subprocess.run(['git', 'diff', '--check'], cwd=ROOT, check=True)
subprocess.run(['git', 'add', str(result_path.relative_to(ROOT)), 'experiments/HANDOFF.md'], cwd=ROOT, check=True)
subprocess.run(['git', 'commit', '-m', 'Validate native query credit and preserve completed cohort diagnostics'],
               cwd=ROOT, check=True)
print('Query numerical contract completed. No quality fit automatically admitted.', flush=True)
