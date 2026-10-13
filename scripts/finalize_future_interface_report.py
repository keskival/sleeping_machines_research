"""Build current report parts and commit the source-bound B10 stage together."""
import subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
assert not subprocess.check_output(['git','diff','--name-only','--diff-filter=U'],cwd=ROOT,text=True).strip()
log=ROOT/'.git/credit-report-preview/curie_b10_future_interface_20261013_v1'
log.mkdir(parents=True,exist_ok=True)
with (log/'split.log').open('w') as f:
 subprocess.run([str(ROOT/'.venv-docker/bin/python'),'report/split_report.py'],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True)
paths=['REPORT.md','experiments/HANDOFF.md','experiments/THEORY.md','report/split_manifest.json',
 'report/II_METHODS.md','report/III_RECORD.md','report/I_SCIENCE.pdf','report/II_METHODS.pdf','report/III_RECORD.pdf',
 'experiments/theory/FORWARD_TAUGHT_STATE_INTERFACE_CREDIT.md','experiments/market/FUTURE_STATE_INTERFACE_PROTOCOL.md',
 'experiments/credit/market_future_interface.py','experiments/credit/market_future_interface_scaled.py',
 'experiments/credit/market_future_interface_geometry.py','experiments/credit/analyze_market_future_interface.py',
 'scripts/finalize_future_interface_report.py']
paths += [str(p.relative_to(ROOT)) for p in (ROOT/'experiments/queue/market').glob('curie_b10_future_interface*_20261013_v1_*.txt')]
paths += [str(p.relative_to(ROOT)) for p in (ROOT/'experiments/results/market').glob('curie_b10_future_interface*_20261013_v1_*.json')]
subprocess.run(['git','add',*paths],cwd=ROOT,check=True)
subprocess.run(['git','commit','-m','Acquire native future-state credit from forward-only teaching and audit scale ablation'],cwd=ROOT,check=True)
