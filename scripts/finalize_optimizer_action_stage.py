"""Publish completed B10 action geometry and commit owned artifacts on main."""
import subprocess
from pathlib import Path
import build_credit_report_parts as builder
ROOT=Path(__file__).resolve().parents[1]
builder.main()
paths=['scripts/finalize_optimizer_action_stage.py','experiments/HANDOFF.md','experiments/THEORY.md','experiments/theory/CAUSAL_ACTION_SPACE_CREDIT.md','experiments/market/OPTIMIZER_ACTION_GEOMETRY_PROTOCOL.md','report/market_future_credit_appendix.py','REPORT.md','report/I_SCIENCE.md','report/II_METHODS.md','report/III_RECORD.md','report/I_SCIENCE.pdf','report/II_METHODS.pdf','report/III_RECORD.pdf','report/split_manifest.json']
for name in ['market_optimizer_transport.py','market_action_geometry.py','market_action_geometry_fit.py','market_action_geometry_full.py','market_action_geometry_confirmation.py','market_adam_action_tangent.py','market_sparse_action_metric.py','market_sparse_action_adapter.py','check_market_adam_action_tangent.py','check_market_sparse_action_metric.py','analyze_market_action_geometry.py']:
    paths.append('experiments/credit/'+name)
for directory,pattern in [('experiments/results/market','curie_b10_action*.json'),('experiments/results/market','curie_b10_sparse_action*.json'),('experiments/results/market','curie_b10_optimizer_transport*.json'),('experiments/queue/market','curie_b10_action*.txt'),('experiments/queue/market','curie_b10_sparse_action*.txt'),('experiments/queue/market','curie_b10_optimizer_transport*.txt')]:
    paths.extend(str(p.relative_to(ROOT)) for p in (ROOT/directory).glob(pattern))
subprocess.run(['git','diff','--check'],cwd=ROOT,check=True)
subprocess.run(['git','add',*sorted(set(paths))],cwd=ROOT,check=True)
subprocess.run(['git','commit','-m','Integrate sparse optimizer credit geometry and reject unconfirmed teaching gains'],cwd=ROOT,check=True)
