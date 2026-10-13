"""Regenerate current report parts and commit the completed B10 stage."""
import subprocess
from pathlib import Path
import build_credit_report_parts as builder
ROOT=Path(__file__).resolve().parents[1]
assert not subprocess.check_output(['git','diff','--name-only','--diff-filter=U'],cwd=ROOT,text=True).strip()
builder.main()
paths=['REPORT.md','experiments/HANDOFF.md','experiments/THEORY.md','experiments/PRODUCT_ORDERS.md',
 'experiments/WIN_CRITERIA.md','report/I_SCIENCE.md','report/II_METHODS.md','report/III_RECORD.md',
 'report/I_SCIENCE.pdf','report/II_METHODS.pdf','report/III_RECORD.pdf',
 'report/optimizer_transport_appendix.py','report/market_future_credit_appendix.py',
 'investment/brief/evidence_brief.tex','investment/forward_credit_evidence_20261011.json',
 'investment/sleeping_machines_evidence_brief.pdf','scripts/prepare_forward_credit_evidence.py',
 'scripts/finalize_market_future_credit_stage.py',
 'experiments/market/FUTURE_CREDIT_ACTUAL_UPDATE_PROTOCOL.md',
 'experiments/theory/STATE_REFRESH_AND_ACTUAL_UPDATE_CREDIT.md',
 'experiments/theory/RANDOM_RAY_FORWARD_CREDIT_TEACHING.md',
 'experiments/credit/analyze_market_future_updates.py']
for directory,pattern in [('experiments/credit','market_future*.py'),('experiments/credit','publish_market_future_credit*.py'),
 ('experiments/results/market','curie_b10_future*.json'),('experiments/queue/market','curie_b10_future*.txt')]:
 paths.extend(str(p.relative_to(ROOT)) for p in (ROOT/directory).glob(pattern))
paths=sorted(set(paths))
subprocess.run(['git','add',*paths],cwd=ROOT,check=True)
subprocess.run(['git','commit','-m','Confirm native forward-taught credit quality gains and fixed-data learning tradeoffs'],cwd=ROOT,check=True)
