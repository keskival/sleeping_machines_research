"""Publish completed B10 delayed learner experiments and owned evidence."""
import subprocess
from pathlib import Path
import build_credit_report_parts as builder
ROOT=Path(__file__).resolve().parents[1]
builder.main()
paths=['scripts/finalize_delayed_action_stage.py','experiments/HANDOFF.md','experiments/THEORY.md','experiments/theory/DELAYED_FINITE_LEARNER_CONSEQUENCES.md','experiments/theory/OPTIMIZER_CONTEXT_IDENTIFIABILITY.md','experiments/market/DELAYED_ACTION_CREDIT_PROTOCOL.md','report/market_future_credit_appendix.py','REPORT.md','report/I_SCIENCE.md','report/II_METHODS.md','report/III_RECORD.md','report/I_SCIENCE.pdf','report/II_METHODS.pdf','report/III_RECORD.pdf','report/split_manifest.json','experiments/credit/market_optimizer_context_alias.py','experiments/credit/analyze_market_delayed_action.py']
for directory,pattern in [('experiments/credit','market_delayed*.py'),('experiments/results/market','curie_b10_delayed_action*.json'),('experiments/results/market','curie_b10_optimizer_context_alias*.json'),('experiments/queue/market','curie_b10_delayed_action*.txt'),('experiments/queue/market','curie_b10_optimizer_context_alias*.txt')]:paths.extend(str(p.relative_to(ROOT)) for p in (ROOT/directory).glob(pattern))
subprocess.run(['git','diff','--check'],cwd=ROOT,check=True)
subprocess.run(['git','add',*sorted(set(paths))],cwd=ROOT,check=True)
subprocess.run(['git','commit','-m','Train delayed native credit from learner consequences and identify pooled optimizer input ambiguity'],cwd=ROOT,check=True)
