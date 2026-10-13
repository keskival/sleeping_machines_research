"""Publish completed signed future-consequence native development."""
import subprocess
from pathlib import Path
import build_credit_report_parts as builder
ROOT=Path(__file__).resolve().parents[1]
builder.main()
paths=['scripts/finalize_native_antithetic_stage.py','experiments/HANDOFF.md','experiments/THEORY.md','experiments/theory/ANTITHETIC_NATIVE_CREDIT_CONSEQUENCES.md','experiments/credit/market_native_receipt_antithetic.py','experiments/credit/analyze_native_antithetic.py','report/market_future_credit_appendix.py','REPORT.md','report/I_SCIENCE.md','report/II_METHODS.md','report/III_RECORD.md','report/I_SCIENCE.pdf','report/II_METHODS.pdf','report/III_RECORD.pdf','report/split_manifest.json']
for directory,pattern in [('experiments/results/market','curie_b10_native_antithetic_20261013_v1*.json'),('experiments/queue/market','curie_b10_native_antithetic_20261013_v1*.txt')]:paths.extend(str(p.relative_to(ROOT)) for p in (ROOT/directory).glob(pattern))
subprocess.run(['git','diff','--check'],cwd=ROOT,check=True)
subprocess.run(['git','add',*sorted(set(paths))],cwd=ROOT,check=True)
subprocess.run(['git','commit','-m','Test signed native future credit and identify optimizer saturation boundary'],cwd=ROOT,check=True)
