"""Rebuild and commit report artifacts without a sampling gap for repo autosync."""
import subprocess,sys
from pathlib import Path
import build_credit_report_parts as builder

ROOT=Path(__file__).resolve().parents[1]
generated={'report/II_METHODS.md','report/III_RECORD.md','report/I_SCIENCE.pdf',
 'report/II_METHODS.pdf','report/III_RECORD.pdf','report/figures/current_native_language_status.svg',
 'report/figures/latest_native_language_fitting.svg','report/sleeping_machines_status.pdf',
 'report/sleeping_machines_status.family.text.pdf'}
unmerged=subprocess.check_output(['git','diff','--name-only','--diff-filter=U'],cwd=ROOT,text=True).splitlines()
assert set(unmerged)<=generated,unmerged
for name in unmerged:
    (ROOT/name).write_bytes(subprocess.check_output(['git','show','HEAD:'+name],cwd=ROOT))
if unmerged:subprocess.run(['git','add',*unmerged],cwd=ROOT,check=True)
builder.main()
paths=['REPORT.md','report/I_SCIENCE.md','report/II_METHODS.md','report/III_RECORD.md',
 'report/I_SCIENCE.pdf','report/II_METHODS.pdf','report/III_RECORD.pdf',
 'investment/brief/evidence_brief.tex','investment/sleeping_machines_evidence_brief.pdf',
 'scripts/build_credit_report_parts.py','scripts/finalize_factual_report.py',
 'experiments/queue/market/curie_b10_factual_report_20261013_v4.txt',
 'experiments/queue/market/curie_b10_factual_report_20261013_v5.txt',
 'experiments/queue/market/curie_b10_factual_report_20261013_v6.txt',
 'experiments/queue/market/curie_b10_factual_brief_20261013_v2.txt']
subprocess.run(['git','reset','HEAD','--','experiments/pilot/event_anomaly.py',
 'experiments/pilot/synthetic_check.py','experiments/queue/curie_language_batched_v6_20261003T111500Z.txt',
 'report/figures/latest_native_language_fitting.png'],cwd=ROOT,check=True)
subprocess.run(['git','add',*paths],cwd=ROOT,check=True)
subprocess.run(['git','commit','-m','Publish combined current report parts and compact factual-credit evidence'],cwd=ROOT,check=True)
