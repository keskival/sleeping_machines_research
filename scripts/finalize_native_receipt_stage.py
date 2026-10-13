"""Publish completed native receipt protocol and paired-randomness correction."""
import subprocess
from pathlib import Path
import build_credit_report_parts as builder
ROOT=Path(__file__).resolve().parents[1]
builder.main()
paths=['scripts/finalize_native_receipt_stage.py','experiments/HANDOFF.md','experiments/market/NATIVE_OPTIMIZER_RECEIPT_PROTOCOL.md','experiments/market/DELAYED_ACTION_CREDIT_PROTOCOL.md','report/market_future_credit_appendix.py','REPORT.md','report/I_SCIENCE.md','report/II_METHODS.md','report/III_RECORD.md','report/I_SCIENCE.pdf','report/II_METHODS.pdf','report/III_RECORD.pdf','report/split_manifest.json']
paths += ['experiments/credit/'+name+'.py' for name in ['check_native_optimizer_receipts','market_delayed_action_teacher_paired','market_delayed_learner_ablation_paired','market_native_receipt_fit','market_native_receipt_fit_v2','native_optimizer_receipts','native_receipt_evaluation']]
for directory,patterns in [('experiments/results/market',['curie_b10_native_receipt_20261013*.json','curie_b10_paired_delayed_action_20261013*.json']),('experiments/queue/market',['curie_b10_native_receipt_20261013*.txt','curie_b10_paired_delayed_action_20261013*.txt'])]:
    for pattern in patterns:paths.extend(str(p.relative_to(ROOT)) for p in (ROOT/directory).glob(pattern))
subprocess.run(['git','diff','--check'],cwd=ROOT,check=True)
subprocess.run(['git','add',*sorted(set(paths))],cwd=ROOT,check=True)
subprocess.run(['git','commit','-m','Integrate native optimizer receipts and repair counterfactual race pairing'],cwd=ROOT,check=True)
