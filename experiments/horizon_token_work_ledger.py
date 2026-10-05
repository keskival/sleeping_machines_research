"""Audit the actual horizon driver's entire fit without modifying its sources."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import horizon_token_language_engine as engine
import horizon_token_language as selection
from sleeping_machines.operation_audit import OperationAudit

class FitAudit(OperationAudit):
    active = False
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if not self.active:
            return func(*args, **(kwargs or {}))
        return super().__torch_dispatch__(func, types, args, kwargs)
    def formula(self, func, args, kwargs, out):
        name = str(func).split('.')[1].rstrip('_')
        if name == 'rsub':
            return out.numel() * (2 if kwargs.get('alpha', 1) != 1 else 1), 0, 0, 'reverse scalar subtraction'
        if name == 'index_fill':
            return 0, 0, 0, 'indexed overwrite; memory work'
        if name in ('exponential', 'multinomial'):
            return 0, 0, 0, 'random sampling; sampling work unquantified'
        return super().formula(func, args, kwargs, out)

p = argparse.ArgumentParser(add_help=False)
p.add_argument('--tag', required=True)
a, _ = p.parse_known_args()
out = ROOT / 'experiments/results/diagnostics' / (a.tag + '.work.json')
if out.exists():
    raise FileExistsError(out)
audit = FitAudit()
old_zero, old_step = torch.optim.AdamW.zero_grad, torch.optim.AdamW.step
steps = 0

def zero(self, *args, **kwargs):
    audit.active = True
    return old_zero(self, *args, **kwargs)

def step(self, *args, **kwargs):
    global steps
    value = old_step(self, *args, **kwargs)
    audit.active = False
    steps += 1
    return value

torch.optim.AdamW.zero_grad, torch.optim.AdamW.step = zero, step
try:
    with audit:
        selection.main()
finally:
    torch.optim.AdamW.zero_grad, torch.optim.AdamW.step = old_zero, old_step
    audit.active = False
result_path = ROOT / 'experiments/results/token_language' / (a.tag + '.json')
result = json.loads(result_path.read_text())
assert steps == result['args']['steps'] and result['status'] == 'completed'
ledger = audit.result()
reference = ROOT / 'experiments/results/token_language/curie_horizon_2k_batch64_credit64_20261005_v1.json'
control = json.loads(reference.read_text())
errors = [abs(x['dev_nll'] - y['dev_nll']) for x, y in zip(result['curve'], control['curve'])]
assert len(result['curve']) == len(control['curve']) and max(errors) < 2e-6
record = dict(status='completed', tag=a.tag, fitting=ledger,
              targets=result['presentations_total'], optimizer_updates=steps,
              arithmetic_flops_per_target=ledger['arithmetic_flops']/result['presentations_total'],
              curve_parity_max_error=max(errors), control_result=str(reference.relative_to(ROOT)),
              source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              scope='Actual entire fitting trajectory from zero_grad through each AdamW step: factual core, all-key scoring, alternative discovery, winner-only suffix replay, both readouts, teacher, backward, clipping and optimizer; includes in-step diagnostics. Initialization, frequency counting, evaluation, hashing and serialization excluded. Random sampling work unquantified; special functions separate. Audited wall time includes instrumentation and is not ordinary throughput. No reference fitting ledger or energy claim.')
out.write_text(json.dumps(record, indent=2) + '\n')
print(json.dumps({k:v for k,v in record.items() if k != 'fitting'}), flush=True)
print(json.dumps({k:v for k,v in ledger.items() if k != 'operators'}), flush=True)
