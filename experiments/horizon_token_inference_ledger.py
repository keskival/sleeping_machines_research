"""Audit selected native inference on exact development targets; guarded queue."""
import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import horizon_token_language_engine as lab
from sleeping_machines.operation_audit import OperationAudit

class InferenceAudit(OperationAudit):
    def formula(self, func, args, kwargs, out):
        name = str(func).split('.')[1].rstrip('_')
        if name == 'rsub':
            return out.numel() * (2 if kwargs.get('alpha', 1) != 1 else 1), 0, 0, 'reverse scalar subtraction'
        if name == 'index_fill':
            return 0, 0, 0, 'indexed overwrite; memory work'
        if name in ('exponential', 'multinomial'):
            return 0, 0, 0, 'random sampling; sampling work unquantified'
        return super().formula(func, args, kwargs, out)

p = argparse.ArgumentParser()
p.add_argument('--output', required=True)
a = p.parse_args()
out = Path(a.output)
if out.exists(): raise FileExistsError(out)
torch.set_num_threads(1)
folder = ROOT / 'experiments/results/token_language'
tag = 'curie_horizon_2k_batch64_credit64_20261005_v1'
r = json.loads((folder / (tag + '.json')).read_text())
s = json.loads((folder / (tag + '.selection.json')).read_text())['selected']
args = SimpleNamespace(**r['args'])
train, dev = lab.load_tokens(args.train_file), lab.load_tokens(args.dev_file)
counts = lab.LaneTokens(train, 0, args.train_tokens, args.lanes).counts()
torch.manual_seed(args.seed)
model = lab.Model(args, torch.argsort(counts, descending=True, stable=True), counts)
checkpoint = Path(s['checkpoint'])
model.load_state_dict(torch.load(checkpoint, weights_only=True))
data = lab.interval_tensor(dev, args.dev_offset, args.dev_tokens, args.eval_lanes)
with InferenceAudit() as audit:
    nll = lab.evaluate(model, data, args)
assert abs(nll - s['dev_nll']) < 2e-6
ledger = audit.result()
targets = data.shape[0] * (data.shape[1] - 1)
record = dict(status='completed', tag=tag, targets=targets, dev_nll=nll,
    checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
    inference=ledger, arithmetic_flops_per_target=ledger['arithmetic_flops']/targets,
    scope='Actual selected model development evaluation, no gradient or optimizer, same stochastic RNG/history/scored-target protocol. Includes model, exact target NLL and scorer reductions; excludes construction, checkpoint loading and frequency counts. Arithmetic and special functions separate; random sampling work unquantified. Not full vocabulary generation, physical traffic or energy.')
out.write_text(json.dumps(record, indent=2) + '\n')
print(json.dumps({k:v for k,v in record.items() if k != 'inference'}), flush=True)
print(json.dumps({k:v for k,v in ledger.items() if k != 'operators'}), flush=True)
