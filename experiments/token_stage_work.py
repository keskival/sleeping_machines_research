"""Replay a completed token fit under full arithmetic accounting and verify parity.

Preserves the pinned engine and selection wrapper. No extrapolation from a step.
Invoke only through run_safe.sh with a unique tag and explicit completed control.
"""
import argparse
import importlib
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import horizon_token_language_engine as engine
import horizon_token_language as selection
from sleeping_machines.operation_audit import OperationAudit

class TokenAudit(OperationAudit):
    active = True
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if not self.active: return func(*args, **(kwargs or {}))
        return super().__torch_dispatch__(func, types, args, kwargs)
    def formula(self, func, args, kwargs, out):
        name = str(func).split('.')[1].rstrip('_')
        if name == 'rsub':
            return out.numel() * (2 if kwargs.get('alpha', 1) != 1 else 1), 0, 0, 'reverse scalar subtraction'
        if name == 'index_fill': return 0, 0, 0, 'indexed overwrite; memory work'
        if name in ('exponential', 'multinomial'):
            return 0, 0, 0, 'random sampling; sampling work unquantified'
        return super().formula(func, args, kwargs, out)

def main():
    global engine
    p = argparse.ArgumentParser()
    p.add_argument('--control', required=True)
    p.add_argument('--tag', required=True)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    output = Path(a.output)
    if output.exists(): raise FileExistsError(output)
    control_path = Path(a.control)
    control = json.loads(control_path.read_text())
    if control['status'] != 'completed': raise ValueError('Completed control required')
    if 'unit_gain_scale' in control['args']:
        engine = importlib.import_module('coupled_token_language_engine')
    selection.lab = engine
    for file, digest in control['identity']['source_sha256'].items():
        assert engine.sha(ROOT / file) == digest, ('Source mismatch', file)
    args = []
    for key, value in control['args'].items():
        if key in ('tag', 'resume', 'stop_after_step') or value is None: continue
        option = '--' + key.replace('_', '-')
        if isinstance(value, bool):
            if value: args.append(option)
        else: args.extend([option, str(value)])
    sys.argv = ['horizon_token_language.py', '--tag', a.tag, *args]
    fit = TokenAudit()
    fit.active = False
    old_zero, old_step = torch.optim.AdamW.zero_grad, torch.optim.AdamW.step
    updates = 0
    def zero(self, *args, **kwargs):
        fit.active = True
        return old_zero(self, *args, **kwargs)
    def step(self, *args, **kwargs):
        nonlocal updates
        value = old_step(self, *args, **kwargs)
        fit.active = False
        updates += 1
        return value
    torch.optim.AdamW.zero_grad, torch.optim.AdamW.step = zero, step
    try:
        with fit: selection.main()
    finally:
        torch.optim.AdamW.zero_grad, torch.optim.AdamW.step = old_zero, old_step
        fit.active = False
    folder = ROOT / 'experiments/results/token_language'
    result = json.loads((folder / (a.tag + '.json')).read_text())
    assert result['status'] == 'completed' and updates == control['args']['steps']
    assert result['presentations_total'] == control['presentations_total']
    assert len(result['curve']) == len(control['curve'])
    errors = [abs(x['dev_nll'] - y['dev_nll']) for x, y in zip(result['curve'], control['curve'])]
    assert max(errors) < 2e-6, errors
    chosen = json.loads((folder / (a.tag + '.selection.json')).read_text())['selected']
    settings = SimpleNamespace(**control['args'])
    train, dev = engine.load_tokens(settings.train_file), engine.load_tokens(settings.dev_file)
    counts = engine.LaneTokens(train, 0, settings.train_tokens, settings.lanes).counts()
    torch.manual_seed(settings.seed)
    model = engine.Model(settings, torch.argsort(counts, descending=True, stable=True), counts)
    checkpoint = Path(chosen['checkpoint'])
    model.load_state_dict(torch.load(checkpoint, weights_only=True))
    data = engine.interval_tensor(dev, settings.dev_offset, settings.dev_tokens, settings.eval_lanes)
    with TokenAudit() as inference:
        nll = engine.evaluate(model, data, settings)
    assert abs(nll - chosen['dev_nll']) < 2e-6
    fitting, evaluation = fit.result(), inference.result()
    targets = data.shape[0] * (data.shape[1] - 1)
    record = dict(status='completed', control=str(control_path), audited_tag=a.tag,
        fitting=fitting, inference=evaluation, fitting_targets=result['presentations_total'],
        inference_targets=targets, optimizer_updates=updates,
        fitting_arithmetic_flops_per_target=fitting['arithmetic_flops']/result['presentations_total'],
        inference_arithmetic_flops_per_target=evaluation['arithmetic_flops']/targets,
        curve_parity_max_error=max(errors), selected_dev_nll=nll,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        scope='Executed whole-fit arithmetic through every optimizer update plus exact selected development evaluation. Fitting includes factual computation, discovery, counterfactual suffix replay, readouts, teacher, backward, clipping, optimizer and in-step diagnostics; excludes initialization, frequency counts, evaluation, hashes/serialization. Evaluation includes scorer reductions; no backward/optimizer. Special functions separate, random work unquantified, not CPU instructions/physical traffic/energy. Instrumented wall is not ordinary throughput.')
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(record, indent=2)+'\n')
    print(json.dumps({k:v for k,v in record.items() if k not in ('fitting','inference')}), flush=True)
    print(json.dumps(dict(fitting_complete=fitting['formula_coverage_complete'],
                         inference_complete=evaluation['formula_coverage_complete'],
                         fitting_unknown=fitting['unsupported_floating_operators'],
                         inference_unknown=evaluation['unsupported_floating_operators'])), flush=True)

if __name__ == '__main__': main()
