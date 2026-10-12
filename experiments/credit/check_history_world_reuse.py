"""R1 delayed history receipts and acquired optimizer expansion contracts."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT/'experiments/causal_induction'))
from history_world_credit_sensor import HistoryWorldCreditSensor
from sleeping_machines.optimizer_expansion import optimizer_for_expanded_model


def optimizer_check():
    # Deliberately change registration order; positional state loading is wrong.
    class Parameters(nn.Module):
        def __init__(self, expanded=False):
            super().__init__()
            if expanded: self.extra = nn.Parameter(torch.tensor([0., 0.], dtype=torch.float64))
            self.w = nn.Parameter(torch.tensor([.2, -.3], dtype=torch.float64))
            self.b = nn.Parameter(torch.tensor([.1], dtype=torch.float64))
    source = Parameters()
    old = torch.optim.AdamW([dict(params=[source.w], lr=.01),
                            dict(params=[source.b], lr=.003)], weight_decay=.02)
    def update(model, optimizer, step):
        optimizer.zero_grad(set_to_none=True)
        model.w.grad = torch.tensor([.3+step*.01, -.2], dtype=torch.float64)
        model.b.grad = torch.tensor([.1-step*.005], dtype=torch.float64)
        optimizer.step()
    for i in range(6): update(source, old, i)
    target = Parameters(True); target.load_state_dict(source.state_dict(), strict=False)
    new = optimizer_for_expanded_model(source, old, target, new_group=1)
    assert target.extra not in new.state
    for name in ('w', 'b'):
        for key, value in old.state[getattr(source, name)].items():
            other = new.state[getattr(target, name)][key]
            assert torch.equal(value, other)
            assert value.data_ptr() != other.data_ptr()
    for i in range(6, 10):
        update(source, old, i); update(target, new, i)
        assert torch.equal(source.w, target.w) and torch.equal(source.b, target.b)
    assert new.param_groups[0]['lr'] == .01 and new.param_groups[1]['lr'] == .003
    # New component can learn without resetting any retained moments.
    new.zero_grad(set_to_none=True); target.extra.grad = torch.ones_like(target.extra); new.step()
    assert int(new.state[target.extra]['step']) == 1
    assert int(new.state[target.w]['step']) == 10
    wrong = Parameters(True)
    try: optimizer_for_expanded_model(source, old, wrong)
    except ValueError: pass
    else: raise AssertionError('Changed acquired parameters accepted')
    return dict(retained_updates=10, exact_parameter_error=0., added_parameter_initial_step=1)


def receipt_check():
    torch.manual_seed(194); model = HistoryWorldCreditSensor(.7); model.eval()
    st = model.fresh(); torch.manual_seed(319)
    for node in range(3): model.observe(dict(node=node, time=.2*node, value=node % 2, exogenous=True), st)
    tokens = []
    hook = model.history_content.register_forward_pre_hook(
        lambda module, args: tokens.append(args[0].detach().clone()))
    generator = torch.Generator().manual_seed(81)
    with torch.no_grad():
        model.issue(dict(node=0, time=2., horizon=.2, query_id=0), st, generator, 'learned', False)
        first = st['issued_history'][0].clone()
        model.issue(dict(node=0, time=3., horizon=.2, query_id=1), st, generator, 'learned', False)
        second = st['issued_history'][1].clone()
        assert first.count_nonzero() == 0 and second.count_nonzero() > 0
        model.feedback(dict(time=4., query_id=0, target=1), st, False)
        assert 0 not in st['issued_history'] and torch.equal(st['issued_history'][1], second)
        model.feedback(dict(time=5., query_id=1, target=0), st, False)
    hook.remove()
    assert len(tokens) == 4 and torch.equal(tokens[0], tokens[2]) and torch.equal(tokens[1], tokens[3])
    assert not st['issued_history'] and st['activation_history'].updates == 2
    return dict(overlapping_queries=2, feedback_token_error=0., history_updates=2)


def main():
    p = argparse.ArgumentParser(); p.add_argument('--tag', required=True); args = p.parse_args()
    torch.set_num_threads(1)
    checks = dict(optimizer_expansion=optimizer_check(), delayed_receipts=receipt_check())
    sources = set()
    for module in list(sys.modules.values()):
        filename = getattr(module, '__file__', None)
        if filename:
            path = Path(filename).resolve()
            if path.is_file() and path.is_relative_to(ROOT) and path.suffix == '.py':
                relative = path.relative_to(ROOT)
                if relative.parts[0] in ('experiments', 'sleeping_machines'):
                    sources.add(str(relative))
    result = dict(status='completed', battle='R1', checks=checks,
                  source_sha256={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in sorted(sources)},
                  scope='Numerical AdamW expansion with acquired moments and reordered parameters; real native two-outstanding-query receipt check. No task fitting, learned geometry, transfer or efficiency claim.')
    out = ROOT/f'experiments/results/diagnostics/{args.tag}.json'
    with out.open('x') as f: json.dump(result, f, indent=2); f.write('\n')
    print(json.dumps(checks), flush=True)


if __name__ == '__main__': main()
