"""R1 integrated causal history-input gate; no model fitting or TEST scoring."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'experiments/causal_induction'))
from history_world_credit_sensor import HistoryWorldCreditSensor
from world_credit_sensor import WorldCreditSensor


def main():
    p = argparse.ArgumentParser(); p.add_argument('--tag', required=True)
    a = p.parse_args(); torch.set_num_threads(1)
    torch.manual_seed(194)
    parent = WorldCreditSensor(.7)
    rng = torch.get_rng_state().clone()
    torch.manual_seed(194)
    child = HistoryWorldCreditSensor(.7)
    assert torch.equal(rng, torch.get_rng_state())
    initial = copy.deepcopy(parent.state_dict())
    for name, value in initial.items(): assert torch.equal(value, child.state_dict()[name]), name
    # Active outcome projection gives a useful gradient witness, not a fit.
    torch.manual_seed(802)
    initial['critic.head.weight'] = torch.randn_like(initial['critic.head.weight'])*.1
    initial['critic.content.weight'] = torch.randn_like(initial['critic.content.weight'])*.1
    parent.load_state_dict(initial)
    child.load_state_dict(initial, strict=False)
    def rollout(model, label):
        st = model.fresh(); model.train(); torch.manual_seed(319)
        for n in range(3): model.observe(dict(node=n, time=.2*n, value=n % 2, exogenous=True), st)
        generator = torch.Generator().manual_seed(81)
        forecasts = []; losses = []
        for i in range(2):
            model.issue(dict(node=0, time=2.+2*i, horizon=.2, query_id=i), st, generator, 'learned', True)
            forecasts.append(st['issued'][i]['world_probability'])
            loss, _ = model.feedback(dict(time=3.+2*i, query_id=i, target=label), st, True)
            losses.append(loss)
        return st, forecasts, torch.stack(losses).mean()
    _, before, _ = rollout(parent, 1)
    state, after, loss = rollout(child, 1)
    parity = max(abs(x-y) for x,y in zip(before, after))
    assert parity == 0., parity
    loss.backward()
    gradient = float(child.history_content.weight.grad.abs().max())
    assert gradient > 0 and torch.isfinite(child.history_content.weight.grad).all()
    assert state['activation_history'].updates == 2 and not state['issued_history']
    # No future label at issue: first forecast and issued feature snapshot agree.
    first = []
    for label in (0, 1):
        model = HistoryWorldCreditSensor(.7); model.load_state_dict(child.state_dict())
        _, forecasts, _ = rollout(model, label); first.append(forecasts[0])
    assert first[0] == first[1]
    # Disabled history has matched parameters and zero history-adapter gradient.
    disabled = HistoryWorldCreditSensor(.7, history_enabled=False)
    disabled.load_state_dict(child.state_dict())
    _, _, null_loss = rollout(disabled, 1); null_loss.backward()
    assert float(disabled.history_content.weight.grad.abs().max()) == 0.
    sources = ['experiments/credit/check_history_world_credit.py',
               'experiments/causal_induction/history_world_credit_sensor.py',
               'sleeping_machines/route_activation_memory.py']
    sources += [str(x.relative_to(ROOT)) for x in (ROOT/'experiments/causal_induction').glob('*.py')]
    sources += [str(x.relative_to(ROOT)) for x in (ROOT/'sleeping_machines').glob('*.py')]
    result = dict(status='completed', battle='R1', initial_forecast_error=parity,
                  history_input_gradient_max=gradient, numerical_memory=state['activation_history'].work(),
                  source_sha256={n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in sources},
                  scope='Two-query integrated numerical witness with active outcome projection, identical parent forecasts at zero history adapter, causal first forecasts, native history-input gradient and matched disabled-history control. No optimizer fit, route-similarity acquisition, transfer or quality/work claim.')
    path = ROOT/f'experiments/results/diagnostics/{a.tag}.json'
    with path.open('x') as f: json.dump(result, f, indent=2); f.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k!='source_sha256'}), flush=True)


if __name__ == '__main__': main()
