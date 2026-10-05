"""Operator ledger for a small integrated training step; guarded queue only."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'experiments'))
from integrated_token_language_lab import Model
from sleeping_machines.operation_audit import OperationAudit
from sleeping_machines.paired_route_credit import paired_route_credit


class TokenAudit(OperationAudit):
    def formula(self, func, args, kwargs, out):
        name = str(func).split('.')[1].rstrip('_')
        if name == 'rsub':
            return out.numel() * (2 if kwargs.get('alpha', 1) != 1 else 1), 0, 0, 'reverse scalar subtraction'
        if name == 'index_fill':
            return 0, 0, 0, 'Indexed overwrite; memory work, no floating arithmetic'
        if name in ('exponential', 'multinomial'):
            return 0, 0, 0, 'Random sampling; draws accounted separately, work unquantified'
        return super().formula(func, args, kwargs, out)


def capture(action):
    with TokenAudit() as audit:
        value = action()
    return value, audit.result()


torch.set_num_threads(1); torch.manual_seed(6)
a = SimpleNamespace(seed=6, payload=16, depth=2, heads=2, pool=4,
                    input_init='balanced', tie_pools=False, readout='adaptive',
                    readout_init='frequency', state_mode='carry', route_credit='sampled',
                    free_bias=0., temperature=1., compiled=False)
model = Model(a, torch.arange(50257), torch.ones(50257, dtype=torch.long))
opt = torch.optim.AdamW(model.parameters(), lr=.003, weight_decay=.01)
ids = torch.tensor([[0, 2001, 10001, 30001], [30002, 10002, 2002, 1]])
targets = ids.roll(-1, dims=1)
g = torch.Generator().manual_seed(7)
before = g.get_state(); model.suppress_site = (0, 0, 0)
(x, state, pis), factual = capture(lambda: model(ids, None, g, a))
nll, readout = capture(lambda: model.readout.nll(x, targets))
pi = pis[0][1][:, 0, :].double()
winner = state['race_winners'][0, 0, :, 0]
def choose():
    eligible = 1 - torch.nn.functional.one_hot(winner, a.pool).double()
    conditional = pi.detach() * eligible
    conditional = conditional / conditional.sum(-1, keepdim=True)
    proposal = .9 * conditional + .1 * eligible / (a.pool - 1)
    alt = torch.multinomial(proposal, 1, generator=model.alternative_generator).squeeze(1)
    return alt, proposal.gather(1, alt[:, None]).squeeze(1)
(alt, q), discovery = capture(choose)
model.force_site = (0, 0, 0, alt); model.shadow_mode = True
shadow_g = torch.Generator(); shadow_g.set_state(before)
with torch.no_grad():
    (shadow, _, _), replay = capture(lambda: model(ids, None, shadow_g, a))
    shadow_loss, replay_head = capture(lambda: model.readout.nll(shadow, targets))
model.force_site = model.suppress_site = None; model.shadow_mode = False
def objective():
    difference = shadow_loss.reshape(2, 4).mean(-1).double() - nll.detach().reshape(2, 4).mean(-1).double()
    return nll.mean() + paired_route_credit(pi, alt, q, difference)
loss, utility = capture(objective)
_, backward = capture(loss.backward)
_, clipping = capture(lambda: torch.nn.utils.clip_grad_norm_(model.parameters(), 1.))
_, optimizer = capture(opt.step)
stages = dict(factual=factual, readout=readout, alternative_sampling=discovery,
              replay=replay, replay_readout=replay_head, utility=utility,
              backward=backward, clipping=clipping, optimizer_first_update=optimizer)
result = dict(status='completed', targets=8, parameters=sum(p.numel() for p in model.parameters()),
              stages=stages, complete=all(v['formula_coverage_complete'] for v in stages.values()),
              scope='Synthetic GPT-2 vocabulary with every adaptive tail exercised; P16/D2/H2/U4; one first AdamW update; random sampling work unquantified. This is an executed ledger, not whole-fit extrapolation or benchmark quality.')
p = ROOT / 'experiments/results/diagnostics/curie_integrated_token_work_audit_20261005_v2.json'
if p.exists(): raise FileExistsError(p)
p.write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps({k:v for k,v in result.items() if k != 'stages'}), flush=True)
for name, value in stages.items():
    print(name, json.dumps(value.get('unsupported_floating_operators', [])), flush=True)
