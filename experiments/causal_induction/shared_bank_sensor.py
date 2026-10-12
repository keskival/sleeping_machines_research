"""Native reciprocal memory access, independently selected by each role.

V1 uses relational destinations and native internal receiver write races.
It does not yet learn a global write address or replace deep actor backprop.
"""
import math
import torch
from torch import nn
from race_credit_sensor import RaceCreditSensor


class SharedBankSensor(RaceCreditSensor):
 def __init__(self, confidence, mode='reciprocal'):
  super().__init__(confidence)
  assert mode in ('separate', 'read_only', 'untaught', 'reciprocal')
  self.mode = mode
  p = self.core.total_payload
  self.bank_query = nn.Linear(p, p, bias=False)
  self.bank_key = nn.Linear(p, p, bias=False)
  self.critic.bank_query = nn.Linear(p, p, bias=False)
  self.critic.bank_key = nn.Linear(p, p, bias=False)
  self.bank_limit = 16

 def fresh(self):
  st = super().fresh()
  st.update(bank={}, bank_order=[], bank_version=0, bank_read_log=[])
  st['counts'].update(bank_scored_keys=0, bank_value_reads=0,
                     bank_discovery_records=0, bank_context_reads=0,
                     bank_writes=0, bank_counterfactual_values=0,
                     forward_credit_reads=0, critic_forward_reads=0,
                     bank_peak_records=0)
  return st

 def write_record(self, address, role, value, arrival, st):
  st['bank_version'] += 1
  # Keep the previous version until the newly issued write is visible.
  versions = st['bank'].setdefault(address, [])
  versions.append(dict(role=role, value=value, arrival=float(arrival),
                       version=st['bank_version']))
  st['bank'][address] = versions[-2:]
  st['bank_order'] = [address]+[a for a in st['bank_order'] if a != address]
  # Numerical capacity grows only with visited public addresses.
  c = st['counts']; c['bank_writes'] += 1
  c['bank_peak_records'] = max(c['bank_peak_records'],
                              sum(map(len, st['bank'].values())))

 def read_bank(self, role, query, t, st):
  visible = []
  for address in st['bank_order']:
   st['counts']['bank_discovery_records'] += 1
   eligible = [r for r in st['bank'][address] if r['arrival'] <= t]
   if eligible: visible.append((address, eligible[-1]))
   if len(visible) == self.bank_limit: break
  if not visible: return query.new_zeros(query.shape)
  owner = self if role == 'F' else self.critic
  q = owner.bank_query(query)
  # Credit regression cannot train observational encodings by proxy.
  transported = []
  for _, r in visible:
   source_core = self.core if r['role'] == 'F' else self.critic
   arrival = torch.full((source_core.heads,), r['arrival'], dtype=torch.float64)
   value = source_core.align(list(r['value'].split(source_core.payload)),
                             arrival, arrival.new_tensor(t), source_core.depth-1)
   transported.append(value if role == 'F' else value.detach())
  values = torch.stack(transported)
  keys = owner.bank_key(values)
  scores = (keys*q).sum(-1)/math.sqrt(query.numel())
  scores = scores.clamp(-12, 12)
  core = self.core if role == 'F' else self.critic
  if role == 'C':
   with torch.random.fork_rng():
    torch.set_rng_state(st['credit_rng'])
    selected, _, winner = core.race(scores, values)
    st['credit_rng'] = torch.get_rng_state()
  else:
   selected, _, winner = core.race(scores, values)
  record = visible[int(winner)][1]
  assert record['arrival'] <= t
  # Bound diagnostic log, independently of numerical bank capacity.
  st['bank_read_log'] = (st['bank_read_log'] +
      [(role, visible[int(winner)][0], record['version'], float(t), record['arrival'])])[-64:]
  c = st['counts']; c['bank_scored_keys'] += len(visible)
  c['bank_context_reads'] += len(visible)
  c['bank_value_reads'] += 1
  c['bank_counterfactual_values'] += len(visible) if self.training else 0
  if role == 'F' and record['role'] == 'C': c['forward_credit_reads'] += 1
  if role == 'C' and record['role'] == 'F': c['critic_forward_reads'] += 1
  return selected

 def observe(self, r, st):
  super().observe(r, st)
  value, times = st['nodes'][r['node']].contexts[0]
  value = self.core.align(list(value.split(self.core.payload)), times,
                          times.max(), self.core.depth-1)
  self.write_record(('F', r['node']), 'F', value,
                    max(r['time'], float(times.max().detach())), st)

 def context(self, node, t, st):
  value = super().context(node, t, st)
  extra = self.read_bank('F', value, t, st)
  if self.mode in ('separate', 'read_only'): return value
  if self.mode == 'untaught': extra = extra.detach()
  # Same residual bound and native actor in both write-sharing arms.
  return value + .1*extra.tanh()

 def credit_event(self, pair, t, features, kind, loss, audited, st):
  p = self.core.total_payload
  extra = self.read_bank('C', features[:p].detach(), t, st)
  if self.mode == 'separate': extra = extra*0
  token_features = torch.cat((features[:p].detach()+.1*extra.tanh(),
                              features[p:].detach()))
  # Parent detaches feature tokens: implement event explicitly to retain
  # teaching of the critic's independently learned read policy.
  local = st['credit_nodes'].setdefault(pair, self.critic.new_state())
  before = (local.candidate_scores, local.selected_updates, local.counterfactual_values)
  token = torch.cat((token_features, features.new_tensor([kind, loss, audited])))
  with torch.random.fork_rng():
   torch.set_rng_state(st['credit_rng'])
   prediction, _ = self.critic.consume_event(0, t, token, local)
   st['credit_rng'] = torch.get_rng_state()
  c = st['counts']; c['credit_events'] += 1
  c['credit_candidate_keys'] += local.candidate_scores-before[0]
  c['credit_selected_updates'] += local.selected_updates-before[1]
  c['credit_counterfactual_values'] += local.counterfactual_values-before[2]
  st['peak_credit_pairs'] = max(st['peak_credit_pairs'], len(st['credit_nodes']))
  value, times = local.contexts[0]
  value = self.critic.align(list(value.split(self.critic.payload)), times,
                            times.max(), self.critic.depth-1)
  self.write_record(('C', pair), 'C', value,
                    max(t, float(times.max().detach())), st)
  return prediction.squeeze()

 def detach(self, st):
  super().detach(st)
  for versions in st['bank'].values():
   for record in versions: record['value'] = record['value'].detach()
