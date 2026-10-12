"""B5/R1 native shared-bank contracts and bounded development comparison."""
import argparse
import hashlib
import json
import resource
import time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from native_sensor import ROOT
from shared_bank_sensor_v2 import SharedBankSensor
from credit_run import run
from public_stream import persistence


def witness(mode, initial):
 m = SharedBankSensor(.7, mode); m.load_state_dict(initial); m.train()
 st = m.fresh(); torch.manual_seed(531)
 for i in range(2):
  m.observe(dict(node=i, time=i*.2, value=i, exogenous=True), st)
 features = torch.linspace(-1, 1, m.critic.content_dim-3)
 m.credit_event((0, 1), .4, features, 1., .3, 1., st)
 generator = torch.Generator().manual_seed(42)
 for i in range(8):
  m.issue(dict(node=i%2, time=2.+i*.1, horizon=.2, query_id=i), st,
          generator, 'credit', False)
 logits = torch.stack([st['issued'][i]['logit'] for i in range(8)])
 loss = F.binary_cross_entropy_with_logits(logits, torch.ones_like(logits))
 loss.backward()
 def norm(p): return float(p.grad.abs().max()) if p.grad is not None else 0.
 return dict(logit=logits.detach().tolist(),
             critic_write_gradient=norm(m.critic.units[0][0][0][0].input.weight),
             critic_read_gradient=norm(m.critic.bank_key.weight),
             actor_read_gradient=norm(m.bank_key.weight),
             counts=st['counts'], read_log=st['bank_read_log'])


def main():
 ap = argparse.ArgumentParser(); ap.add_argument('--tag', required=True)
 ap.add_argument('--mode', choices=['contract', 'fit'], required=True)
 args = ap.parse_args(); torch.set_num_threads(1)
 out = ROOT/f'experiments/results/causal_induction/{args.tag}.json'
 assert not out.exists(); start = time.monotonic()
 parent = ROOT/'experiments/results/causal_induction/curie_causal_induction_20261012_v1_prepare.json'
 data = json.loads(parent.read_text())
 def path(row):
  p = ROOT/row['path']; assert hashlib.sha256(p.read_bytes()).hexdigest() == row['sha256']; return p
 train = [r for r in data['records'] if r['split'] == 'train'][:8]
 dev = [r for r in data['records'] if r['split'] == 'heldout_graph' and r['variant'] == 'base'][:4]
 prior = (1., 2.); prior_start = time.monotonic()
 for row in train: _, prior = persistence(path(row), prior, True)
 prior_wall = time.monotonic()-prior_start; confidence = prior[0]/prior[1]
 torch.manual_seed(194); initial = SharedBankSensor(confidence).state_dict()
 rows = []; checks = {}
 if args.mode == 'contract':
  # Activate the local output path solely for the mechanism witness;
  # ordinary fitting keeps the validated zero-residual initialization.
  torch.manual_seed(801)
  initial['sensor.net.2.weight'] = torch.randn_like(initial['sensor.net.2.weight'])*.1
  taught = witness('reciprocal', initial); untaught = witness('untaught', initial)
  assert taught['logit'] == untaught['logit']
  assert taught['critic_write_gradient'] > 0, taught
  assert taught['critic_read_gradient'] > 0, taught
  assert untaught['critic_write_gradient'] == 0
  assert untaught['critic_read_gradient'] == 0
  assert untaught['actor_read_gradient'] > 0
  assert taught['counts']['forward_credit_reads'] > 0
  assert taught['counts']['critic_forward_reads'] > 0
  m = SharedBankSensor(confidence); st = m.fresh(); m.eval()
  m.observe(dict(node=0, time=0., value=1, exogenous=True), st)
  torch.manual_seed(998); before = m.context(0, 2., st).detach()
  m.write_record(('C', (99, 99)), 'C', torch.ones(16)*999, 99., st)
  torch.manual_seed(998); after = m.context(0, 2., st).detach()
  assert torch.equal(before, after)
  assert all(arrival <= t for _, _, _, t, arrival in st['bank_read_log'])
  # Full existing causal update schedule checks truncation/version lifetime.
  torch.manual_seed(194); m = SharedBankSensor(confidence)
  smoke = run(m, path(train[0]), 'credit', torch.optim.Adam(m.parameters(), lr=.001), seed=194)
  checks = dict(task_loss_teaches_native_critic_reads_and_writes=taught,
                matched_untaught_control=untaught, future_record_invisible=True,
                issued_prediction_parity=True, smoke=smoke)
 else:
  contract = ROOT/'experiments/results/causal_induction/curie_shared_bank_20261012_v3_contract.json'
  certificate = json.loads(contract.read_text()); assert certificate['status'] == 'completed'
  for name, digest in certificate['source_sha256'].items():
   assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == digest
  for mode in ('separate', 'read_only', 'untaught', 'reciprocal'):
   m = SharedBankSensor(confidence, mode); m.load_state_dict(initial)
   opt = torch.optim.Adam(m.parameters(), lr=.001); acq = []
   for epoch in range(2):
    for i, row in enumerate(train):
     acq.append(dict(epoch=epoch, **run(m, path(row), 'credit', opt, seed=194+epoch*100+i)))
   final = [run(m, path(row), 'credit', seed=900+i) for i, row in enumerate(dev)]
   ckpt = out.with_name(out.stem+'_'+mode+'.pt')
   torch.save(dict(model=m.state_dict(), optimizer=opt.state_dict(), confidence=confidence), ckpt)
   row = dict(arm=mode, train=acq, dev=final,
              mean_dev_nll=float(np.mean([r['nll'] for r in final])),
              whole_fit_wall_s=prior_wall+sum(r['wall_s'] for r in acq),
              checkpoint=str(ckpt.relative_to(ROOT)),
              checkpoint_sha256=hashlib.sha256(ckpt.read_bytes()).hexdigest())
   rows.append(row); print(json.dumps({k:row[k] for k in ('arm','mean_dev_nll','whole_fit_wall_s')}), flush=True)
 sources = [Path(__file__), ROOT/'experiments/causal_induction/shared_bank_sensor_v2.py']
 sources += [ROOT/'experiments/causal_induction'/n for n in
             ('race_credit_sensor.py','structural_sensor.py','native_sensor.py','credit_run.py','public_stream.py')]
 sources += [ROOT/'sleeping_machines'/n for n in
             ('addressed_event_heads.py','paired_route_credit.py','parallel_head_race_language.py','sparse_race_language.py')]
 result = dict(status='completed', battle='B5/R1', tag=args.tag, mode=args.mode,
               rows=rows, checks=checks, prior_acquisition_targets=512,
               whole_pipeline_wall_s=time.monotonic()-start,
               peak_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
               parent_sha256=hashlib.sha256(parent.read_bytes()).hexdigest(),
               source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
               scope='Private affine development. Independent native hard-race shared reads; critic native receiver writes exposed as versioned relational records. Subsequent task loss teaches critic memory/read parameters through shared records within eight-feedback graph windows; numeric state persists. Bank destinations remain role/node or role/pair, not globally learned. Local native route derivative is a declared counterfactual surrogate, not exact expected hard-route gradient. All candidate payload transports/scans charged. No publicTEST, neural-depth scaling, full-future intervention teaching, optimizer meta-gradient or efficiency claim.')
 out.write_text(json.dumps(result, indent=2)+'\n'); print(json.dumps(dict(tag=args.tag,checks=checks)), flush=True)


if __name__ == '__main__': main()
