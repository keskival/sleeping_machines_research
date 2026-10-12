"""B5/R1 native outcome-law acquisition and independent causal reuse arms."""
import argparse
import copy
import hashlib
import json
import math
import resource
import time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from native_sensor import ROOT
from public_stream import persistence
from world_credit_sensor import WorldCreditSensor
from world_credit_run import run

CONTRACT = 'curie_world_credit_20261012_v1_contract'
ACQUISITION = 'curie_world_credit_20261012_v1_acquisition'
PARENT = ROOT/'experiments/results/causal_induction/curie_causal_nonlinear_20261012_v2_prepare.json'
RESULTS = ROOT/'experiments/results/causal_induction'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fingerprint(value):
    digest = hashlib.sha256()
    def visit(x):
        if torch.is_tensor(x):
            digest.update(str((x.dtype, tuple(x.shape))).encode())
            digest.update(x.detach().cpu().contiguous().numpy().tobytes())
        elif isinstance(x, dict):
            for key in sorted(x, key=lambda k: (type(k).__name__, repr(k))):
                digest.update(repr(key).encode()); visit(x[key])
        elif isinstance(x, (list, tuple)):
            for v in x: visit(v)
        else: digest.update(repr(x).encode())
    visit(value)
    return digest.hexdigest()


def verify_result(tag):
    path = RESULTS/(tag+'.json')
    result = json.loads(path.read_text())
    assert result['status'] == 'completed'
    for name, digest in result['source_sha256'].items():
        assert sha(ROOT/name) == digest, name
    assert sha(PARENT) == result['parent_sha256']
    for row in result['data_records']:
        assert sha(ROOT/row['path']) == row['sha256']
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', required=True)
    ap.add_argument('--mode', choices=['contract', 'acquisition', 'reuse'], required=True)
    ap.add_argument('--arm', choices=['exact', 'learned', 'prior'])
    a = ap.parse_args()
    torch.set_num_threads(1)
    begin = time.monotonic()
    out = RESULTS/(a.tag+'.json')
    assert not out.exists()
    data = json.loads(PARENT.read_text())
    assert data['status'] == 'completed'
    train = [r for r in data['records'] if r['split'] == 'train' and r['variant'] == 'base']
    assert len(train) == 16
    cases = []
    for split in ['heldout_graph', 'heldout_factors', 'heldout_depth', 'heldout_factors_depth']:
        base = [r for r in data['records'] if r['split'] == split and r['variant'] == 'base']
        rewire = [r for r in data['records'] if r['split'] == split and r['variant'] == 'rewire']
        cases.extend([base[0], rewire[1]])
    def path(row):
        p = ROOT/row['path']; assert sha(p) == row['sha256']; return p
    prior = (1., 2.)
    start = time.monotonic()
    for r in train:
        _, prior = persistence(path(r), prior, True)
    confidence_wall = time.monotonic()-start
    confidence = prior[0]/prior[1]
    torch.manual_seed(194)
    initial = copy.deepcopy(WorldCreditSensor(confidence).state_dict())
    rows, checks = [], {}
    acquisition = None
    if a.mode == 'contract':
        # Proper-loss projection and coherent differences, including saturation.
        z = torch.tensor([-20., -2., .3, 3., 20.], dtype=torch.float64)
        q = .31
        risk = F.softplus(z)-q*z
        identity_error = 0.
        for y in [0., 1.]:
            actual = F.binary_cross_entropy_with_logits(z, z.new_full(z.shape, y), reduction='none')
            projected = F.softplus(z)-y*z
            identity_error = max(identity_error, float((actual-projected).abs().max()))
        cycle = WorldCreditSensor.contrast(z[0], z[1], q)+WorldCreditSensor.contrast(z[1], z[2], q)-WorldCreditSensor.contrast(z[0], z[2], q)
        assert identity_error < 1e-12 and abs(float(cycle)) < 1e-12
        # Enumerate labels for the declared fixed blend's conditional MSE.
        truth_q, issued_q = .27, .38
        dz = z[3]-z[1]
        expected_mse = sum((truth_q if y else 1-truth_q)*float((.5*(y+issued_q)-truth_q)**2*dz**2) for y in [0., 1.])
        formula = .25*((issued_q-truth_q)**2+truth_q*(1-truth_q))*float(dz**2)
        assert abs(expected_mse-formula) < 1e-12
        active = copy.deepcopy(initial)
        torch.manual_seed(802)
        active['sensor.net.2.weight'] = torch.randn_like(active['sensor.net.2.weight'])*.1
        active['critic.head.weight'] = torch.randn_like(active['critic.head.weight'])*.1
        def fixture(mode='reciprocal'):
            model = WorldCreditSensor(confidence, mode); model.load_state_dict(active)
            model.train(); st = model.fresh(); torch.manual_seed(319)
            for n in range(3):
                model.observe(dict(node=n, time=.2*n, value=n % 2, exogenous=True), st)
            return model, st
        m, st = fixture()
        features = torch.linspace(-1, 1, m.critic.content_dim-3)
        predicted = m.world_event(0, 2., features, 0., 0., st)
        F.binary_cross_entropy_with_logits(predicted, predicted.new_tensor(1.)).backward()
        body_grad = max(float(p.grad.abs().max()) for n, p in m.critic.named_parameters()
                        if not n.startswith('head.') and p.grad is not None)
        read_grad = float(m.critic.bank_query.weight.grad.abs().max())
        assert body_grad > 0 and read_grad > 0
        # Actual native world writes are consumed by a later forward task.
        def shared_task(mode):
            model, state = fixture(mode)
            model.world_event(0, 2., features, 0., 0., state)
            address = ('C', (0, 0))
            state['bank'] = {address: state['bank'][address]}
            state['bank_order'] = [address]; state['bank_by_node'] = {0: [address]}
            future = model.context(0, 100., state)
            # A genuine forward sensor forecast, not a direct norm of C state.
            token = torch.cat((future, future.new_zeros(16), future.new_tensor([1., 0., 0.]),
                               future.new_zeros(2), future.new_zeros(4)))
            output = model.sensor(token).squeeze()
            F.binary_cross_entropy_with_logits(output, output.new_tensor(0.)).backward()
            gradient = max((float(p.grad.abs().max()) for n, p in model.critic.named_parameters()
                            if not n.startswith('head.') and p.grad is not None), default=0.)
            return float(output.detach()), gradient
        taught, untaught = shared_task('reciprocal'), shared_task('untaught')
        assert taught[0] == untaught[0] and taught[1] > 0 and untaught[1] == 0
        # Future target never enters the issued world query; feedback separates it.
        probabilities, future_values = [], []
        for label in [0, 1]:
            model, state = fixture()
            request = dict(node=0, time=2., horizon=.2, query_id=0)
            model.issue(request, state, torch.Generator().manual_seed(81), 'learned', False)
            packet = state['issued'][0]
            probabilities.append((packet['probability'], packet['world_probability']))
            model.feedback(dict(time=3., query_id=0, target=label), state, False)
            future_values.append(state['bank'][('C', (0, 0))][-1]['value'].detach())
            assert state['counts']['world_queries'] == state['counts']['world_feedback'] == 1
        assert probabilities[0] == probabilities[1]
        outcome_difference = float((future_values[0]-future_values[1]).abs().max())
        assert outcome_difference > 0
        # Route draws do not alter the separately queried world belief.
        world_predictions = []
        for seed in [81, 927]:
            model, state = fixture()
            model.issue(dict(node=0, time=2., horizon=.2, query_id=0), state,
                        torch.Generator().manual_seed(seed), 'learned', False)
            world_predictions.append(state['issued'][0]['world_probability'])
        assert world_predictions[0] == world_predictions[1]
        model, state = fixture(); model.eval()
        torch.manual_seed(643); before = model.context(0, 2., state).detach()
        model.write_record(('C', (0, 0)), 'C', torch.ones(16)*999, 999., state)
        torch.manual_seed(643); after = model.context(0, 2., state).detach()
        assert torch.equal(before, after)
        base = cases[0]
        relabel = next(r for r in data['records'] if r['split'] == base['split']
                       and r['variant'] == 'relabel' and Path(r['path']).stem.replace('_relabel', '_base') == Path(base['path']).stem)
        model = WorldCreditSensor(confidence); model.load_state_dict(active)
        with torch.no_grad():
            model.bank_relation.weight.copy_(torch.tensor([[.2, .3, -.1, .4, .1]]))
            model.critic.bank_relation.weight.copy_(torch.tensor([[-.1, .4, .2, .1, .3]]))
        first = run(model, path(base), 'learned', seed=991, max_queries=16)
        second = run(model, path(relabel), 'learned', seed=991, max_queries=16)
        assert first['nll'] == second['nll'] and first['world_nll'] == second['world_nll']
        smoke = []
        for row in [train[0], cases[-1]]:
            model = WorldCreditSensor(confidence); model.load_state_dict(initial)
            result = run(model, path(row), 'learned', torch.optim.Adam(model.parameters(), lr=.001), seed=194, max_queries=16)
            assert result['world_queries'] == result['world_feedback'] == result['targets']
            assert result['wall_s'] < 90.
            smoke.append(dict(path=row['path'], **result))
        checks = dict(bce_identity_error=identity_error, risk_cycle_error=float(cycle),
                      blend_mse_error=expected_mse-formula, native_auxiliary_body_gradient=body_grad,
                      independent_world_read_gradient=read_grad, shared_future_task=taught,
                      untaught_future_task=untaught, label_independent_issued_forecast=True,
                      revealed_outcome_state_difference=outcome_difference,
                      chosen_route_independent_world_query=True, active_relabel_parity=True,
                      future_record_invisible=True, smoke=smoke)
    elif a.mode == 'acquisition':
        contract = verify_result(CONTRACT)
        assert contract['peak_rss_kb'] < 800000
        model = WorldCreditSensor(confidence); model.load_state_dict(initial)
        optimizer = torch.optim.Adam(model.parameters(), lr=.001)
        for i, row in enumerate(train):
            result = run(model, path(row), 'exact', optimizer, seed=194+i)
            rows.append(dict(path=row['path'], **result))
            print(json.dumps(dict(stage='acquisition', episode=i, nll=result['nll'], world_nll=result['world_nll'], wall_s=result['wall_s'])), flush=True)
        checkpoint = out.with_suffix('.pt')
        snapshot = dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                        confidence=confidence, torch_rng=torch.get_rng_state(), numpy_rng=np.random.get_state())
        snapshot['learner_fingerprint'] = fingerprint(dict(model=snapshot['model'], optimizer=snapshot['optimizer']))
        torch.save(snapshot, checkpoint)
        acquired_world_nll = float(np.mean([r['world_nll'] for r in rows[8:]]))
        checks = dict(finite_common_acquisition=True, last_eight_world_nll=acquired_world_nll,
                      acquisition_gate=acquired_world_nll < math.log(2),
                      checkpoint=str(checkpoint.relative_to(ROOT)), checkpoint_sha256=sha(checkpoint),
                      learner_fingerprint=snapshot['learner_fingerprint'])
    else:
        assert a.arm is not None
        acquisition = verify_result(ACQUISITION)
        assert acquisition['checks']['acquisition_gate'], 'Acquisition gate failed; no reuse admitted'
        checkpoint = ROOT/acquisition['checks']['checkpoint']
        assert sha(checkpoint) == acquisition['checks']['checkpoint_sha256']
        source = torch.load(checkpoint, map_location='cpu', weights_only=False)
        for i, row in enumerate(cases):
            torch.manual_seed(194)
            model = WorldCreditSensor(source['confidence']); model.load_state_dict(source['model'])
            optimizer = torch.optim.Adam(model.parameters(), lr=.001)
            optimizer.load_state_dict(copy.deepcopy(source['optimizer']))
            incoming = fingerprint(dict(model=model.state_dict(), optimizer=optimizer.state_dict()))
            assert incoming == source['learner_fingerprint']
            result = run(model, path(row), a.arm, optimizer, seed=777+i)
            rows.append(dict(split=row['split'], variant=row['variant'], depth=row['depth'], width=row['width'], path=row['path'], incoming_learner_fingerprint=incoming, **result))
            print(json.dumps(dict(stage='reuse', arm=a.arm, episode=i, nll=result['nll'], world_nll=result['world_nll'], wall_s=result['wall_s'])), flush=True)
        checks = dict(identical_common_source=True,
                      acquisition_result_sha256=sha(RESULTS/(ACQUISITION+'.json')),
                      source_checkpoint_sha256=sha(checkpoint))
    files = ['experiments/causal_induction/'+name for name in [
        'world_credit_pipeline.py', 'world_credit_sensor.py', 'world_credit_run.py',
        'WORLD_CREDIT_PROTOCOL.md', 'addressed_shared_bank.py', 'task_taught_shared_bank.py',
        'shared_bank_sensor_v2.py', 'race_credit_sensor.py', 'structural_sensor.py',
        'native_sensor.py', 'task_taught_run.py', 'credit_run.py', 'public_stream.py']]
    files += ['sleeping_machines/addressed_event_heads.py', 'sleeping_machines/paired_route_credit.py',
              'sleeping_machines/parallel_head_race_language.py', 'sleeping_machines/sparse_race_language.py']
    targets = sum(r['targets'] for r in rows)
    measured_fit_wall = sum(r['wall_s'] for r in rows)
    source_wall = ((acquisition['confidence_acquisition_wall_s']+sum(r['wall_s'] for r in acquisition['rows']))
                   if acquisition else 0.)
    result = dict(status='completed', battle='B5/R1', tag=a.tag, mode=a.mode, arm=a.arm,
                  checks=checks, rows=rows, targets=targets,
                  pooled_nll=sum(r['nll']*r['targets'] for r in rows)/targets if targets else None,
                  pooled_world_nll=sum(r['world_nll']*r['targets'] for r in rows)/targets if targets else None,
                  confidence_acquisition_targets=2048,
                  confidence_acquisition_wall_s=confidence_wall,
                  fitting_wall_s=measured_fit_wall, source_acquisition_wall_s=source_wall,
                  whole_learning_wall_s=source_wall+measured_fit_wall+(confidence_wall if a.mode == 'acquisition' else 0.),
                  whole_pipeline_wall_s=time.monotonic()-begin,
                  peak_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                  parent_sha256=sha(PARENT), data_records=train+cases,
                  source_sha256={name: sha(ROOT/name) for name in files},
                  scope='Exploratory common-acquisition native outcome-law credit on public nonlinear event systems. Source16graphs/2048unique targets and4096charged presentations. Reuse8independent source-checkpoint clones,1024causal targets per arm; numerical state fresh, actor/world online updates every8feedbacks. Identical shared-memory/outcome training and sampled alternative decoding; exact vs learned/prior differs source-choice teaching only. Learned/prior half blends intentionally change conditional teaching mean; no per-example unbiasedness, low-audit economy, calibrated response geometry, full optimizer meta-gradient, neural-depth replacement, publicTEST or efficiency claim.')
    assert result['peak_rss_kb'] < 1500000
    out.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(tag=a.tag, mode=a.mode, arm=a.arm, checks=checks if a.mode != 'contract' else {k:v for k,v in checks.items() if k != 'smoke'}, pooled_nll=result['pooled_nll'], peak_rss_kb=result['peak_rss_kb'])), flush=True)


if __name__ == '__main__':
    main()
