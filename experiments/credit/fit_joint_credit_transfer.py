"""B1 fixed-context local credit acquisition; forward labels, fresh receivers.

This is an acquisition certificate, not new task learning: three observed
targets in a fixed native TRAIN prefix, frozen actor, fresh artificial terminal
cotangents. Exact VJP is final validation only. No DEV/TEST or model tuning.
"""
import argparse
import hashlib
import json
from pathlib import Path
import resource
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'experiments/tpp'))
from check_joint_forward_risk import SharedProducerActor, native


def evaluate_observed_credit(actor, estimator, times, marks, scope):
    actor.zero_grad()
    mask = torch.ones_like(marks, dtype=torch.bool)
    tl, ml, count, _ = actor.loglik(times, marks, mask)
    loss = -(tl + ml) / count
    loss.backward()
    c = actor.packet.output.grad.detach()
    with torch.no_grad(): predicted = estimator(c.flatten())[None]
    gx, gt = actor.packet.transpose_with_time(c[None], context='taxi', version=0)
    exact = torch.cat([gx.flatten(1), gt.flatten(1)], 1)
    zero_risk = float(exact.square().sum())
    error = float((predicted - exact).square().sum())
    x, elapsed, _, _, _ = actor.produce(times, marks)
    params = [actor.embed.weight, actor.gap.weight, actor.gap.bias, actor.tempo.weight, actor.tempo.bias]
    projected = torch.autograd.grad((x * (predicted[0, :x.numel()].reshape_as(x) - gx[0])).sum() +
                 (elapsed * (predicted[0, x.numel():].reshape_as(elapsed) - gt[0])).sum(), params, retain_graph=True)
    true_projected = torch.autograd.grad((x * gx[0]).sum() + (elapsed * gt[0]).sum(), params)
    producer_error = sum(float(g.square().sum()) for g in projected)
    producer_zero = sum(float(g.square().sum()) for g in true_projected)
    return dict(scope=scope, observed_loss=float(loss.detach()), joint_credit_error=error,
                zero_credit_error=zero_risk, residual_energy_reduction=1-error/zero_risk,
                producer_credit_error=producer_error, producer_zero_error=producer_zero,
                producer_energy_reduction=1-producer_error/producer_zero)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tag', required=True)
    parser.add_argument('--seed', type=int, required=True, choices=[0, 1, 2])
    args = parser.parse_args()
    gate_path = ROOT/'experiments/results/credit/aws_joint_credit_training_20261011_v1.json'
    gate = json.loads(gate_path.read_text())
    assert gate['status'] == 'completed'
    for source, digest in gate['source_sha256'].items():
        assert hashlib.sha256((ROOT/source).read_bytes()).hexdigest() == digest
    torch.set_num_threads(1); torch.set_default_dtype(torch.float64)
    torch.manual_seed(950 + args.seed)
    start = time.monotonic()
    train = native.load_split('taxi', 'train')
    seq = next(s for s in train if len(s[1]) >= 4)
    K = 1 + max(int(m.max()) for _, m in train)
    pos = np.diff(seq[0][:4]); pos = pos[pos > 0]
    scale = float(np.median(pos)); qs = np.log(np.quantile(pos, [.1, .5, .9])).tolist()
    t, m = torch.tensor(seq[0][:4])[None], torch.tensor(seq[1][:4])[None]
    mask = torch.ones_like(m, dtype=torch.bool)
    rows = []; bundles = {}
    for depth in [2, 4, 8]:
        actor = SharedProducerActor(K, 8, 4, depth, 2, 3, 2, 0., scale, qs).eval()
        tick = time.monotonic()
        tl, ml, count, _ = actor.loglik(t, m, mask)
        primal_s = time.monotonic() - tick
        output_dimension = actor.packet.output.numel()
        time_dimension = t.numel()
        input_dimension = output_dimension + time_dimension
        receivers = torch.randn(512, output_dimension) / np.sqrt(output_dimension)
        # A fresh full-support probe block for every receiver, not one shared subspace.
        probe = (torch.randint(2, (512, 4, input_dimension)).to(t.dtype) * 2 - 1)
        flat_probe = probe.flatten(0, 1)
        tick = time.monotonic()
        response = actor.packet.apply_with_time(
            flat_probe[:, :output_dimension].reshape(-1, *actor.packet.output.shape),
            flat_probe[:, output_dimension:].reshape(-1, *t.shape), context='taxi', version=0)
        target = (response.reshape(512, 4, output_dimension) * receivers[:, None, :]).sum(-1)
        label_s = time.monotonic() - tick
        estimator = torch.nn.Linear(output_dimension, input_dimension, bias=False)
        with torch.no_grad(): estimator.weight.zero_()
        optimizer = torch.optim.Adam(estimator.parameters(), lr=.01)
        normalizer = target.square().mean().clamp_min(1e-12)
        tick = time.monotonic()
        snapshots = []
        for step in range(400):
            optimizer.zero_grad()
            predicted = (estimator(receivers)[:, None, :] * probe).sum(-1)
            teaching_loss = (predicted-target).square().mean()/normalizer
            teaching_loss.backward(); optimizer.step()
            if step in [0, 49, 99, 199, 399]:
                snapshots.append(dict(step=step+1, normalized_training_loss=float(teaching_loss.detach())))
        fitting_s = time.monotonic() - tick
        assert all(torch.isfinite(p).all() for p in estimator.parameters())
        # Freeze fit, then generate fresh receiver and probe observations for calibration.
        fresh = torch.randn(64, output_dimension) / np.sqrt(output_dimension)
        held_probes = torch.randint(2, (64, 4, input_dimension)).to(t.dtype)*2-1
        tick = time.monotonic()
        flat = held_probes.flatten(0, 1)
        held_responses = actor.packet.apply_with_time(
            flat[:, :output_dimension].reshape(-1, *actor.packet.output.shape),
            flat[:, output_dimension:].reshape(-1, *t.shape), context='taxi', version=0)
        held_target = (held_responses.reshape(64, 4, output_dimension)*fresh[:, None, :]).sum(-1)
        with torch.no_grad(): prediction = estimator(fresh)
        held_prediction = (prediction[:, None, :]*held_probes).sum(-1)
        forward_validation_s = time.monotonic() - tick
        # Separate reverse reference scores the whole field after fitting, never its labels.
        tick = time.monotonic()
        gx, gt = actor.packet.transpose_with_time(fresh.reshape(-1, *actor.packet.output.shape),
                                                 context='taxi', version=0)
        exact = torch.cat([gx.flatten(1), gt.flatten(1)], 1)
        reverse_validation_s = time.monotonic() - tick
        zero_error = exact.square().sum(-1)
        learned_error = (prediction-exact).square().sum(-1)
        difference = zero_error-learned_error
        mean = float(difference.mean()); se = float(difference.std(unbiased=True))/np.sqrt(len(fresh))
        tick = time.monotonic()
        trained_actor_state={k:v.detach().clone() for k,v in actor.state_dict().items()}
        original = evaluate_observed_credit(actor, estimator, t, m, 'same actor/context, actual observed loss')
        changed = []
        for index, other in enumerate([z for z in train if len(z[1]) >= 4][1:3]):
            ot, om = torch.tensor(other[0][:4])[None], torch.tensor(other[1][:4])[None]
            changed.append(evaluate_observed_credit(actor, estimator, ot, om, f'new TRAIN context {index+1}, same weights'))
        # Recompute the actual loss for one conventional update; this is a drift diagnostic.
        actor.zero_grad(); tl, ml, n, _ = actor.loglik(t, m, mask)
        actor_loss = -(tl+ml)/n; actor_loss.backward()
        optimizer_actor = torch.optim.Adam(actor.parameters(), lr=.001); optimizer_actor.step()
        drift = evaluate_observed_credit(actor, estimator, t, m, 'same context, after one Adam model update, expired credit map tested diagnostically')
        transfer_s = time.monotonic() - tick
        bundles[str(depth)] = dict(estimator=estimator.state_dict(), actor=trained_actor_state, actor_args=(K,8,4,depth,2,3,2,0.,scale,qs), trained_context=dict(times=t,marks=m),
                                  trained_actor_version=0, acquisition_config=dict(scale=scale,d=8,modes=4,depth=depth),
                                  scope='Frozen-context linear map; requires revalidation/audits after context or weights change')
        rows.append(dict(depth=depth, unique_actor_targets=int(count),
                         frozen_prefix_events=4, estimator_parameters=sum(p.numel() for p in estimator.parameters()),
                         training_receivers=512, probes_per_receiver=4, optimizer_updates=400,
                         heldout_receivers=64, heldout_probe_count=256,
                         zero_credit_error=float(zero_error.mean()), learned_credit_error=float(learned_error.mean()),
                         residual_energy_reduction=1-float(learned_error.mean()/zero_error.mean()),
                         paired_receiver_gain=mean, paired_receiver_ci95=[mean-1.96*se,mean+1.96*se],
                         zero_probe_mse=float(held_target.square().mean()),
                         learned_probe_mse=float((held_prediction-held_target).square().mean()),
                         primal_s=primal_s, forward_label_s=label_s, estimator_fitting_s=fitting_s,
                         forward_validation_s=forward_validation_s, reverse_validation_s=reverse_validation_s,
                         teaching_snapshots=snapshots, same_context_observed_credit=original, new_context_observed_credit=changed, changed_weights_observed_credit=drift, transfer_diagnostic_s=transfer_s, packet_cached_floats=actor.packet.cached_floats()))
    sources = ['experiments/credit/fit_joint_credit_transfer.py',
               'experiments/credit/check_joint_forward_risk.py',
               'experiments/credit/joint_forward_residual_risk.py',
               'experiments/credit/check_deep_temporal_packet.py',
               'experiments/credit/native_fused_temporal_packet.py',
               'experiments/credit/native_deep_temporal_packet.py',
               'experiments/credit/native_temporal_teaching_packet.py',
               'experiments/credit/native_training_temporal_packet.py',
               'experiments/credit/native_training_linearization.py',
               'experiments/credit/native_cached_linearization.py',
               'experiments/credit/native_cached_jvp.py',
               'experiments/credit/predictable_audit_allocation_v2.py',
               'experiments/tpp/race_tpp_v19.py']
    weight_path=ROOT/'experiments/results/credit'/f'{args.tag}.pt'
    assert not weight_path.exists();torch.save(bundles, weight_path)
    result = dict(status='completed', tag=args.tag, seed=args.seed, battle='B1 local joint-credit acquisition',
                  config=dict(depths=[2,4,8], receivers=512, queries=4, steps=400, lr=.01,
                              heldout_receivers=64, initialization='zero linear field'), rows=rows,
                  final_weights=str(weight_path.relative_to(ROOT)), checkpoint_sha256=hashlib.sha256(weight_path.read_bytes()).hexdigest(), gate=str(gate_path.relative_to(ROOT)), input_sha256={str(gate_path.relative_to(ROOT)):
                      hashlib.sha256(gate_path.read_bytes()).hexdigest()},
                  source_sha256={s: hashlib.sha256((ROOT/s).read_bytes()).hexdigest() for s in sources},
                  data_sha256=hashlib.sha256((ROOT/'data/easytpp/taxi/train.json').read_bytes()).hexdigest(),
                  wall_s=time.monotonic()-start, max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                  scope='Local map acquired on frozen TRAIN-prefix/artificial cotangents; separate real observed-loss credit diagnostics on same/new contexts and one Adam-changed actor, no estimator refitting; saved credit fields are expired approximations outside their registered context/version. Acquisition uses 3unique task targets per depth, not512 new data examples. Forward-only probe labels fit a1152parameter linear local credit map. Fresh receiver/probe calibration, final reverse reference charged separately. Cross-context/version drift measured, not assumed. No future task learning, hard-route, full-learning efficiency or Transformer claim; this is a local acquisition certificate.')
    with (ROOT/'experiments/results/credit'/f'{args.tag}.json').open('x') as output:
        json.dump(result, output, indent=2)
    print(json.dumps(result))


if __name__ == '__main__':
    main()
