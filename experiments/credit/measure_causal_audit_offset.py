"""B1/R1 locked TRAIN calibration and causal sparse audits on fresh native episodes."""
import argparse
import hashlib
import json
import resource
import time

import numpy as np
import torch
from torch.utils.flop_counter import FlopCounterMode

from task_valued_route_credit import ROOT, PREFIX, SOURCE, load, Correction, native
from shared_prefix_receiver_credit import SharedPrefixCredit
from cached_learning_effect import values
from causal_audit_offset import CausalOffset
from policy_difference_audit import issue
from paired_native_replay import replay_pairs

TAG = 'curie_causal_offset_20261011T0630Z'
SOURCES = ['experiments/credit/measure_causal_audit_offset.py',
           'experiments/credit/causal_audit_offset.py',
           'experiments/credit/policy_difference_audit.py',
           'experiments/credit/paired_native_replay.py',
           'experiments/credit/optimizer_aware_route_credit.py',
           'experiments/credit/task_valued_route_credit.py',
           'experiments/credit/cached_learning_effect.py',
           'experiments/credit/fit_optimizer_aware_credit.py',
           'experiments/credit/shared_prefix_receiver_credit.py', *native.SOURCES]


def train_offset(data, head):
    n, r, _ = data['base'].shape
    with torch.no_grad():
        pred = data['base']+values(head, data['z'], native.base.actions()[None, None].expand(n, r, -1, -1))
    raw = pred.flatten(1)
    old, new = data['base'].flatten(1).argmin(1), raw.argmin(1)
    mask = old != new
    residual = data['aware'].flatten(1)-raw
    means = (residual.gather(1, old[:, None])+residual.gather(1, new[:, None])).squeeze(1)/2
    return float(means[mask].mean()) if bool(mask.any()) else 0., int(mask.sum())


def prefix_predictions(actor, critic, head, t, m):
    with torch.no_grad():
        x = native.prefix(actor, t, m, 3, list(range(8)))
        context = critic.encode_context(x[:, 0, :, :30], 0, 0)
        r = critic.receiver(x[:, :, :, 30:].reshape(8, 4, 8).tanh())[0][:, -1].reshape(1, 8, 7)
        z = torch.cat((context.value[:, None].expand(-1, 8, -1), r), -1)
        actions = native.base.actions()[None, None].expand(1, 8, -1, -1)
        base = critic.values(context, x[:, :, :, 30:], actions, 0, 0)[0]
        pred = base+values(head, z, actions)[0]
    return base.flatten(), pred.flatten()


def measure(smoke, stream_seed=None, audit_seed=492):
    oldresult, saved = load(SOURCE)
    selected, fitted = load(f'experiments/results/credit/{PREFIX}_select.json')
    _, prior = load(oldresult['prior_result'])
    actor = native.actor(4, 170); actor.load_state_dict(prior['actor'])
    critic = SharedPrefixCredit(actor.M, native.prior.bound(actor)).double()
    critic.load_state_dict(prior['critic']); critic.requires_grad_(False)
    head = Correction(fitted['preprocessing']['task']['scale']).double()
    head.load_state_dict(fitted['heads']['task'])
    with FlopCounterMode(display=False) as preparation:
        initial, train_changed = train_offset(saved['train_tensors'], head)
    online = CausalOffset(initial, rate=.125)
    count, futures = (2, 1) if smoke else (32, 8)
    stream_seed = (772 if smoke else 472) if stream_seed is None else stream_seed
    methods = ['zero_predictor', 'raw_task', 'train_centered', 'causal_centered']
    records = []
    policy_flops = teacher_flops = 0
    policy_wall = teacher_wall = 0.
    sampled_rng = np.random.default_rng(audit_seed)
    for i in range(count):
        # New namespace; only the prefix exists before policies and receipts are issued.
        t, m = native.base.batch(4100000+stream_seed*1000+i, size=1, pairs=2)
        tick = time.monotonic()
        with FlopCounterMode(display=False) as policywork:
            base, pred = prefix_predictions(actor, critic, head, t[:, :4], m[:, :4])
        policy_wall += time.monotonic()-tick; policy_flops += policywork.get_total_flops()
        old, new = int(base.argmin()), int(pred.argmin())
        before = online.value
        if old == new:
            records.append(dict(prefix=i, old=old, new=new, offset_before=before, offset_after=before,
                                variance={name: 0. for name in methods}, realized_error={name: 0. for name in methods},
                                gain=0., observed=[], worlds=0))
            continue
        raw = {old: float(pred[old]), new: float(pred[new])}
        prior_policy, task_policy = {old: 1.}, {new: 1.}
        target_version = (selected['checkpoint_sha256'], i, 'native-readout-adam-.05', 4)
        receipts = [online.issue(prior_policy, task_policy, raw, target_version, (i, j)) for j in range(futures)]
        packets = {
            'zero_predictor': issue(prior_policy, task_policy, {old: 0., new: 0.}, target_version),
            'raw_task': issue(prior_policy, task_policy, raw, target_version),
            'train_centered': issue(prior_policy, task_policy, {a: p+initial for a, p in raw.items()}, target_version),
            'causal_centered': receipts[0].packet,
        }
        rng = np.random.default_rng(43000000+stream_seed*1000+i)
        actions = native.base.actions()[[old%9, new%9, 0]]
        observed, actual_samples = [], []
        tick = time.monotonic()
        with FlopCounterMode(display=False) as teacher:
            for j in range(futures):
                ft, fm = native.prior.continuation(t, m, 2, rng)
                scores, _ = replay_pairs(actor, ft, fm, [old//9, new//9, 0], actions, .05)
                costs = scores.mean(1); effect = costs[:2]-costs[2]
                observed.append({old: float(effect[0]), new: float(effect[1])})
                chosen = old if sampled_rng.integers(2) == 0 else new
                actual_samples.append((chosen, observed[-1][chosen]))
        teacher_wall += time.monotonic()-tick; teacher_flops += teacher.get_total_flops()
        gain = np.mean([o[old]-o[new] for o in observed])
        var, realized, mean_errors = {}, {}, {}
        for name, packet in packets.items():
            estimates = [packet.corrected_gain(a, o[a], packet.version) for o in observed for a in [old, new]]
            mean_errors[name] = abs(float(np.mean(estimates))-gain)
            var[name] = float(np.mean((np.asarray(estimates)-gain)**2))
            sampled = [packet.corrected_gain(a, value, packet.version) for a, value in actual_samples]
            realized[name] = float(np.mean(sampled)-gain)
        assert max(mean_errors.values()) < 1e-11
        # Only sampled feedback teaches calibration, after all receipts/forecasts were issued.
        for receipt, (chosen, value) in zip(receipts, actual_samples):
            online.consume(receipt, chosen, value, target_version)
        assert not online.pending
        records.append(dict(prefix=i, old=old, new=new, offset_before=before, offset_after=online.value,
                            variance=var, realized_error=realized, gain=float(gain),
                            observed=observed, sampled=actual_samples, mean_errors=mean_errors,
                            worlds=3*futures, operational_sampled_worlds=2*futures))
    bootstrap = np.random.default_rng(493).integers(0, count, (4000, count))
    zeros = np.array([r['variance']['zero_predictor'] for r in records])
    rows = []
    for name in methods:
        var = np.array([r['variance'][name] for r in records]); saving = zeros-var
        errors = np.array([r['realized_error'][name] for r in records])
        rows.append(dict(arm=name, mean_sampling_variance=float(var.mean()),
                         variance_ratio_vs_zero=float(var.mean()/zeros.mean()) if zeros.mean() else None,
                         variance_saving_vs_zero=float(saving.mean()),
                         variance_saving_bootstrap95=np.quantile(saving[bootstrap].mean(1), [.025, .975]).tolist(),
                         realized_mean_error=float(errors.mean()), realized_squared_error=float(np.mean(errors**2))))
    metrics = dict(rows=rows, prefixes=count, continuations=futures, stream_seed=stream_seed, audit_seed=audit_seed,
                changed_prefixes=sum(r['old'] != r['new'] for r in records),
                train_changed_prefixes=train_changed, initial_train_offset=initial,
                final_offset=online.value, calibration_feedback_updates=online.version,
                max_mean_reconstruction_error=max((max(r.get('mean_errors', {}).values(), default=0.) for r in records), default=0.),
                policy_supported_flops=policy_flops, policy_wall_s=policy_wall,
                teacher_supported_flops=teacher_flops, teacher_wall_s=teacher_wall,
                train_calibration_preparation_supported_flops=preparation.get_total_flops(),
                diagnostic_worlds=sum(r['worlds'] for r in records),
                operational_sampled_worlds=sum(r.get('operational_sampled_worlds', 0) for r in records),
                prior_checkpoint_sha256=oldresult['prior_checkpoint_sha256'], selection_checkpoint_sha256=selected['checkpoint_sha256'])
    payload = dict(records=records, offset_state=dict(value=online.value, version=online.version, rate=online.rate),
                   initial_train_offset=initial)
    return metrics, payload


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--tag', required=True)
    ap.add_argument('--mode', choices=['smoke', 'measure'], required=True)
    ap.add_argument('--stream-seed', type=int)
    ap.add_argument('--audit-seed', type=int, default=492)
    ap.add_argument('--smoke-result', default=f'experiments/results/credit/{TAG}_smoke.json')
    args = ap.parse_args()
    torch.set_default_dtype(torch.float64); torch.set_num_threads(1)
    out = ROOT/'experiments/results/credit'/f'{args.tag}.json'; assert not out.exists(); start = time.monotonic()
    contract = json.loads((ROOT/f'experiments/results/credit/{TAG}_contract.json').read_text())
    assert contract['status'] == 'completed'
    for path, digest in contract['source_sha256'].items():
        assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest() == digest, path
    if args.mode == 'measure':
        smoke = json.loads((ROOT/args.smoke_result).read_text())
        assert smoke['status'] == 'completed'
        for path, digest in smoke['source_sha256'].items():
            assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest() == digest, path
    metrics, payload = measure(args.mode == 'smoke', args.stream_seed, args.audit_seed)
    checkpoint = out.with_suffix('.pt'); torch.save(payload, checkpoint)
    result = dict(status='completed', battle='B1/R1', tag=args.tag, mode=args.mode, metrics=metrics,
                  whole_wall_s=time.monotonic()-start, peak_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                  checkpoint=str(checkpoint.relative_to(ROOT)), checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                  source_sha256={p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in SOURCES},
                  scope='One native actor seed170, synthetic independent episode prefixes with four causal readout Adam updates each. Baseline and proposal fixed; common offset cannot change route choices. TRAIN-only initialization; calibration learns only sampled earlier feedback across episodes. Includes raw future-continuation and route-sampling variation on eight generated continuations; finite prefix bootstrap, not a population guarantee. Three worlds paid per changed continuation for diagnostics; operational two-world cost is a count, not a measured total-fit speed. Dense eight-receiver discovery, prior teacher/baseline/critic fits and all current work remain charged. No TEST, full actor/critic co-adaptation, causal market sequence, deployment or depth scaling claim.')
    out.write_text(json.dumps(result, indent=2)+'\n'); print(json.dumps(metrics, indent=2))


if __name__ == '__main__': main()
