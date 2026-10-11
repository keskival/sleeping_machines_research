"""B1/R1 native saved-route witness for sparse issued policy comparisons."""
import argparse
import hashlib
import json
import resource
import time

import numpy as np
import torch
from torch.utils.flop_counter import FlopCounterMode

from task_valued_route_credit import ROOT, PREFIX, load, Correction, native
from cached_learning_effect import values
from policy_difference_audit import issue


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', required=True)
    args = ap.parse_args()
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)
    out = ROOT/'experiments/results/credit'/f'{args.tag}.json'
    assert not out.exists()
    begin = time.monotonic()
    selected, fit = load(f'experiments/results/credit/{PREFIX}_select.json')
    result, fresh = load(f'experiments/results/credit/{PREFIX}_fresh.json')
    data = fresh['packets']
    head = Correction(fit['preprocessing']['task']['scale']).double()
    head.load_state_dict(fit['heads']['task'])
    with torch.no_grad(), FlopCounterMode(display=False) as counter:
        pred = data['base']+values(head, data['z'], native.base.actions()[None, None].expand(32, 8, -1, -1))
    predictions = pred.flatten(1).numpy()
    truth = data['aware'].flatten(1).numpy()
    rows = {r['policy']: r for r in result['metrics']['rows']}
    indices = {name: np.array([r*9+a for r, a in row['choices']]) for name, row in rows.items()}
    paired = truth[np.arange(32), indices['mse_full_diagnostic']]-truth[np.arange(32), indices['task_full_diagnostic']]
    boot = np.random.default_rng(294).integers(0, 32, (4000, 32))
    mean_error = 0.
    corrected_variances, zero_variances = [], []
    changed = 0
    for i, (old, new) in enumerate(zip(indices['baseline'], indices['task_full_diagnostic'])):
        version = (selected['checkpoint_sha256'], result['checkpoint_sha256'], i, 'readout-adam-.05', 4)
        packet = issue({int(old): 1.}, {int(new): 1.},
                       {int(a): float(predictions[i, a]) for a in {old, new}}, version)
        if not packet.entries:
            corrected_variances.append(0.)
            zero_variances.append(0.)
            continue
        changed += 1
        observed = {int(a): float(truth[i, a]) for a in {old, new}}
        target = observed[int(old)]-observed[int(new)]
        mean = sum(p*packet.corrected_gain(a, observed[a], version) for a, p in packet.proposal())
        mean_error = max(mean_error, abs(mean-target))
        corrected_variances.append(sum(p*(packet.corrected_gain(a, observed[a], version)-target)**2 for a, p in packet.proposal()))
        no_prediction = issue({int(old): 1.}, {int(new): 1.}, {int(a): 0. for a in {old, new}}, version)
        zero_variances.append(sum(p*(no_prediction.corrected_gain(a, observed[a], version)-target)**2 for a, p in no_prediction.proposal()))
    assert mean_error < 1e-12
    sources = ['experiments/credit/check_native_policy_audit.py', 'experiments/credit/policy_difference_audit.py',
               'experiments/credit/cached_learning_effect.py', 'experiments/credit/task_valued_route_credit.py']
    summary = dict(status='completed', tag=args.tag, battle='B1/R1',
                   metrics=dict(native_exhaustive_mean_error=mean_error, prefixes=32,
                                changed_prefixes=changed, max_audit_support=2,
                                task_over_mse_gain=float(paired.mean()),
                                task_over_mse_bootstrap95=np.quantile(paired[boot].mean(1), [.025, .975]).tolist(),
                                mean_residual_action_sampling_variance=float(np.mean(corrected_variances)),
                                mean_zero_predictor_action_sampling_variance=float(np.mean(zero_variances)),
                                prediction_supported_flops=counter.get_total_flops()),
                   whole_wall_s=time.monotonic()-begin, peak_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                   source_sha256={p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in sources},
                   checkpoint_sha256=[selected['checkpoint_sha256'], result['checkpoint_sha256']],
                   scope='Already-paid fresh native synthetic response tables now examined for development. Exhaustive route-sampling witness conditional on empirical eight-continuation means, not future-innovation variance or live sampled-audit performance. Policy/prediction discovery, teacher and prior fitting remain charged. Diagnostic task-versus-MSE comparison, not the locked primary policy, replicated utility or whole-fit efficiency.')
    out.write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__': main()
