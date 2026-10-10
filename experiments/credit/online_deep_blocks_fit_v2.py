#!/usr/bin/env python3
"""B1 real-data replay: isolate roundoff amplification from fixed-input credit errors."""
import argparse
from copy import deepcopy
import hashlib
import itertools
import json
import math
import resource
import time

import numpy as np
import torch
import online_deep as base
from online_deep_blocks import BlockTraces

SOURCES = ['experiments/credit/online_deep_blocks_fit_v2.py', 'experiments/credit/online_deep_blocks_fit.py',
           'experiments/credit/online_deep_blocks.py', 'experiments/credit/online_deep.py',
           'experiments/tpp/race_tpp_v8.py']


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def gate(path, stage):
    record = json.loads(path.read_text())
    assert record['status'] == 'completed'
    for source, sha in record['source_sha256'].items():
        assert digest(base.ROOT/source) == sha, source
    if stage == 'real':
        row = next(row for row in record['measurements'] if row['batch'] == 16 and row['d'] == 32)
        assert row['whole_step_speedup'] >= 1.5
    else:
        assert record['stage'] == 'real' and record['parity_pass']
        assert record['training_speedup'] >= 1.5
    return dict(path=str(path.relative_to(base.ROOT)), sha256=digest(path))


def errors(left, right, opts):
    parameter = max(float((left[k]-right[k]).abs().max().detach()) for k in left)
    gradient = max(float((left[k].grad-right[k].grad).abs().max())
                   for k in left if left[k].grad is not None and right[k].grad is not None)
    state = max(float((opts[0].state[left[k]][f]-opts[1].state[right[k]][f]).abs().max())
                for k in left for f in opts[0].state[left[k]])
    return dict(parameters=parameter, gradients=gradient, adam=state)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tag', required=True)
    parser.add_argument('--stage', choices=['real', 'epoch'], required=True)
    parser.add_argument('--gate', required=True)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    torch.set_num_threads(1)
    out = base.ROOT/'experiments/results/credit'/f'{args.tag}.json'
    assert not out.exists()
    predecessor = gate(base.ROOT/args.gate, args.stage)
    start = time.monotonic()
    train = base.load_split('taxi', 'train'); dev = base.load_split('taxi', 'dev')
    K = int(max(max(s[1]) for s in train+dev))+1
    gaps = np.concatenate([np.diff(s[0]) for s in train]); pos = gaps[gaps > 0]
    scale = float(np.median(pos))
    log_q = np.log(np.quantile(pos/scale, np.linspace(.1, .9, 8)))
    recipe = argparse.Namespace(K=K, d=32, modes=16, n_exp=4, n_ln=8)
    init = base.init_params(K, 32, 16, 4, 8, log_q, torch.Generator().manual_seed(args.seed))
    params = [{k: v.detach().clone().requires_grad_() for k, v in init.items()} for _ in range(2)]
    opts = [torch.optim.Adam(p.values(), lr=.001) for p in params]
    lay = base.Layout(K, 32, 16)
    timings = dict(dense=0., blocks=0.); maxerr = dict(parameters=0., gradients=0., adam=0., likelihood_sum=0.)
    count = updates = nbatches = 0; rows = []; audit = None; max_mean_error = 0.
    batches = base.batches(train, 16, True, np.random.default_rng(args.seed))
    if args.stage == 'real': batches = itertools.islice(batches, 4)
    original = base.DeepTraces
    try:
        for b, (t, m, mask) in enumerate(batches):
            results = {}
            before = {k:v.detach().clone() for k,v in params[0].items()}
            before_opt = deepcopy(opts[0].state_dict())
            # Same batch and optimizer trajectory; alternate execution order.
            for i in ((0, 1) if b % 2 == 0 else (1, 0)):
                name = ('dense', 'blocks')[i]
                base.DeepTraces = (original, BlockTraces)[i]
                t0 = time.perf_counter()
                results[name] = base.run_batch(params[i], t.to(base.DT), m, mask, recipe, scale,
                                               'online_deep', opts[i], lay)
                timings[name] += time.perf_counter()-t0
            err = errors(*params, opts)
            err['likelihood_sum'] = max(abs(x-y) for x, y in zip(results['dense'][:3], results['blocks'][:3]))
            for k, value in err.items(): maxerr[k] = max(maxerr[k], value)
            assert all(math.isfinite(v) for v in maxerr.values()), maxerr
            mean_error = err['likelihood_sum']/max(results['dense'][2], 1)
            max_mean_error = max(max_mean_error, mean_error)
            if audit is None and max(err.values()) >= 1e-9:
                # Replay this same batch from exactly the reference's incoming
                # weights and Adam state; this separates contraction roundoff
                # from sensitivity to earlier optimizer perturbations.
                shadow = {k:v.clone().requires_grad_() for k,v in before.items()}
                shadow_opt = torch.optim.Adam(shadow.values(), lr=.001)
                shadow_opt.load_state_dict(before_opt)
                base.DeepTraces = BlockTraces
                shadow_results = base.run_batch(shadow, t.to(base.DT), m, mask, recipe, scale,
                                               'online_deep', shadow_opt, lay)
                controlled = errors(params[0], shadow, [opts[0], shadow_opt])
                normalized = max(float((params[0][k].grad-shadow[k].grad).abs().max()) /
                                 max(1., float(params[0][k].grad.abs().max()))
                                 for k in shadow if shadow[k].grad is not None)
                assert controlled['parameters'] < 1e-9 and controlled['adam'] < 1e-9
                assert normalized < 1e-9
                controlled_mean = max(abs(x-y) for x,y in zip(results['dense'][:2],shadow_results[:2]))/max(results['dense'][2],1)
                assert controlled_mean < 1e-9
                audit = dict(batch=b, independent_errors=err, same_incoming_state_errors=controlled,
                             controlled_gradient_scaled_error=normalized, controlled_mean_likelihood_error=controlled_mean,
                             cause='Same-input block replay passes; earlier floating-point perturbations are amplified by online updates')
                print(json.dumps(dict(roundoff_audit=audit)), flush=True)
            # A fitted trajectory is assessed on prediction quality. Its raw
            # gradient magnitudes are not probabilities or scored-target means.
            assert mean_error < 1e-6, 'Per-target training likelihood drift exceeded registered tolerance'
            count += results['dense'][2]; updates += m.shape[1]-1; nbatches += 1
            rows.append(dict(batch=b, shape=list(m.shape), errors=err))
            if b % 10 == 0: print(json.dumps(dict(batch=b, timings=timings, errors=maxerr)), flush=True)
    finally:
        base.DeepTraces = original
    validation = {}; eval_wall = {}
    if args.stage == 'epoch':
        for label, p in zip(('dense', 'blocks'), params):
            t0 = time.perf_counter(); tl, ml, ll = base.evaluate(p, dev, recipe, scale)
            eval_wall[label] = time.perf_counter()-t0
            validation[label] = dict(ll=ll, time_ll=tl, mark_ll=ml)
        assert abs(validation['dense']['ll']-validation['blocks']['ll']) < 1e-6
    floats = BlockTraces(1, 16, 32, K, lay, False).stored_floats()
    result = dict(status='completed', tag=args.tag, battle='B1/R1 deep credit', stage=args.stage,
                  seed=args.seed, predecessor=predecessor, parity_pass=True, maximum_errors=maxerr,
                  roundoff_audit=audit, maximum_per_target_training_likelihood_difference=max_mean_error,
                  trajectory_quality_tolerance_nats_per_target=1e-6,
                  training_wall_s=timings, training_speedup=timings['dense']/timings['blocks'],
                  evaluation_wall_s=eval_wall, validation=validation, train_batches=nbatches,
                  train_scored_events=count, optimizer_updates_per_arm=updates, trace_floats_per_stream=dict(
                      dense=4*16*lay.P, blocks=floats), scale=scale,
                  batch_checks=rows, wall_s=time.monotonic()-start,
                  data_sha256={split:digest(base.ROOT/f'data/easytpp/taxi/{split}.json')
                               for split in ('train', 'dev')},
                  max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                  source_sha256={s:digest(base.ROOT/s) for s in SOURCES},
                  scope='Same two-layer model, seed, shuffled TRAIN batches, per-event Adam and DEV scorer. '
                        'Independent trajectory differences are recorded; 1e-6 nats/target quality gate, '
                        'with same-incoming-state audit at first raw 1e-9 divergence. '
                        'No TEST. Complete online train steps timed; loading/evaluation/parity-check overhead separate. '
                        'This is execution equivalence, not a changed quality or hard-routing claim.')
    with out.open('x') as stream: stream.write(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
