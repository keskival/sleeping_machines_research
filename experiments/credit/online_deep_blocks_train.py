#!/usr/bin/env python3
"""B1 full-fit replay of existing online3 learning with exact structural blocks."""
import argparse
import json
import math
import socket
import resource
import time

import numpy as np
import torch
import online_deep as base
from online_deep_blocks import BlockTraces
from online_deep_blocks_fit import digest

SOURCES = ['experiments/credit/online_deep_blocks_train.py',
           'experiments/credit/online_deep_blocks_fit_v2.py',
           'experiments/credit/online_deep_blocks_fit.py', 'experiments/credit/online_deep_blocks.py',
           'experiments/credit/online_deep.py', 'experiments/tpp/race_tpp_v8.py']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tag', required=True)
    parser.add_argument('--gate', required=True)
    parser.add_argument('--reference', required=True)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    torch.set_num_threads(1)
    folder = base.ROOT/'experiments/results/credit'
    out = folder/f'{args.tag}.json'
    weights = folder/f'{args.tag}.final.pt'
    assert not out.exists() and not weights.exists()
    prerequisite = base.ROOT/args.gate
    record = json.loads(prerequisite.read_text())
    assert record['status'] == 'completed' and record['stage'] == 'roundoff_diagnosis' and record['identical_state_audit_pass']
    assert record['prefix_training_speedup'] >= 1.5
    assert record['quality_rule'] == dict(mean_final_dev_drop_at_most=.002, every_seed_drop_at_most=.005, seeds=[0,1,2], epochs=20)
    for source, sha in record['source_sha256'].items(): assert digest(base.ROOT/source) == sha, source
    reference_path = base.ROOT/args.reference
    reference = json.loads(reference_path.read_text())
    assert reference['status'] == 'completed' and reference['args']['seed'] == args.seed
    assert reference['args']['arm'] == 'online_deep' and reference['args']['epochs'] == 20
    for source, sha in reference['source_sha256'].items(): assert digest(base.ROOT/source) == sha, source
    start = time.monotonic()
    train = base.load_split('taxi', 'train'); dev = base.load_split('taxi', 'dev')
    data_sha = {s:digest(base.ROOT/f'data/easytpp/taxi/{s}.json') for s in ('train', 'dev')}
    assert data_sha == record['data_sha256']
    a = argparse.Namespace(K=int(max(max(s[1]) for s in train+dev))+1,
                           d=32, modes=16, n_exp=4, n_ln=8)
    recipe = reference['args']
    assert all(recipe[k] == getattr(a, k) for k in ('K', 'd', 'modes', 'n_exp', 'n_ln'))
    assert recipe['batch'] == 16 and recipe['lr'] == .001 and recipe['dataset'] == 'taxi'
    gaps = np.concatenate([np.diff(s[0]) for s in train]); pos = gaps[gaps > 0]
    scale = float(np.median(pos)); assert scale == reference['scale']
    log_q = np.log(np.quantile(pos/scale, np.linspace(.1, .9, 8)))
    p = base.init_params(a.K, a.d, a.modes, 4, 8, log_q, torch.Generator().manual_seed(args.seed))
    lay = base.Layout(a.K, a.d, a.modes); opt = torch.optim.Adam(p.values(), lr=.001)
    rng = np.random.default_rng(args.seed)
    preview = base.ROOT/'.git/aws-deep-blocks-preview'/args.tag
    preview.mkdir(parents=True, exist_ok=False)
    hist = []; updates = 0; targets = 0; train_wall = eval_wall = 0.
    original = base.DeepTraces
    try:
        base.DeepTraces = BlockTraces
        for ep in range(20):
            epoch_start = time.monotonic(); fit_start = time.monotonic()
            for t, m, mask in base.batches(train, 16, True, rng):
                _, _, cnt, _ = base.run_batch(p, t.to(base.DT), m, mask, a, scale,
                                             'online_deep', opt, lay)
                updates += m.shape[1]-1; targets += cnt
            ts = time.monotonic()-fit_start; train_wall += ts
            eval_start = time.monotonic(); tl, ml, ll = base.evaluate(p, dev, a, scale)
            es = time.monotonic()-eval_start; eval_wall += es
            delta = ll-reference['history'][ep]['dev_ll']
            row = dict(epoch=ep, dev_ll=ll, dev_time_ll=tl, dev_mark_ll=ml, updates=updates,
                       train_s=ts, evaluation_s=es, epoch_s=time.monotonic()-epoch_start,
                       reference_dev_ll=reference['history'][ep]['dev_ll'], delta_from_reference=delta)
            hist.append(row)
            checkpoint = dict(epoch=ep, params={k:v.detach() for k, v in p.items()}, optimizer=opt.state_dict(),
                              numpy_rng=rng.bit_generator.state, torch_rng=torch.get_rng_state(),
                              history=hist, args=vars(a), data_sha256=data_sha,
                              source_sha256={s:digest(base.ROOT/s) for s in SOURCES})
            tmp = preview/'resume.tmp.pt'; torch.save(checkpoint, tmp); tmp.replace(preview/'resume.pt')
            (preview/'progress.json').write_text(json.dumps(row, indent=2)+'\n')
            print(json.dumps(row), flush=True)
            # Numerical trajectories are compared as quality, not bitwise identity.
            assert math.isfinite(ll), 'Non-finite DEV likelihood; checkpoint preserved'
            assert updates == reference['history'][ep]['updates']
    finally:
        base.DeepTraces = original
    torch.save(checkpoint, weights)
    result = dict(status='completed', tag=args.tag, battle='B1/R1 exact online credit execution',
                  seed=args.seed, stage='full_fit', epochs=20, history=hist,
                  hardware=dict(host=socket.gethostname(), threads=torch.get_num_threads(), dtype='float64'),
                  best_dev_ll=max(h['dev_ll'] for h in hist), final_dev_ll=hist[-1]['dev_ll'],
                  maximum_epoch_dev_difference=max(abs(h['delta_from_reference']) for h in hist),
                  quality_rule=record['quality_rule'],
                  source_sha256={s:digest(base.ROOT/s) for s in SOURCES}, data_sha256=data_sha,
                  predecessor=dict(path=args.gate, sha256=digest(prerequisite)),
                  reference=dict(path=args.reference, sha256=digest(reference_path)),
                  train_scored_target_presentations=targets, optimizer_updates=updates,
                  training_wall_s=train_wall, evaluation_wall_s=eval_wall, wall_s=time.monotonic()-start,
                  max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                  trace_floats_per_stream=BlockTraces(1, 16, 32, a.K, lay, False).stored_floats(),
                  final_weights=str(weights.relative_to(base.ROOT)),
                  final_weights_sha256=digest(weights),
                  scope='20-pass fixed-recipe replay of the saved two-layer Taxi DEV fit; exact model and update schedule, '
                        'structural block execution. No TEST. Whole training/evaluation wall measured; '
                        'saved reference wall is historical, not concurrent paired timing. Floating-point trajectories may differ; '
                        'three-seed final-quality preservation is evaluated by the registered separate decision.')
    with out.open('x') as stream: stream.write(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
