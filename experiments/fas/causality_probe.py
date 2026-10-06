"""Suffix-mutation causality probe for FAS learned detectors (FAS_V2_CONFIRMATORY_PROTOCOL.md, Stage 0; as the E171 audit).

For validation runs longer than --cut process events, every event after position --cut is replaced: random event types and
a permuted gap sequence. The probe scores the original and mutated runs through the model's evaluation path. Contract:
every prefix score for N <= cut, under every rule, is unchanged (max |difference| <= --tol). The prefix score at N uses
events 1..N only. As a sanity check, some prefix score for N > cut must change.

Models:
- native: rebuilt from a completed native.py result (selected weights, same evaluation path as native.scores:
  batched race, carried recruitment layer when --segment, expected reception when chosen);
- dense: rebuilt from a completed dense.py result with saved weights.
--untrained probes the architecture with random weights (args given on the command line) before any run exists.
Evaluation only; small.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'experiments')); sys.path.insert(0, str(ROOT / 'experiments/fas'))
import native as N  # noqa: E402
from baselines import PREFIXES  # noqa: E402

OUT = ROOT / 'experiments/results/diagnostics'


def mutate(run, cut, rng):
    ids, t = run
    ids2 = ids.copy(); t2 = t.copy()
    ids2[cut:] = rng.integers(1, N.V, len(ids) - cut)
    gaps = np.diff(t[cut - 1:]); rng.shuffle(gaps)
    t2[cut:] = t[cut - 1] + np.cumsum(gaps)
    return ids2, t2


def native_scorer(args, weights):
    from sleeping_machines.addressed_event_heads import AddressedEventHeads
    from sleeping_machines.batched_episodes import batched_logits
    from sleeping_machines.fast_native_core import fast_class
    model = fast_class(AddressedEventHeads)(sources=1, content_dim=N.V, classes=N.V + 2, payload=args['payload'],
                                            depth=args['depth'], heads=args['heads'], pool=args['pool'])
    if args.get('tie_pools'):
        from dvs_tied_pool_benchmark import tie_pools
        model = tie_pools(model)
    if weights:
        model.load_state_dict(torch.load(weights, weights_only=True))
    rkw = exp_step = None
    if args.get('segment'):
        rkw = dict(free_bias=args.get('free_bias', 0.), stale_bias=args.get('stale_bias', 0.), temperature=1., eager=True)
    if args.get('reception') == 'expected':
        from sleeping_machines.expected_reception import expected_layer
        exp_step, rkw = expected_layer, None
    return lambda runs: N.scores(model, runs, 64, batched_logits, rkw, exp_step)[0]


def dense_scorer(args, weights):
    import dense as Dn
    model = Dn.build(args['model'], args['d'], args['layers'])
    if weights:
        model.load_state_dict(torch.load(weights, weights_only=True))
    return lambda runs: Dn.scores(model, runs, 64)[0]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--result', default='', help='completed native.py or dense.py result JSON')
    p.add_argument('--untrained', choices=('native', 'dense'), default=None)
    p.add_argument('--model-args', default='{}', help='JSON args for --untrained (native: payload, depth, heads, pool, '
                                                       'segment, reception; dense: model, d, layers)')
    p.add_argument('--data', default='fas_v1_20261004'); p.add_argument('--runs', type=int, default=8)
    p.add_argument('--cut', type=int, default=200); p.add_argument('--max-events', type=int, default=600)
    p.add_argument('--tol', type=float, default=1e-9); p.add_argument('--tag', required=True)
    a = p.parse_args()
    out = OUT / f'{a.tag}.json'
    if out.exists():
        raise ValueError('Unique unused tag required')
    torch.set_num_threads(1); torch.manual_seed(0); rng = np.random.default_rng(0)
    if a.untrained:
        kind, args, weights = a.untrained, json.loads(a.model_args), None
    else:
        res = json.loads(Path(a.result).read_text()); args = res['args']
        kind = 'dense' if 'model' in args and args['model'] in ('lstm', 'transformer', 'lru', 's5t', 'mamba') else 'native'
        sw = res['selected_weights']
        if isinstance(sw, dict):
            weights = Path(a.result).resolve().parent / sw['path']
        else:
            weights = ROOT / sw
        a.data = args['data']
    score = (dense_scorer if kind == 'dense' else native_scorer)(args, weights)
    runs, _ = N.load(ROOT / 'experiments/data/fas' / a.data / 'val_clean.npz', a.max_events)
    runs = [r for r in runs if len(r[0]) > a.cut + 50][:a.runs]
    mutated = [mutate(r, a.cut, rng) for r in runs]
    s0, s1 = score(runs), score(mutated)
    pre = [j for j, n in enumerate(PREFIXES) if n <= a.cut]; post = [j for j, n in enumerate(PREFIXES) if n > a.cut]
    diff_pre = max(float(np.nanmax(np.abs(s0[r][:, pre] - s1[r][:, pre]))) for r in s0)
    diff_post = max(float(np.nanmax(np.abs(s0[r][:, post] - s1[r][:, post]))) if post else 0. for r in s0)
    result = dict(status='completed', kind=kind, model_args=args, untrained=bool(a.untrained), runs=len(runs), cut=a.cut,
                  prefixes_checked=[PREFIXES[j] for j in pre], max_abs_diff_prefix_le_cut=diff_pre,
                  max_abs_diff_prefix_gt_cut=diff_post, passed=bool(diff_pre <= a.tol and diff_post > a.tol),
                  source=a.result or 'untrained', data=a.data)
    OUT.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1) + '\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
