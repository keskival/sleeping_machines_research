"""Streaming rescoring of a trained native language model (evaluation only; no training).

The E64 window protocol resets the native state every window (T = S, stride S/2, second half scored), so each scored
target sees 128-255 characters of context and every target costs about two evaluated positions.  The saved LSTMs
instead carry state across the whole test interval.  This script carries the native state too: the test interval
text8[95M:95M+test] is split into L contiguous lanes, and each lane is streamed once through the exact winner-only
SparseStepper (sleeping_machines/sparse_inference.py, replay-equal to the training evaluator), scoring every target
once.  Stamps are lane-local character positions, as in training windows.  It reports the streamed bpc overall and by
context length (position within the lane), and optionally the window protocol scored by the same stepper on the same
interval, so the comparison is evaluator-matched.  The weights and training are unchanged; the model never saw
contexts longer than its training segment, so long-context behaviour is extrapolation and is measured, not assumed.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import platform
import sys
import time

import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'experiments'))
from language_batched_benchmark import EYE, load_text  # noqa: E402
from sleeping_machines.addressed_event_heads import AddressedEventHeads  # noqa: E402
from sleeping_machines.fast_native_core import fast_class  # noqa: E402
from sleeping_machines.sparse_inference import SparseStepper  # noqa: E402

BUCKETS = [0, 128, 256, 1024, 4096, 16384, math.inf]
OUT = ROOT / 'experiments/results/language_stream'


def build(args, weights):
    model = fast_class(AddressedEventHeads)(sources=1, content_dim=27, classes=27, payload=args['payload'],
                                            depth=args['depth'], heads=args['heads'], pool=args['pool'])
    if args.get('tie_pools'):
        from dvs_tied_pool_benchmark import tie_pools
        model = tie_pools(model)
    if args.get('skip_init_from'):
        from dvs_batched_large_benchmark import skip_init
        model = skip_init(model, args['skip_init_from'], args['skip_gate_bias'])
    model.load_state_dict(torch.load(weights, weights_only=True))
    return model.eval()


@torch.no_grad()
def stream(model, text, lanes, seed, deterministic=False, limit=0):
    """Each lane streams a contiguous block once; every target scored once.  Returns bits/targets per bucket."""
    n = len(text) - 1
    m = math.ceil(n / lanes)
    starts = [i * m for i in range(lanes) if i * m < n]
    lengths = torch.tensor([min(m, n - s) for s in starts])
    if limit:
        lengths = lengths.clamp(max=limit)
    L = len(starts)
    stepper = SparseStepper(model, L, seed, deterministic=deterministic)
    bits = np.zeros(len(BUCKETS) - 1); count = np.zeros(len(BUCKETS) - 1, dtype=np.int64)
    idx = np.array(starts)
    for k in range(int(lengths.max())):
        active = lengths > k
        pos = np.minimum(idx + k, n - 1)
        marks = torch.from_numpy(EYE[text[pos]])
        logits = stepper.step(torch.full((L,), float(k), dtype=torch.float64), marks, active)
        ce = F.cross_entropy(logits, torch.from_numpy(text[pos + 1].astype(np.int64)), reduction='none') / math.log(2)
        b = next(j for j in range(len(BUCKETS) - 1) if BUCKETS[j] <= k < BUCKETS[j + 1])
        bits[b] += float(ce[active].sum()); count[b] += int(active.sum())
    return bits, count, L


@torch.no_grad()
def windows(model, text, S, seed, lanes, limit=0):
    """E64 window protocol (T = S, stride S/2, second half after the first window) with the same stepper."""
    from sleeping_machines.sparse_inference import sparse_logits
    half = S // 2; starts = list(range(0, len(text) - S - 1, half))
    if limit:
        starts = starts[:limit]
    bits = 0.; n = 0
    for b in range(0, len(starts), lanes):
        chunk = starts[b:b + lanes]
        rows = [dict(events=[(float(t), EYE[c]) for t, c in enumerate(text[s:s + S])]) for s in chunk]
        z = sparse_logits(model, rows, seed, all_logits=True)
        y = torch.tensor(np.stack([text[s + 1:s + S + 1] for s in chunk])).long()
        ce = F.cross_entropy(z.reshape(-1, 27), y.reshape(-1), reduction='none').view(len(chunk), S)
        for i, s in enumerate(chunk):
            part = ce[i] if s == 0 else ce[i, half:]
            bits += float(part.sum()) / math.log(2); n += part.numel()
    return bits / n, n, len(starts) * S / max(n, 1)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--result', required=True, help='completed language_batched_benchmark result JSON')
    p.add_argument('--tag', required=True); p.add_argument('--lanes', type=int, default=64)
    p.add_argument('--seed', type=int, default=314159); p.add_argument('--test', type=int, default=0)
    p.add_argument('--windows', type=int, default=0, help='also score the window protocol (T) with the same stepper')
    p.add_argument('--window-lanes', type=int, default=256)
    p.add_argument('--deterministic', action='store_true', help='highest score wins (labelled variant)')
    p.add_argument('--limit', type=int, default=0, help='smoke: steps per lane / windows (0: all)')
    a = p.parse_args()
    out = OUT / f'{a.tag}.json'
    if out.exists():
        raise ValueError('Unique unused tag required')
    torch.set_num_threads(1); started = time.perf_counter()
    res_path = Path(a.result).resolve(); res = json.loads(res_path.read_text())
    if res.get('status') != 'completed':
        raise ValueError('Completed result required')
    args = res['args']
    weights = res_path.parent / 'checkpoints' / f"{args['tag']}_final.pt"
    model = build(args, weights)
    test_n = a.test or args['test']
    text = load_text(95_000_000, test_n)
    bits, count, L = stream(model, text, a.lanes, a.seed, a.deterministic, a.limit)
    t_stream = time.perf_counter() - started
    result = dict(status='completed' if not a.limit else 'smoke', args=vars(a), parent=str(res_path.relative_to(ROOT)),
                  parent_sha256=hashlib.sha256(res_path.read_bytes()).hexdigest(),
                  weights=str(weights.relative_to(ROOT)), weights_sha256=hashlib.sha256(weights.read_bytes()).hexdigest(),
                  parent_test_bpc_T256=res.get('test_bpc_eval_segment', res.get('test_bpc')),
                  protocol=dict(test=[95_000_000, 95_000_000 + test_n], lanes=L,
                                evaluator='SparseStepper winner-only (exact replay of the training evaluator)',
                                routing='deterministic (highest score)' if a.deterministic else f'sampled races, seed {a.seed}',
                                state='carried across the whole lane; reset only at the L lane starts',
                                evaluated_positions_per_target=1.0),
                  stream_bpc=float(bits.sum() / count.sum()), stream_targets=int(count.sum()),
                  by_context={f'{BUCKETS[j]}-{BUCKETS[j + 1]}': dict(bpc=float(bits[j] / count[j]), targets=int(count[j]))
                              for j in range(len(bits)) if count[j]},
                  stream_wall_s=t_stream,
                  hardware=dict(platform=platform.platform(), torch=torch.__version__, device='cpu', threads=1))
    if a.windows:
        w_bpc, w_n, per = windows(model, text, a.windows, a.seed, a.window_lanes, a.limit)
        result['window'] = dict(T=a.windows, bpc=w_bpc, targets=w_n, evaluated_positions_per_target=per,
                                wall_s=time.perf_counter() - started - t_stream)
    result['wall_s'] = time.perf_counter() - started
    OUT.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k in ('stream_bpc', 'by_context', 'window', 'parent_test_bpc_T256',
                                                                 'wall_s')}, indent=1))


if __name__ == '__main__':
    main()
