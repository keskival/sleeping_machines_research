"""History-use curve of a saved dense reference LSTM (evaluation only; no training).

Companion to experiments/language_stream_rescore.py --context-curve: the same test interval text8[95M:95M+test], the
same history bins, disjoint windows of C from a fresh (zero) state, every target scored by its position in the window.
Also reports the saved protocol (one stream with carried state, as e64_lm_baselines.score) as a reproduction check,
and the E64 T-window protocol with reset state (the native protocol).  Answers where the LSTM's lead over the native
model sits: in the recent history (short bins) or in the long tail (long bins / streaming).
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
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'experiments'))
from e64_lm_baselines import A, LSTMLM  # noqa: E402
from language_batched_benchmark import load_text  # noqa: E402
from language_stream_rescore import CURVE  # noqa: E402

OUT = ROOT / 'experiments/results/language_stream'


@torch.no_grad()
def ce_bits(net, x, y, state=None):
    logits, state = net(x, state)
    return nn.functional.cross_entropy(logits.reshape(-1, A), y.reshape(-1), reduction='none').view(y.shape) / math.log(2), state


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True); p.add_argument('--tag', required=True)
    p.add_argument('--test', type=int, default=1_000_000); p.add_argument('--context-curve', type=int, default=1024,
                                                                       help='0: skip the history curve')
    p.add_argument('--offset', type=int, default=95_000_000, help='interval start (90_000_000: the E64 validation interval)')
    p.add_argument('--window', type=int, default=256); p.add_argument('--batch', type=int, default=256)
    a = p.parse_args()
    out = OUT / f'{a.tag}.json'
    if out.exists():
        raise ValueError('Unique unused tag required')
    torch.set_num_threads(1); started = time.perf_counter()
    ck_path = Path(a.checkpoint).resolve(); ck = torch.load(ck_path, weights_only=False)
    args = ck['args']
    if args['model'] != 'lstm':
        raise ValueError('LSTM checkpoint required')
    net = LSTMLM(args['size'], args.get('dropout', 0.0)); net.load_state_dict(ck['state']); net.eval()
    text = torch.from_numpy(load_text(a.offset, a.test).astype(np.int64))
    n = len(text)
    # 1. saved protocol: one stream, carried state, blocks of 4096 (e64_lm_baselines.score)
    tot = 0.; cnt = 0; state = None
    for s0 in range(0, n - 1, 4096):
        y = text[s0 + 1:s0 + 4097][None]; x = text[s0:s0 + y.shape[1]][None]
        b, state = ce_bits(net, x, y, state); tot += float(b.sum()); cnt += y.numel()
    stream = tot / cnt
    # 2. E64 window protocol with reset state: windows of T at stride T/2, second half after the first window
    T = a.window; half = T // 2; starts = list(range(0, n - T - 1, half)); tot = 0.; cnt = 0
    for i in range(0, len(starts), a.batch):
        s = starts[i:i + a.batch]
        x = torch.stack([text[j:j + T] for j in s]); y = torch.stack([text[j + 1:j + T + 1] for j in s])
        b, _ = ce_bits(net, x, y)
        for r, j in enumerate(s):
            part = b[r] if j == 0 else b[r, half:]
            tot += float(part.sum()); cnt += part.numel()
    window = tot / cnt
    curve = None
    if a.context_curve:
        # 3. history curve: disjoint windows of C from a fresh state, binned by position
        C = a.context_curve; edges = [e for e in CURVE if e < C] + [C]
        which = np.searchsorted(edges, np.arange(C), side='right') - 1
        bits = np.zeros(len(edges) - 1); count = np.zeros(len(edges) - 1, dtype=np.int64)
        starts = list(range(0, n - C - 1, C))
        for i in range(0, len(starts), a.batch):
            s = starts[i:i + a.batch]
            x = torch.stack([text[j:j + C] for j in s]); y = torch.stack([text[j + 1:j + C + 1] for j in s])
            b, _ = ce_bits(net, x, y)
            per_pos = b.sum(0).numpy()
            for k in range(C):
                bits[which[k]] += per_pos[k]; count[which[k]] += len(s)
    result = dict(status='completed', args=vars(a), checkpoint=str(ck_path.relative_to(ROOT)),
                  checkpoint_sha256=hashlib.sha256(ck_path.read_bytes()).hexdigest(), reference_args=args,
                  stream_bpc=stream, stream_targets=n - 1,
                  window=dict(T=T, bpc=window, targets=cnt, state='reset per window'),
                  interval=[a.offset, a.offset + a.test],
                  context_curve=dict(C=C, by_history={f'{edges[j]}-{edges[j + 1]}': dict(bpc=float(bits[j] / count[j]),
                                                                                         targets=int(count[j]))
                                                      for j in range(len(bits)) if count[j]}) if a.context_curve else None,
                  wall_s=time.perf_counter() - started,
                  hardware=dict(platform=platform.platform(), torch=torch.__version__, device='cpu', threads=1),
                  scope='Evaluation of saved weights only; no training. Bins and test interval match language_stream_rescore.')
    OUT.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1) + '\n')
    print(json.dumps({k: result[k] for k in ('stream_bpc', 'window', 'context_curve', 'wall_s')}, indent=1))


if __name__ == '__main__':
    main()
