"""R1 work accounting: measured matmul FLOPs per training token for plan-B deep keyed members (torch FlopCounterMode on
one forward+backward of a 256-token lane, the one-pass driver's lane length), against the pinned public reference's
6N + 6 n_layer T d per-token training FLOPs (Kaplan et al. convention; GPT-2 124M, T=1024)."""
import argparse, json, sys, hashlib
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.flop_counter import FlopCounterMode
ROOT = Path(__file__).resolve().parents[1]; sys.path[:0] = [str(ROOT/'experiments'), str(ROOT/'experiments/tpp')]
from r1_deep_keyed_lm import DeepKeyedLM
from r1_token_keyed_lm import load, V


def measure(d, layers, batch=1, lane=256):
    torch.manual_seed(0); m = DeepKeyedLM(d, 32, layers, 4, 16, 0.1); m.train()
    x = torch.from_numpy(np.asarray(load(ROOT/'data/fineweb_gpt2/fineweb_train_000001.bin')[:batch*lane]).astype(np.int64).reshape(batch, lane))
    with FlopCounterMode(display=False) as fc:
        h, e = m.states(x[:, :-1]); loss = F.cross_entropy(m.logits(h, e, x[:, :-1], list(range(lane - 1))).reshape(-1, V), x[:, 1:].reshape(-1)); loss.backward()
    return dict(d=d, layers=layers, parameters=sum(p.numel() for p in m.parameters()), train_flops_per_target=fc.get_total_flops()/(batch*(lane - 1)))


def main():
    p = argparse.ArgumentParser(); p.add_argument('--tag', required=True); a = p.parse_args(); torch.set_num_threads(1)
    rows = [measure(d, L) for d, L in ((64, 2), (64, 4), (64, 8), (64, 16), (128, 8))]
    n, nl, T, dm = 124e6, 12, 1024, 768
    ref = dict(model='GPT-2 124M (modded-nanogpt record 95a9fd44)', parameters=n, train_flops_per_token=6*n + 6*nl*T*dm, convention='6N + 6 n_layer T d (forward+backward matmul)')
    for r in rows: r['reference_flops_ratio'] = ref['train_flops_per_token']/r['train_flops_per_target']
    out = dict(status='completed', battle='R1', tag=a.tag, convention='measured matmul FLOPs (torch FlopCounterMode), forward+backward, one 1x256 lane (per-target work is batch-independent); elementwise/optimizer work excluded on both sides',
               members=rows, reference=ref, source_sha256={f: hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in ('analysis/plan_b_training_flops.py', 'experiments/r1_deep_keyed_lm.py')})
    with (ROOT/f'experiments/results/diagnostics/{a.tag}.json').open('x') as f: json.dump(out, f, indent=1); f.write('\n')
    print(json.dumps(out['members']), json.dumps(ref))


if __name__ == '__main__':
    main()
