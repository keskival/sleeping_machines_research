"""R1 throughput: plan-B deep keyed blocks eager vs torch.compile (identical batches; loss agreement and s/update).

Builds DeepKeyedLM (r1_deep_keyed_lm.py) twice from the same initial weights, compiles each Block of the second copy,
runs the same N training updates (AdamW, clipping, dropout off for exact comparison) and records per-update losses,
steady-state seconds per update (after compile warm-up) and peak RSS.
"""
import argparse, json, sys, time, hashlib, resource, copy
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
ROOT = Path(__file__).resolve().parents[1]; sys.path[:0] = [str(ROOT/'experiments'), str(ROOT/'experiments/tpp')]
from r1_deep_keyed_lm import DeepKeyedLM
from r1_token_keyed_lm import load, V


def run(model, batches, lane):
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01); losses, times = [], []
    for x in batches:
        t0 = time.time(); h, e = model.states(x[:, :-1]); loss = F.cross_entropy(model.logits(h, e, x[:, :-1], list(range(lane - 1))).reshape(-1, V), x[:, 1:].reshape(-1))
        opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); times.append(time.time() - t0); losses.append(loss.item())
    return losses, times


def main():
    p = argparse.ArgumentParser(); p.add_argument('--tag', required=True); p.add_argument('--layers', type=int, default=8); p.add_argument('--batch', type=int, default=4)
    p.add_argument('--lane', type=int, default=256); p.add_argument('--updates', type=int, default=12); p.add_argument('--threads', type=int, default=1); a = p.parse_args()
    torch.set_num_threads(a.threads); torch.manual_seed(0); tr = load(ROOT/'data/fineweb_gpt2/fineweb_train_000001.bin')
    data = np.asarray(tr[:a.updates*a.batch*a.lane]).astype(np.int64).reshape(a.updates, a.batch, a.lane); batches = [torch.from_numpy(b) for b in data]
    eager = DeepKeyedLM(64, 32, a.layers, 4, 16, 0.0); comp = copy.deepcopy(eager)
    for i, b in enumerate(comp.blocks): comp.blocks[i] = torch.compile(b, dynamic=False)
    le, te = run(eager, batches, a.lane); lc, tc = run(comp, batches, a.lane)
    warm = 3; out = dict(status='completed', battle='R1', tag=a.tag, args=vars(a), eager_losses=le, compiled_losses=lc, max_abs_loss_diff=max(abs(x - y) for x, y in zip(le, lc)),
                         eager_s_per_update=float(np.mean(te[warm:])), compiled_s_per_update=float(np.mean(tc[warm:])), compile_warmup_s=float(sum(tc[:warm])),
                         speedup=float(np.mean(te[warm:]) / np.mean(tc[warm:])), peak_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                         source_sha256={n: hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in ('analysis/plan_b_compile_benchmark.py', 'experiments/r1_deep_keyed_lm.py')})
    with (ROOT/f'experiments/results/diagnostics/{a.tag}.json').open('x') as f: json.dump(out, f, indent=2); f.write('\n')
    print(json.dumps({k: out[k] for k in ('max_abs_loss_diff', 'eager_s_per_update', 'compiled_s_per_update', 'speedup', 'compile_warmup_s', 'peak_rss_kb')}))


if __name__ == '__main__':
    main()
