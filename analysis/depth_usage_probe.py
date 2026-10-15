"""R1 depth ladder: how much each depth's features carry in the trained readout (no fitting).

For each completed arm, one forward pass over the recorded long-history DEV protocol
(8192 targets in 1024-token windows, 8 lanes) collects the readout input, which is the
concatenation of every stage's output. Then, with the readout fixed:
  * per-depth mean ablation: replace one stage's slice by its DEV mean (loss increase = that stage's unique value);
  * prefix truncation: keep stages 0..k-1, mean-ablate the rest (marginal value of going deeper);
  * per-depth readout weight energy share.
A stack whose deep stages carry nothing has flat prefix curves after the first stages.
"""
import argparse, json, sys, time, hashlib
from pathlib import Path
from types import SimpleNamespace
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'experiments')]
import torch
import residual_depth_token_language as driver
from public_token_scoring import windows, reference_forward


def load(tag):
    ck = torch.load(ROOT / f'experiments/results/token_language/{tag}.pt', weights_only=False); args = SimpleNamespace(**ck['settings'])
    tr = driver.load_tokens(args.train_file); counts = driver.BlockLaneTokens(tr, args.train_tokens, args.lanes, args.exposure, args.steps, args.chunk).counts()
    order = torch.argsort(counts, descending=True, stable=True)
    torch.manual_seed(args.seed); model = driver.Model(args, order, counts); model.load_state_dict(ck['model'])
    for name in ('site_generator', 'position_generator', 'local_generator', 'alternative_generator'): getattr(model, name).set_state(ck[name])
    return model, args


@torch.no_grad()
def features(model, args, dev):
    model.eval(); forward = reference_forward(model, driver); gen = torch.Generator().manual_seed(args.seed + 100000); xs, ys = [], []
    for inputs, labels, _ in windows(dev, 25165824, 8192, 1024, 8):
        state = None
        for pos in range(0, 1024, args.chunk):
            x, state, _ = forward(inputs[:, pos:pos + args.chunk], state, gen, args, deterministic=args.deterministic_eval)
            xs.append(x.reshape(-1, x.shape[-1])); ys.append(labels[:, pos:pos + args.chunk].reshape(-1))
    return torch.cat(xs), torch.cat(ys)


@torch.no_grad()
def nll(model, x, y):
    return float(sum(model.readout.nll(x[i:i + 1024], y[i:i + 1024]).double().sum() for i in range(0, len(y), 1024)) / len(y))


def readout_weight_share(model, depth, width):
    energy = torch.zeros(depth)
    for name, p in model.readout.named_parameters():
        if p.dim() == 2 and depth * width in p.shape:
            w = p if p.shape[-1] == depth * width else p.T
            energy += w.detach().reshape(w.shape[0], depth, width).square().sum((0, 2)).double().float()
    return (energy / energy.sum().clamp_min(1e-30)).tolist() if energy.sum() > 0 else None


def main():
    p = argparse.ArgumentParser(); p.add_argument('--tag', required=True); p.add_argument('--arms', nargs='+', required=True); a0 = p.parse_args()
    torch.set_num_threads(1); began = time.monotonic(); rows = {}
    for run in a0.arms:
        model, args = load(run); dev = driver.load_tokens(args.dev_file)
        x, y = features(model, args, dev); D = args.depth; W = x.shape[1] // D; mean = x.mean(0)
        full = nll(model, x, y)
        def ablate(keep):
            z = x.clone().reshape(-1, D, W); m = mean.reshape(D, W)
            for d in range(D):
                if not keep(d): z[:, d] = m[d]
            return nll(model, z.reshape(len(y), -1), y)
        single = [ablate(lambda d, k=k: d != k) - full for k in range(D)]
        prefix = {k: ablate(lambda d, k=k: d < k) for k in sorted({1, 2, 4, 8, 16, D}) if k <= D}
        energy = (x.reshape(-1, D, W).double().square().sum((0, 2)) / x.double().square().sum()).tolist()
        rows[run] = dict(depth=D, residual=bool(getattr(args, 'residual', False)), stage_reads=bool(getattr(args, 'stage_reads', False)), nll=full,
                         ablation_loss_increase_by_depth=single, prefix_nll=prefix, feature_energy_share_by_depth=energy,
                         readout_weight_share_by_depth=readout_weight_share(model, D, W))
        print(run, 'nll', round(full, 4), 'ablate', [round(v, 3) for v in single], 'prefix', {k: round(v, 3) for k, v in prefix.items()}, flush=True)
    out = dict(status='completed', battle='R1', tag=a0.tag, rows=rows, protocol=dict(targets=8192, target_start=25165825, window=1024, lanes=8, eos_state_reset=False, ablation='slice replaced by DEV mean'),
               source_sha256={n: hashlib.sha256((ROOT / n).read_bytes()).hexdigest() for n in ('analysis/depth_usage_probe.py', 'experiments/residual_depth_token_language.py', 'sleeping_machines/residual_race_episodes.py')},
               process_wall_s=time.monotonic() - began, scope='Frozen single-seed 1M-target ladder arms; readout fixed, mean ablation of readout inputs.')
    with (ROOT / f'experiments/results/diagnostics/{a0.tag}.json').open('x') as f: json.dump(out, f, indent=2); f.write('\n')


if __name__ == '__main__':
    main()
