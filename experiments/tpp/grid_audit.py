#!/usr/bin/env python3
"""B1 audit: does a trained model exploit the recording grid of event times? (DEV only, evaluation only.)

Scores DEV as released and with every gap dequantized uniformly within its recording cell (integer seconds on Retweet:
g -> g + U(-1/2, 1/2), zero gaps -> U(0, 1/2); generic grids via --cell). A density that is smooth at the scale of the
grid scores nearly the same on both; a density with spikes on grid points loses heavily on the dequantized data.
Also reports learned clock dispersions. Compare with the same audit of a model trained under a resolution floor.
"""
import argparse, importlib, json, math, random, sys
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--result', required=True)
    ap.add_argument('--cell', type=float, default=1.0)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--quad-nodes', type=int, default=0, help='re-evaluate with this many compensator nodes (state clock)')
    a = ap.parse_args()
    torch.set_num_threads(1); torch.set_default_dtype(torch.float64)
    r = json.loads((ROOT / a.result).read_text())
    src = next(iter(r['source_sha256'])); mod = importlib.import_module(Path(src).stem)
    args = r['args']; ds = args['dataset']
    train, dev = mod.load_split(ds, 'train'), mod.load_split(ds, 'dev')
    gaps = np.concatenate([np.diff(t) for t, _ in train]); pos = gaps[gaps > 0]
    qs = np.log(np.quantile(pos, np.linspace(0.1, 0.9, args['n_lognormal']))).tolist()
    kw = {k: args[k] for k in ('floor_cell',) if k in args}
    if args.get('n_window', 0):
        kw.update(n_window=args['n_window'], window_edges=([0.1] * args['n_window'], [0.2] * args['n_window']))
    if args.get('state_modes', 0):
        kw.update(state_modes=args['state_modes'])
        if a.quad_nodes:
            kw.update(quad_nodes=a.quad_nodes)
    model = mod.RaceTPP(r['K'], args['d'], args['modes'], args['layers'], args['n_exp'], args['n_lognormal'], args['dv'],
                        0.0, r['scale'], qs, **kw)
    state = torch.load(ROOT / r['checkpoint'])
    if a.quad_nodes:
        state = {k: v for k, v in state.items() if k not in ('gl_x', 'gl_w')}
    model.load_state_dict(state, strict=not a.quad_nodes); model.eval()
    rng = np.random.default_rng(a.seed)
    deq = []
    for t, m in dev:
        g = np.diff(t)
        g = np.where(g > 0, g + rng.uniform(-a.cell / 2, a.cell / 2, g.shape), rng.uniform(0, a.cell / 2, g.shape))
        deq.append((np.concatenate([[t[0]], t[0] + np.cumsum(np.maximum(g, 1e-9))]), m))
    raw = mod.evaluate(model, dev, 64, random.Random(0), predict=False)
    dq = mod.evaluate(model, deq, 64, random.Random(0), predict=False)
    sig = []
    with torch.no_grad():
        for tt, mm, mask in mod.batches(dev[:200], 64, False, random.Random(0)):
            h, slots = model.encode(tt, mm, mask)
            _, mu, sigma, _, _ = model.clocks(h, slots)
            sigma = sigma[..., :args['n_lognormal']]
            sig.append(sigma[mask].reshape(-1, sigma.shape[-1]))
    sig = torch.cat(sig)
    out = dict(result=a.result, cell=a.cell, dev_raw=raw, dev_dequantized=dq, drop=raw['ll'] - dq['ll'],
               sigma_quantiles=np.quantile(sig.numpy(), [0.01, 0.1, 0.5, 0.9]).tolist(),
               frac_sigma_below_0_05=float((sig < 0.05).double().mean()))
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
