#!/usr/bin/env python3
"""B1: score TEST once from a completed run's best-DEV checkpoint (evaluation only, no training).

Rebuilds the model exactly as its driver does, loads the checkpoint, and requires the run's recorded DEV log-likelihood
to be reproduced to 1e-9 before TEST is touched; otherwise it stops without scoring. Writes a new result file
(never overwrites): experiments/results/tpp/<out-prefix>_s<seed>.json, carrying the source run, its DEV record and TEST.
"""
import argparse, hashlib, importlib, json, random, sys
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--result', required=True, help='completed run JSON (with checkpoint)')
    ap.add_argument('--out-prefix', required=True, help='e.g. b1_final_amazon_v18')
    ap.add_argument('--dev-only', action='store_true', help='check the DEV reproduction contract; do not score TEST')
    a = ap.parse_args()
    torch.set_num_threads(1); torch.set_default_dtype(torch.float64)
    r = json.loads((ROOT / a.result).read_text()); assert r['status'] == 'completed'
    drv = next(iter(r['source_sha256']))
    assert hashlib.sha256((ROOT / drv).read_bytes()).hexdigest() == r['source_sha256'][drv], 'driver changed since the run'
    mod = importlib.import_module(Path(drv).stem); x = r['args']
    out = ROOT / 'experiments/results/tpp' / f"{a.out_prefix}_s{x['seed']}.json"
    assert not out.exists(), f'{out} exists; results are never overwritten'
    train = mod.load_split(x['dataset'], 'train')
    gaps = np.concatenate([np.diff(t) for t, _ in train]); pos = gaps[gaps > 0]
    qs = np.log(np.quantile(pos, np.linspace(0.1, 0.9, x['n_lognormal']))).tolist()
    edges = mod.cluster_windows(pos, x['n_window'], x['seed']) if x.get('n_window') else None
    m = mod.RaceTPP(r['K'], x['d'], x['modes'], x['layers'], x['n_exp'], x['n_lognormal'], x['dv'], x['dropout'],
                    r['scale'], qs, x['floor_cell'], x['n_window'], edges, x['state_modes'])
    m.load_state_dict(torch.load(ROOT / r['checkpoint'])); m.eval()
    rng = random.Random(x['seed'])
    dev = mod.evaluate(m, mod.load_split(x['dataset'], 'dev'), 8, rng)
    diff = abs(dev['ll'] - r['dev']['ll'])
    print(f"DEV reproduced {dev['ll']:.10f} vs recorded {r['dev']['ll']:.10f} (|diff| {diff:.2e})", flush=True)
    assert diff < 1e-9, 'DEV not reproduced: TEST not scored'
    if a.dev_only:
        return
    test = mod.evaluate(m, mod.load_split(x['dataset'], 'test'), 8, rng)
    me = Path(__file__).resolve()
    res = dict(status='completed', battle='B1', tag=out.stem, dataset=x['dataset'], args=x, K=r['K'], scale=r['scale'],
               parameters=r['parameters'], best_epoch=r['best_epoch'], scored_from=a.result, checkpoint=r['checkpoint'],
               dev=r['dev'], dev_reproduced=dev['ll'], test=test,
               source_sha256={drv: r['source_sha256'][drv],
                              str(me.relative_to(ROOT)): hashlib.sha256(me.read_bytes()).hexdigest()})
    out.write_text(json.dumps(res, indent=1) + '\n')
    print('TEST', json.dumps(test), flush=True)


if __name__ == '__main__':
    main()
