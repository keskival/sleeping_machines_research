#!/usr/bin/env python3
"""G2: one shared temporal core across several EasyTPP datasets (development; DEV only).

Each dataset keeps its own mark embedding, addressed mark memory and clock/mark head (mark sets differ); the temporal
memory layers and the gap encoder are one shared module trained on all datasets at once (batches interleaved in
proportion to dataset size). Per-dataset DEV LL is tracked every epoch and the best epoch is kept per dataset. Compare
with the per-dataset v8 models of the same encoder size.
"""
import argparse, hashlib, json, math, random, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import race_tpp_v8 as base  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--datasets', default='taxi,taobao,stackoverflow,amazon')
    ap.add_argument('--tag', required=True)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--epochs', type=int, default=150)
    ap.add_argument('--patience', type=int, default=30)
    ap.add_argument('--lr', type=float, default=3e-3)
    ap.add_argument('--batch', type=int, default=32)
    ap.add_argument('--dropout', type=float, default=0.3)
    ap.add_argument('--n-lognormal', type=int, default=8)
    ap.add_argument('--n-window', type=int, default=4)
    a = ap.parse_args()
    torch.set_num_threads(1); torch.set_default_dtype(torch.float64)
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed); rng = random.Random(a.seed)
    names = a.datasets.split(',')
    data, models = {}, {}
    shared_layers = shared_gap = None
    for ds in names:
        train, dev = base.load_split(ds, 'train'), base.load_split(ds, 'dev')
        K = max(int(m.max()) for _, m in train + dev) + 1
        gaps = np.concatenate([np.diff(t) for t, _ in train]); pos = gaps[gaps > 0]
        scale = float(np.median(pos)); qs = np.log(np.quantile(pos, np.linspace(0.1, 0.9, a.n_lognormal))).tolist()
        qq = np.quantile(pos, np.linspace(0.02, 0.98, a.n_window + 1))
        m = base.RaceTPP(K, 32, 16, 2, 2, a.n_lognormal, 4, a.dropout, scale, qs, 0.0, a.n_window,
                         (qq[:-1].tolist(), qq[1:].tolist()))
        if shared_layers is None:
            shared_layers, shared_gap = m.layers, m.gap
        else:
            m.layers, m.gap = shared_layers, shared_gap                      # one temporal core for all datasets
        data[ds] = (train, dev); models[ds] = m
    params = list({id(p): p for m in models.values() for p in m.parameters()}.values())
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)
    core = sum(p.numel() for p in list(shared_layers.parameters()) + list(shared_gap.parameters()))
    total = sum(p.numel() for p in params)
    out_dir = ROOT / 'experiments/results/tpp'
    best = {ds: (-math.inf, -1) for ds in names}; history = []; start = time.time()
    for epoch in range(a.epochs):
        plan = []
        for ds in names:
            plan += [(ds, b) for b in base.batches(data[ds][0], a.batch, True, rng)]
        rng.shuffle(plan)
        for m in models.values():
            m.train()
        for ds, (t, mk, mask) in plan:
            tl, ml, n, _ = models[ds].loglik(t, mk, mask)
            loss = -(tl + ml) / n
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(params, 1.0); opt.step()
        row = dict(epoch=epoch)
        for ds in names:
            dv = base.evaluate(models[ds], data[ds][1], 64, rng, predict=False)
            row[ds] = dv['ll']
            if dv['ll'] > best[ds][0]:
                best[ds] = (dv['ll'], epoch); torch.save(models[ds].state_dict(), out_dir / f'{a.tag}_{ds}.pt')
        history.append(row); print(json.dumps(row), flush=True)
        if all(epoch - best[ds][1] >= a.patience for ds in names):
            break
    src = Path(__file__).resolve()
    res = dict(status='completed', track='G2', tag=a.tag, args=vars(a), datasets=names, shared_core_parameters=core,
               total_parameters=total, best_dev={ds: dict(ll=best[ds][0], epoch=best[ds][1]) for ds in names},
               history=history, wall_s=time.time() - start,
               source_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                              for p in (src, ROOT / 'experiments/tpp/race_tpp_v8.py')})
    (out_dir / f'{a.tag}.json').write_text(json.dumps(res, indent=1) + '\n')
    print('BEST', json.dumps(res['best_dev']), flush=True)


if __name__ == '__main__':
    main()
