#!/usr/bin/env python3
"""B4 developed attempt (15 October 2026): the frozen unified race-of-clocks model of race_tpp_b4g.py with two generic
training/stability repairs, declared after the frozen-transfer verdicts (all seven lost against the composite target).

1. Resolution-bounded exponential clocks. No exponential clock may exceed one event per recording resolution r (the
   recording cell when the data have one, else the smallest positive TRAIN gap): log_rate <= -log r, applied as a smooth
   bound log_rate_max - softplus(log_rate_max - log_rate). This is the resolution principle already applied to the
   delayed clocks' spread; it caps the survival cost of an ordinary silence, which was unbounded in the split-0
   Wikipedia failure (published diagnosis, experiments/B4_NTPP_BENCHMARK_PROPOSAL.md).
2. Finite gradient repair. A batch whose loss or gradient norm is non-finite is not skipped: it is recomputed sequence
   by sequence, sequences with a non-finite loss or gradient are dropped from that update only, and the update is applied
   from the remaining sequences. The longest training sequences therefore keep training.
3. Optional local clipping (--local-clip): each parameter tensor is clipped to its own cap, calibrated on TRAIN as
   3x the median of that tensor's gradient norm over the first 32 batches (no validation use), instead of one global norm.
   A bank that produces an extreme gradient is limited locally and cannot rescale or corrupt the others.
Configuration, data rules, splits, selection (validation mean per-sequence L_T + L_M) and scoring are unchanged.
Disclosure: repair 1 was motivated by the published split-0 test-tail diagnosis; it has no tuned constant.
"""
import argparse, hashlib, json, math, random, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import race_tpp_v19 as base  # noqa: E402
from race_tpp_b4g import load, n_components, recording_cell, per_sequence  # noqa: E402

CFG = dict(d=32, modes=16, layers=2, n_exp=2, n_lognormal=8, dv=4, dropout=0.3, lr=3e-3, batch=32, state_modes=16)


class BoundedRaceTPP(base.RaceTPP):
    def set_rate_bound(self, resolution):
        self.log_rate_max = -math.log(resolution)

    def clocks(self, h, slots):
        out = list(super().clocks(h, slots))
        out[0] = self.log_rate_max - F.softplus(self.log_rate_max - out[0])
        return tuple(out)


CAPS = None


def clip(model):
    """Global norm 1.0, or per-tensor calibrated local caps. Returns a finite/non-finite indicator norm."""
    if CAPS is None: return nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    worst = torch.zeros(())
    for p, cap in zip(model.parameters(), CAPS):
        if p.grad is None: continue
        g = p.grad.norm(); worst = worst + g*0 if torch.isfinite(g) else g
        if torch.isfinite(g) and g > cap: p.grad.mul_(cap/g)
    return worst


def finite_update(model, opt, batch_seqs, cell, repairs, epoch, bi):
    t, m, mask = batch_seqs
    tl, ml, n, _ = model.loglik(t, m, mask, cell); loss = -(tl + ml)/n
    opt.zero_grad(); loss.backward(); gn = clip(model)
    if torch.isfinite(loss) and torch.isfinite(gn):
        opt.step(); return
    # Recompute per sequence; keep finite ones, then apply one clipped update from their summed gradients.
    keep_grads = None; kept = 0; events = 0; dropped = []
    for r in range(t.shape[0]):
        L = int(mask[r].sum())
        if L < 2: continue
        tr, mr, kr = t[r:r+1, :L], m[r:r+1, :L], mask[r:r+1, :L]
        a, b, nr, _ = model.loglik(tr, mr, kr, cell); lr_ = -(a + b)
        opt.zero_grad(); lr_.backward()
        grads = [p.grad.detach().clone() if p.grad is not None else None for p in model.parameters()]
        ok = torch.isfinite(lr_) and all(g is None or torch.isfinite(g).all() for g in grads)
        if not ok:
            dropped.append(dict(row=r, events=int(nr))); continue
        keep_grads = grads if keep_grads is None else [x + y if x is not None else None for x, y in zip(keep_grads, grads)]
        kept += 1; events += int(nr)
    opt.zero_grad()
    if keep_grads is not None and events:
        for p, g in zip(model.parameters(), keep_grads):
            if g is not None: p.grad = g/events
        clip(model); opt.step()
    repairs.append(dict(epoch=epoch, batch=bi, kept_sequences=kept, dropped=dropped[:8], dropped_count=len(dropped)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True); ap.add_argument('--split', type=int, required=True); ap.add_argument('--tag', required=True)
    ap.add_argument('--seed', type=int, default=0); ap.add_argument('--epochs', type=int, default=200); ap.add_argument('--patience', type=int, default=30)
    ap.add_argument('--max-wall-s', type=float, default=6*3600); ap.add_argument('--score-test', action='store_true'); ap.add_argument('--local-clip', action='store_true')
    a = ap.parse_args(); cfg = CFG
    torch.set_num_threads(1); torch.set_default_dtype(torch.float64)
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed); rng = random.Random(a.seed)
    train, K = load(a.dataset, a.split, 'train'); val, _ = load(a.dataset, a.split, 'val')
    gaps = np.concatenate([np.diff(t) for t, _ in train]); pos = gaps[gaps > 0]
    cell = recording_cell(pos); comps = n_components(pos); n_window = 4 if comps >= 2 else 0
    scale = float(np.median(pos)); resolution = cell if cell > 0 else float(np.min(pos))
    qs = np.log(np.quantile(pos, np.linspace(0.1, 0.9, cfg['n_lognormal']))).tolist()
    edges = base.cluster_windows(pos, n_window, a.seed) if n_window else None
    model = BoundedRaceTPP(K + 1, cfg['d'], cfg['modes'], cfg['layers'], cfg['n_exp'], cfg['n_lognormal'], cfg['dv'],
                           cfg['dropout'], scale, qs, cell, n_window, edges, cfg['state_modes'])
    model.set_rate_bound(resolution); params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=0.0)
    global CAPS
    if a.local_clip:                                                # TRAIN-only calibration of per-tensor caps
        norms = []; cal_rng = random.Random(a.seed + 1)
        for bi, (t, m, mask) in enumerate(base.batches(train, cfg['batch'], True, cal_rng)):
            if bi >= 32: break
            tl, ml, n, _ = model.loglik(t, m, mask, cell); opt.zero_grad(); (-(tl + ml)/n).backward()
            norms.append([float(p.grad.norm()) if p.grad is not None else 0. for p in model.parameters()])
        opt.zero_grad(); arr = np.array(norms); arr = np.where(np.isfinite(arr), arr, np.nan)
        CAPS = [max(3*float(np.nanmedian(arr[:, k])), 1e-6) for k in range(arr.shape[1])]
    out_dir = ROOT/'experiments/results/tpp_b4'; ckpt = out_dir/f'{a.tag}.pt'; out = out_dir/f'{a.tag}.json'; assert not out.exists()
    best, best_epoch, history, start = math.inf, -1, [], time.time(); repairs = []
    for epoch in range(a.epochs):
        model.train(); e0 = time.time()
        for bi, b in enumerate(base.batches(train, cfg['batch'], True, rng)): finite_update(model, opt, b, cell, repairs, epoch, bi)
        v = per_sequence(model, val)
        history.append(dict(epoch=epoch, val_L_T=v['L_T'], val_L_M=v['L_M'], val_total=v['total'], epoch_s=time.time() - e0)); print(json.dumps(history[-1]), flush=True)
        if math.isfinite(v['total']) and v['total'] < best:
            best, best_epoch = v['total'], epoch; torch.save(model.state_dict(), ckpt)
        if epoch - best_epoch >= a.patience or time.time() - start > a.max_wall_s: break
    model.load_state_dict(torch.load(ckpt))
    res = dict(status='completed', battle='B4', attempt='developed (resolution-bounded clocks, finite gradient repair' + (', calibrated local clipping)' if a.local_clip else ')'), tag=a.tag, dataset=a.dataset, split=a.split,
               args=vars(a), config=cfg, K=K, recording_cell=cell, rate_resolution=resolution, gap_components=int(comps), n_window=n_window, scale=scale, parameters=params,
               best_epoch=best_epoch, epochs_run=len(history), repaired_updates=len(repairs), repairs_first=repairs[:20], wall_s=time.time() - start, val=per_sequence(model, val),
               history=history, checkpoint=str(ckpt.relative_to(ROOT)))
    if a.score_test: res['test'] = per_sequence(model, load(a.dataset, a.split, 'test')[0])
    res['source_sha256'] = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in (Path(__file__).resolve(), ROOT/'experiments/tpp/race_tpp_v19.py', ROOT/'experiments/tpp/race_tpp_b4g.py')}
    out.write_text(json.dumps(res, indent=1) + '\n'); print('VAL', json.dumps(res['val']), flush=True)
    if a.score_test: print('TEST', json.dumps(res['test']), flush=True)


if __name__ == '__main__':
    main()
