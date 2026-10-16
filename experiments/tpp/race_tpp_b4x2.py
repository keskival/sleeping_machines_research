#!/usr/bin/env python3
"""B4 developed fighter v2 (15 October 2026; v1 plus exact token-budgeted micro-batching so long-sequence batches fit the
memory cap: the update is identical, gradients are accumulated over micro-batches of at most --event-budget padded events): the strongest assembled race-of-clocks construction from the B4 development
line, on any dataset/split, validation-selected; TEST is scored later, once, by b4x_score.py from the saved checkpoint.

Model: race_tpp_logspread_marks.RaceTPP = log-domain clock-spread repair (zero skipped updates where the guarded run
skipped up to 60) + established addressed per-event mark memory (direct read gains 13.8 nats/sequence on Github DEV).
Training: B4's unchanged data rules (recording cell, window clocks when the log-gap histogram has >= 2 components),
unchanged core configuration and 30-pass validation patience; a non-finite batch is repaired sequence by sequence instead
of skipped (race_tpp_b4d.finite_update); optional calibrated local clipping (--local-clip, per-tensor caps from TRAIN).
"""
import argparse, hashlib, json, math, random, sys, time
from pathlib import Path
import numpy as np
import torch
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(Path(__file__).resolve().parent))
import race_tpp_v19 as base  # noqa: E402
import race_tpp_b4d as rep  # noqa: E402
from race_tpp_b4g import load, n_components, recording_cell, per_sequence  # noqa: E402
from race_tpp_logspread_marks import RaceTPP as MarksTPP  # noqa: E402
from race_tpp_pointer import RaceTPP as PointerTPP  # noqa: E402

CFG = dict(d=32, modes=16, layers=2, n_exp=2, n_lognormal=8, dv=4, dropout=0.3, lr=3e-3, batch=32, state_modes=16)
SOURCES = ['experiments/tpp/race_tpp_b4x2.py', 'experiments/tpp/race_tpp_b4x.py', 'experiments/tpp/race_tpp_pointer.py', 'experiments/tpp/race_tpp_b4d.py', 'experiments/tpp/race_tpp_b4g.py', 'experiments/tpp/race_tpp_logspread_marks.py',
           'experiments/tpp/race_tpp_logspread.py', 'experiments/tpp/race_tpp_v20.py', 'experiments/tpp/race_tpp_v19.py']


def build(dataset, split, seed, pointer=False):
    RaceTPP = PointerTPP if pointer else MarksTPP
    train, K = load(dataset, split, 'train'); gaps = np.concatenate([np.diff(t) for t, _ in train]); pos = gaps[gaps > 0]
    cell = recording_cell(pos); comps = n_components(pos); n_window = 4 if comps >= 2 else 0; scale = float(np.median(pos))
    qs = np.log(np.quantile(pos, np.linspace(0.1, 0.9, CFG['n_lognormal']))).tolist(); edges = base.cluster_windows(pos, n_window, seed) if n_window else None
    model = RaceTPP(K + 1, CFG['d'], CFG['modes'], CFG['layers'], CFG['n_exp'], CFG['n_lognormal'], CFG['dv'], CFG['dropout'], scale, qs, cell, n_window, edges, CFG['state_modes'])
    return model, train, K, dict(recording_cell=cell, gap_components=int(comps), n_window=n_window, scale=scale)


def micro_update(model, opt, batch, cell, repairs, epoch, bi, budget):
    """Exact gradient accumulation over row micro-batches of at most `budget` padded events; non-finite -> sequence repair."""
    t, m, mask = batch; Lb = t.shape[1]; rows = max(1, budget//max(Lb, 1)); total = int(mask[:, 1:].sum())
    if rows >= t.shape[0]: return rep.finite_update(model, opt, batch, cell, repairs, epoch, bi)
    opt.zero_grad(); finite = True
    for r0 in range(0, t.shape[0], rows):
        sl = slice(r0, r0 + rows); L = int(mask[sl].sum(1).max()); tl, ml, n, _ = model.loglik(t[sl, :L], m[sl, :L], mask[sl, :L], cell)
        loss = -(tl + ml)/total
        if not torch.isfinite(loss): finite = False; break
        loss.backward()
    gn = rep.clip(model) if finite else torch.tensor(float('nan'))
    if finite and torch.isfinite(gn): opt.step(); return
    opt.zero_grad(); rep.finite_update(model, opt, batch, cell, repairs, epoch, bi)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--dataset', required=True); ap.add_argument('--split', type=int, required=True); ap.add_argument('--tag', required=True)
    ap.add_argument('--seed', type=int, default=0); ap.add_argument('--epochs', type=int, default=200); ap.add_argument('--patience', type=int, default=30)
    ap.add_argument('--max-wall-s', type=float, default=6*3600); ap.add_argument('--local-clip', action='store_true'); ap.add_argument('--pointer', action='store_true'); ap.add_argument('--event-budget', type=int, default=4096)
    a = ap.parse_args(); torch.set_num_threads(1); torch.set_default_dtype(torch.float64)
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed); rng = random.Random(a.seed)
    model, train, K, rules = build(a.dataset, a.split, a.seed, a.pointer); val, _ = load(a.dataset, a.split, 'val'); cell = rules['recording_cell']
    opt = torch.optim.AdamW(model.parameters(), lr=CFG['lr'], weight_decay=0.0)
    if a.local_clip:
        norms = []; cal_rng = random.Random(a.seed + 1)
        for bi, (t, m, mask) in enumerate(base.batches(train, CFG['batch'], True, cal_rng)):
            if bi >= 32: break
            tl, ml, n, _ = model.loglik(t, m, mask, cell); opt.zero_grad(); (-(tl + ml)/n).backward()
            norms.append([float(p.grad.norm()) if p.grad is not None else 0. for p in model.parameters()])
        opt.zero_grad(); arr = np.array(norms); arr = np.where(np.isfinite(arr), arr, np.nan)
        rep.CAPS = [max(3*float(np.nanmedian(arr[:, k])), 1e-6) for k in range(arr.shape[1])]
    out_dir = ROOT/'experiments/results/tpp_b4'; ckpt = out_dir/f'{a.tag}.pt'; out = out_dir/f'{a.tag}.json'; assert not out.exists()
    best, best_epoch, history, start, repairs = math.inf, -1, [], time.time(), []
    for epoch in range(a.epochs):
        model.train(); e0 = time.time()
        for bi, b in enumerate(base.batches(train, CFG['batch'], True, rng)): micro_update(model, opt, b, cell, repairs, epoch, bi, a.event_budget)
        v = per_sequence(model, val)
        history.append(dict(epoch=epoch, val_L_T=v['L_T'], val_L_M=v['L_M'], val_total=v['total'], epoch_s=time.time() - e0)); print(json.dumps(history[-1]), flush=True)
        if math.isfinite(v['total']) and v['total'] < best: best, best_epoch = v['total'], epoch; torch.save(model.state_dict(), ckpt)
        if epoch - best_epoch >= a.patience or time.time() - start > a.max_wall_s: break
    model.load_state_dict(torch.load(ckpt))
    res = dict(status='completed', battle='B4', attempt='developed: logspread clocks + addressed mark memory' + (' + episodic pointer race' if a.pointer else '') + ' + finite repair' + (' + calibrated local clipping' if a.local_clip else ''),
               tag=a.tag, dataset=a.dataset, split=a.split, args=vars(a), config=CFG, K=K, **rules, parameters=sum(p.numel() for p in model.parameters()),
               best_epoch=best_epoch, epochs_run=len(history), repaired_updates=len(repairs), repairs_first=repairs[:10], wall_s=time.time() - start,
               val=per_sequence(model, val), history=history, checkpoint=str(ckpt.relative_to(ROOT)), test='not scored here (b4x_score.py, once, selected variant)',
               source_sha256={n: hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in SOURCES})
    out.write_text(json.dumps(res, indent=1) + '\n'); print('VAL', json.dumps(res['val']), flush=True)


if __name__ == '__main__':
    main()
