#!/usr/bin/env python3
"""B4 DEVELOPMENT (validation only): race_tpp_v20 with the persistent mark pathways (--mark-mem, --mark-stats), otherwise the
unified configuration. Diagnosis and theory: B4 MOOC analysis (9 Oct). Copy of race_tpp_b4.py (pinned by the protocol).

Original B4 description: the frozen unified race-of-clocks model (race_tpp_v19, unchanged) on the neural-TPP benchmark of Bosser & Ben
Taieb (TMLR 2023; five fixed random 60/20/20 splits; data and splits from github.com/tanguybosser/ntpp-tmlr2023).

Protocol of the reference code (commit 54c15fd): with no window, the observation window is [0, t_last]; per sequence
    L_T = −Σ_i log f(τ_i)   over all events, the first measured from 0 with an empty history, no terminal survival,
    L_M = −Σ_i log p(k_i | τ_i, history),
reported as the mean over test sequences. Adapter: each sequence is prefixed by a start event at t = 0 with a dedicated
start mark (index K); the model then scores every real event exactly as above. The start mark is never a target, so
scoring over K + 1 marks can only make L_M conservative.

Two data rules, as in B1's unified protocol, computed on TRAIN only:
  (1) logistic windows only when the log-gap histogram separates into >= 2 components of >= 5% each;
  (2) recording cell: the largest c = q/j (q = smallest positive gap, j = 1..10) with >= 95% of positive gaps integer
      multiples of c (tolerance 1% of c); none otherwise. The cell drives the resolution principle and target-only
      dequantization.
Selection on validation mean per-sequence L_T + L_M; TEST scored once with --score-test.
"""
import argparse, hashlib, json, math, random, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import race_tpp_v20 as base  # noqa: E402

DATA = ROOT / 'data/ntpp_bosser/extracted/processed/data'


def load(ds, split, part):
    raw = json.loads((DATA / ds / f'split_{split}' / f'{part}.json').read_text())
    K = json.loads((DATA / ds / f'split_{split}' / 'args.json').read_text())['marks']
    seqs = []
    for s in raw:
        assert all(len(e['labels']) == 1 for e in s), 'multi-label events are not part of the marked protocol'
        t = np.array([0.0] + [float(e['time']) for e in s], np.float64)
        m = np.array([K] + [int(e['labels'][0]) for e in s], np.int64)
        assert np.all(np.diff(t) >= 0)
        seqs.append((t, m))
    return seqs, K


def n_components(pos, bins=200, empty=1e-4, minfrac=0.05):
    lg = np.log(pos); h, e = np.histogram(lg, bins=bins); occ = h > empty * len(lg)
    comps, cur = [], None
    for i, o in enumerate(occ):
        if o and cur is None:
            cur = [i, i]
        elif o:
            cur[1] = i
        elif cur is not None:
            comps.append(cur); cur = None
    if cur is not None:
        comps.append(cur)
    return sum(((lg >= e[a]) & (lg <= e[b + 1])).mean() >= minfrac for a, b in comps)


def recording_cell(pos, frac=0.95, tol=0.01):
    q = float(np.min(pos))
    for j in range(1, 11):
        c = q / j; r = pos / c
        if np.mean(np.abs(r - np.round(r)) < tol) >= frac:
            return c
    return 0.0


@torch.no_grad()
def per_sequence(model, seqs, batch=32):
    """Mean over sequences of the summed log densities: returns L_T, L_M (positive NLLs, lower is better)."""
    model.eval(); lt = lm = 0.0
    for t, m, mask in base.batches(seqs, batch, False, random.Random(0)):
        time_lp, joint_lp, _ = model.event_terms(t, m, mask)
        v = mask[:, 1:].to(time_lp.dtype)
        lt += float((time_lp * v).sum()); lm += float(((joint_lp - time_lp) * v).sum())
    n = len(seqs)
    return dict(L_T=-lt / n, L_M=-lm / n, total=-(lt + lm) / n, sequences=n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True)
    ap.add_argument('--split', type=int, required=True)
    ap.add_argument('--tag', required=True)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--epochs', type=int, default=200)
    ap.add_argument('--patience', type=int, default=30)
    ap.add_argument('--max-wall-s', type=float, default=6 * 3600)
    ap.add_argument('--mark-mem', type=int, default=0)
    ap.add_argument('--mark-stats', action='store_true')
    ap.add_argument('--d', type=int, default=32)
    a = ap.parse_args()
    # frozen unified configuration (B1 pre-registered protocol, 8 Oct 2026)
    cfg = dict(d=a.d, modes=16, layers=2, n_exp=2, n_lognormal=8, dv=4, dropout=0.3, lr=3e-3, batch=32, state_modes=16)
    torch.set_num_threads(1); torch.set_default_dtype(torch.float64)
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed); rng = random.Random(a.seed)
    train, K = load(a.dataset, a.split, 'train'); val, _ = load(a.dataset, a.split, 'val')
    gaps = np.concatenate([np.diff(t) for t, _ in train]); pos = gaps[gaps > 0]
    cell = recording_cell(pos); comps = n_components(pos); n_window = 4 if comps >= 2 else 0
    scale = float(np.median(pos))
    qs = np.log(np.quantile(pos, np.linspace(0.1, 0.9, cfg['n_lognormal']))).tolist()
    edges = base.cluster_windows(pos, n_window, a.seed) if n_window else None
    model = base.RaceTPP(K + 1, cfg['d'], cfg['modes'], cfg['layers'], cfg['n_exp'], cfg['n_lognormal'], cfg['dv'],
                         cfg['dropout'], scale, qs, cell, n_window, edges, cfg['state_modes'],
                         mark_mem=a.mark_mem, mark_stats=a.mark_stats)
    params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=0.0)
    out_dir = ROOT / 'experiments/results/tpp_b4dev'; out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = out_dir / f'{a.tag}.pt'
    best, best_epoch, history, start = math.inf, -1, [], time.time()
    for epoch in range(a.epochs):
        model.train(); e0 = time.time()
        for t, m, mask in base.batches(train, cfg['batch'], True, rng):
            tl, ml, n, _ = model.loglik(t, m, mask, cell)
            loss = -(tl + ml) / n
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        v = per_sequence(model, val)
        history.append(dict(epoch=epoch, val_L_T=v['L_T'], val_L_M=v['L_M'], val_total=v['total'], epoch_s=time.time() - e0))
        print(json.dumps(history[-1]), flush=True)
        if v['total'] < best:
            best, best_epoch = v['total'], epoch; torch.save(model.state_dict(), ckpt)
        if epoch - best_epoch >= a.patience or time.time() - start > a.max_wall_s:
            break
    model.load_state_dict(torch.load(ckpt))
    res = dict(status='completed', battle='B4-dev', tag=a.tag, dataset=a.dataset, split=a.split, args=vars(a), config=cfg,
               K=K, recording_cell=cell, gap_components=int(comps), n_window=n_window, scale=scale, parameters=params,
               best_epoch=best_epoch, epochs_run=len(history), wall_s=time.time() - start, val=per_sequence(model, val),
               history=history, checkpoint=str(ckpt.relative_to(ROOT)))
    res['source_sha256'] = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                            for p in (Path(__file__).resolve(), ROOT / 'experiments/tpp/race_tpp_v20.py')}
    (out_dir / f'{a.tag}.json').write_text(json.dumps(res, indent=1) + '\n')
    print('VAL', json.dumps(res['val']), flush=True)


if __name__ == '__main__':
    main()
