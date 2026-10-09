#!/usr/bin/env python3
"""B4 guarded driver (9 Oct 2026, protocol repair): identical to race_tpp_b4.py except that an update whose loss or
gradient norm is non-finite is skipped instead of applied. In race_tpp_b4.py one such batch makes clip_grad_norm_ scale
every gradient by NaN and AdamW writes NaN into all parameters; Github splits 1-4 and Wikipedia split 0 ended that way
(selected checkpoints from epochs 2-5). Skipped updates are counted and the first ones are described (which term is
non-finite, batch length, smallest positive gap, zero gaps) for diagnosis. Configuration, data rules, selection and
scoring are unchanged; with no non-finite update the run is identical to race_tpp_b4.py (--contract checks this bitwise).

B4: the frozen unified race-of-clocks model (race_tpp_v19, unchanged) on the neural-TPP benchmark of Bosser & Ben
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
import race_tpp_v19 as base  # noqa: E402

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


def train_epoch(model, opt, train, cfg, rng, cell, skips=None, epoch=0):
    """One training pass. skips=None: the original race_tpp_b4.py update; else non-finite updates are skipped and logged."""
    model.train()
    for bi, (t, m, mask) in enumerate(base.batches(train, cfg['batch'], True, rng)):
        tl, ml, n, _ = model.loglik(t, m, mask, cell)
        loss = -(tl + ml) / n
        opt.zero_grad(); loss.backward(); gn = nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if skips is None or (torch.isfinite(loss) and torch.isfinite(gn)):
            opt.step()
        else:
            g = np.diff(t.numpy(), axis=1)[mask[:, 1:].numpy()]
            skips.append(dict(epoch=epoch, batch=bi, time_ll_finite=bool(torch.isfinite(tl)), mark_ll_finite=bool(torch.isfinite(ml)),
                              grad_norm=float(gn), length=int(mask.shape[1]), events=int(mask.sum()),
                              min_positive_gap=float(g[g > 0].min()) if (g > 0).any() else None, zero_gaps=int((g == 0).sum())))


def contract(a):
    """Guarded and original updates are bitwise identical when every update is finite (2 epochs, MIMIC2 split 0)."""
    cfg = dict(d=32, modes=16, layers=2, n_exp=2, n_lognormal=8, dv=4, dropout=0.3, lr=3e-3, batch=32, state_modes=16)
    states = []
    for guarded in (False, True):
        random.seed(0); np.random.seed(0); torch.manual_seed(0); rng = random.Random(0)
        train, K = load('mimic2_filtered', 0, 'train')
        gaps = np.concatenate([np.diff(t) for t, _ in train]); pos = gaps[gaps > 0]
        cell = recording_cell(pos); comps = n_components(pos); n_window = 4 if comps >= 2 else 0
        qs = np.log(np.quantile(pos, np.linspace(0.1, 0.9, cfg['n_lognormal']))).tolist()
        edges = base.cluster_windows(pos, n_window, 0) if n_window else None
        model = base.RaceTPP(K + 1, cfg['d'], cfg['modes'], cfg['layers'], cfg['n_exp'], cfg['n_lognormal'], cfg['dv'],
                             cfg['dropout'], float(np.median(pos)), qs, cell, n_window, edges, cfg['state_modes'])
        opt = torch.optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=0.0); sk = [] if guarded else None
        for ep in range(2):
            train_epoch(model, opt, train, cfg, rng, cell, sk, ep)
        states.append({k: v.clone() for k, v in model.state_dict().items()})
    diff = max((states[0][k] - states[1][k]).abs().max().item() for k in states[0])
    print(f'contract: guarded vs original after 2 epochs, max parameter difference {diff:.1e}', 'PASS' if diff == 0 else 'FAIL')
    return diff


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True)
    ap.add_argument('--split', type=int, required=True)
    ap.add_argument('--tag', required=True)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--epochs', type=int, default=200)
    ap.add_argument('--patience', type=int, default=30)
    ap.add_argument('--max-wall-s', type=float, default=6 * 3600)
    ap.add_argument('--score-test', action='store_true'); ap.add_argument('--contract', action='store_true')
    a = ap.parse_args()
    # frozen unified configuration (B1 pre-registered protocol, 8 Oct 2026)
    cfg = dict(d=32, modes=16, layers=2, n_exp=2, n_lognormal=8, dv=4, dropout=0.3, lr=3e-3, batch=32, state_modes=16)
    torch.set_num_threads(1); torch.set_default_dtype(torch.float64)
    if a.contract:
        diff = contract(a); od = ROOT / 'experiments/results/tpp_b4'
        (od / f'{a.tag}.json').write_text(json.dumps(dict(status='completed' if diff == 0 else 'failed', tag=a.tag, max_difference=diff,
            contract='guarded update == race_tpp_b4.py update when all updates are finite (2 epochs, MIMIC2 split 0)',
            source_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                           for p in (Path(__file__).resolve(), ROOT / 'experiments/tpp/race_tpp_v19.py')}), indent=1) + '\n')
        return
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed); rng = random.Random(a.seed)
    train, K = load(a.dataset, a.split, 'train'); val, _ = load(a.dataset, a.split, 'val')
    gaps = np.concatenate([np.diff(t) for t, _ in train]); pos = gaps[gaps > 0]
    cell = recording_cell(pos); comps = n_components(pos); n_window = 4 if comps >= 2 else 0
    scale = float(np.median(pos))
    qs = np.log(np.quantile(pos, np.linspace(0.1, 0.9, cfg['n_lognormal']))).tolist()
    edges = base.cluster_windows(pos, n_window, a.seed) if n_window else None
    model = base.RaceTPP(K + 1, cfg['d'], cfg['modes'], cfg['layers'], cfg['n_exp'], cfg['n_lognormal'], cfg['dv'],
                         cfg['dropout'], scale, qs, cell, n_window, edges, cfg['state_modes'])
    params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=0.0)
    out_dir = ROOT / 'experiments/results/tpp_b4'; out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = out_dir / f'{a.tag}.pt'
    best, best_epoch, history, start = math.inf, -1, [], time.time(); skips = []
    for epoch in range(a.epochs):
        model.train(); e0 = time.time()
        train_epoch(model, opt, train, cfg, rng, cell, skips, epoch)
        v = per_sequence(model, val)
        history.append(dict(epoch=epoch, val_L_T=v['L_T'], val_L_M=v['L_M'], val_total=v['total'], epoch_s=time.time() - e0))
        print(json.dumps(history[-1]), flush=True)
        if v['total'] < best:
            best, best_epoch = v['total'], epoch; torch.save(model.state_dict(), ckpt)
        if epoch - best_epoch >= a.patience or time.time() - start > a.max_wall_s:
            break
    model.load_state_dict(torch.load(ckpt))
    res = dict(status='completed', battle='B4', tag=a.tag, dataset=a.dataset, split=a.split, args=vars(a), config=cfg,
               K=K, recording_cell=cell, gap_components=int(comps), n_window=n_window, scale=scale, parameters=params,
               best_epoch=best_epoch, epochs_run=len(history), skipped_updates=len(skips), skipped_first=skips[:20], wall_s=time.time() - start, val=per_sequence(model, val),
               history=history, checkpoint=str(ckpt.relative_to(ROOT)))
    if a.score_test:
        res['test'] = per_sequence(model, load(a.dataset, a.split, 'test')[0])
    res['source_sha256'] = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                            for p in (Path(__file__).resolve(), ROOT / 'experiments/tpp/race_tpp_v19.py')}
    (out_dir / f'{a.tag}.json').write_text(json.dumps(res, indent=1) + '\n')
    print('VAL', json.dumps(res['val']), flush=True)
    if a.score_test:
        print('TEST', json.dumps(res['test']), flush=True)


if __name__ == '__main__':
    main()
