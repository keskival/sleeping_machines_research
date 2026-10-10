#!/usr/bin/env python3
"""ENABLER measurement for theory note 165 (credit-rate scaling): the residual credit spectrum of the per-event
temporal-memory learner (online_race.py, unchanged) after forward prediction, at two memory sizes.

Trains the `online_trace` learner exactly as online_race.py does (same init, data order, optimizer), then on Taxi DEV
computes for every event the exact memory adjoint lam_t = c_t + conj(a_{t+1}) lam_{t+1} (c_t = d loss_{t+1} / d z_t)
and the residual after three forward predictors:
  P0 none (lam itself); P1 shared transport, arrived outcome only (lam - c_t); P2 shared transport over a 4-event window.
Reports eigenvalues of each residual's covariance (real 2n coordinates), the power-law exponent alpha (note 165 Theorem
2), and the reverse water-filling component count k(eps) and rate R(eps) for eps = 0.1, 0.01. Size independence
predicts k(eps) flat in n when alpha > 1. DEV only, no TEST, no selection.
"""
import argparse, hashlib, json, math, sys, time
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'experiments/credit')); sys.path.insert(0, str(ROOT / 'experiments/tpp'))
import online_race as O  # noqa: E402
from race_tpp_v8 import load_split, batches  # noqa: E402

DT = torch.float64


def train(a, tr_s, dev_s):
    torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed)
    a.K = int(max(max(s[1]) for s in tr_s + dev_s)) + 1
    gaps = np.concatenate([np.diff(s[0]) for s in tr_s]); pos = gaps[gaps > 0]; scale = float(np.median(pos))
    log_q = np.log(np.quantile(pos / scale, np.linspace(0.1, 0.9, a.n_ln)))
    p = O.init_params(a.K, a.d, a.modes, a.n_exp, a.n_ln, log_q, DT, torch.Generator().manual_seed(a.seed))
    opt = torch.optim.Adam(p.values(), lr=a.lr); tr = O.Traces(a.batch, a.modes, a.d, a.K, DT); hist = []
    for ep in range(a.epochs):
        for t, m, mask in batches(tr_s, a.batch, True, rng):
            tr_b = tr if len(m) == a.batch else O.Traces(len(m), a.modes, a.d, a.K, DT)
            O.run_batch(p, t.to(DT), m, mask, a, scale, 'online_trace', opt=opt, tr=tr_b)
        tll, mll, ll = O.evaluate(p, dev_s, a, scale); hist.append(dict(epoch=ep, dev_ll=ll)); print(json.dumps(hist[-1]), flush=True)
    return p, scale, hist


def adjoints(p, seqs, a, scale, max_seqs):
    """Exact per-event memory adjoints on DEV. Returns lists of complex arrays [n] for lam, c and window-4 prediction."""
    out = dict(lam=[], c=[], w4=[])
    seqs = seqs[:max_seqs] if max_seqs else seqs
    for t, m, mask in batches(seqs, 32, False, None):
        t = t.to(DT); B, L = m.shape; n = a.modes
        dt = torch.diff(t, dim=1, prepend=t[:, :1]).clamp_min(0) / scale
        zr = torch.zeros(B, n, dtype=DT); zi = torch.zeros_like(zr)
        C = np.zeros((B, L, n), complex); A = np.zeros((B, L, n), complex)
        for i in range(L):
            u, f = O.embed(p, m[:, i], dt[:, i])
            if i > 0:
                valid = mask[:, i]
                zr_l = zr.detach().requires_grad_(True); zi_l = zi.detach().requires_grad_(True)
                up, _ = O.embed(p, m[:, i - 1], dt[:, i - 1])
                tl, ml = O.event_ll(p, up.detach(), zr_l, zi_l, dt[:, i].clamp_min(1e-9), m[:, i], a.n_exp, a.n_ln, a.K)
                gr, gi = torch.autograd.grad(-((tl + ml) * valid).sum(), (zr_l, zi_l))
                C[:, i - 1] = (gr + 1j * gi).numpy()                      # c_{i-1} = d loss_i / d z_{i-1}
            with torch.no_grad():
                br_, bi_, _ = O.write(p, u); ar, ai, _ = O.decay(p, dt[:, i])
                A[:, i] = (ar + 1j * ai).numpy()
                zr, zi = ar * zr - ai * zi + br_, ar * zi + ai * zr + bi_
        Mv = mask.numpy()
        for b in range(B):
            last = int(Mv[b].sum()) - 1                                     # index of the last event
            lam = np.zeros((L + 1, n), complex)
            for s in range(last - 1, -1, -1):                               # states with at least one future event
                lam[s] = C[b, s] + np.conj(A[b, s + 1]) * lam[s + 1]
            for s in range(0, last):
                w = np.zeros(n, complex); prod = np.ones(n, complex)
                for j in range(4):
                    if s + j >= last: break
                    if j > 0: prod = prod * np.conj(A[b, s + j])
                    w += prod * C[b, s + j]
                out['lam'].append(lam[s]); out['c'].append(C[b, s]); out['w4'].append(w)
    return {k: np.array(v) for k, v in out.items()}


def spectrum_stats(X):
    R = np.concatenate([X.real, X.imag], 1)                               # real 2n coordinates
    R = R - R.mean(0, keepdims=True)
    ev = np.sort(np.linalg.eigvalsh(R.T @ R / len(R)))[::-1].clip(min=0)
    tot = ev.sum(); keep = ev > tot * 1e-12
    i = np.arange(1, keep.sum() + 1); half = max(3, len(i) // 2)
    alpha = float(-np.polyfit(np.log(i[:half]), np.log(ev[:half]), 1)[0])
    alpha_all = float(-np.polyfit(np.log(i), np.log(ev[keep]), 1)[0])
    res = dict(eigenvalues=ev.tolist(), total=float(tot), alpha_top_half=alpha, alpha_all=alpha_all,
               participation_ratio=float(tot ** 2 / (ev ** 2).sum()))
    for eps in (0.1, 0.01):                                               # reverse water-filling at relative distortion eps
        lo, hi = 0.0, float(ev[0])
        for _ in range(200):
            th = 0.5 * (lo + hi); D = np.minimum(ev, th).sum() / tot
            lo, hi = (th, hi) if D < eps else (lo, th)
        th = 0.5 * (lo + hi); act = ev > th
        res[f'k_eps{eps}'] = int(act.sum()); res[f'rate_nats_eps{eps}'] = float(0.5 * np.log(ev[act] / th).sum())
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', required=True); ap.add_argument('--dataset', default='taxi')
    ap.add_argument('--modes', type=int, nargs='+', default=[16, 64])
    ap.add_argument('--d', type=int, default=32); ap.add_argument('--n-exp', type=int, default=4); ap.add_argument('--n-ln', type=int, default=8)
    ap.add_argument('--epochs', type=int, default=20); ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--batch', type=int, default=16); ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--max-dev-seqs', type=int, default=0); ap.add_argument('--max-train-seqs', type=int, default=0)
    a = ap.parse_args(); torch.set_num_threads(1); t0 = time.time()
    tr_s, dev_s = load_split(a.dataset, 'train'), load_split(a.dataset, 'dev')
    if a.max_train_seqs: tr_s = tr_s[:a.max_train_seqs]
    res = dict(status='completed', battle='ENABLER (theory note 165: credit-rate scaling; DEEP_LEARNING_SCALING)', tag=a.tag,
               args=vars(a), by_modes={})
    for n in a.modes:
        a.modes_cur = n; aa = argparse.Namespace(**{**vars(a), 'modes': n})
        p, scale, hist = train(aa, tr_s, dev_s)
        X = adjoints(p, dev_s, aa, scale, a.max_dev_seqs)
        res['by_modes'][str(n)] = dict(dev_ll_history=hist, events=int(len(X['lam'])),
                                      P0_none=spectrum_stats(X['lam']), P1_arrived_outcome=spectrum_stats(X['lam'] - X['c']),
                                      P2_window4=spectrum_stats(X['lam'] - X['w4']))
        print(json.dumps({k: {kk: v[kk] for kk in ('alpha_top_half', 'participation_ratio', 'k_eps0.1', 'k_eps0.01')}
                          for k, v in res['by_modes'][str(n)].items() if k.startswith('P')}), flush=True)
    res['wall_s'] = time.time() - t0
    res['source_sha256'] = {str(Path(f).resolve().relative_to(ROOT)): hashlib.sha256(Path(f).read_bytes()).hexdigest()
                            for f in (__file__, ROOT / 'experiments/credit/online_race.py', ROOT / 'experiments/tpp/race_tpp_v8.py')}
    od = ROOT / 'experiments/results/credit'; od.mkdir(parents=True, exist_ok=True)
    (od / f'{a.tag}.json').write_text(json.dumps(res, indent=1) + '\n')
    print('RESULT', a.tag, round(res['wall_s']), flush=True)


if __name__ == '__main__':
    main()
