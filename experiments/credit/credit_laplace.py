#!/usr/bin/env python3
"""ENABLER measurement for theory note 167: does the forward race's predicted gap law predict the credit's magnitude?

Retrains online_race's `online_trace` learner exactly as credit_spectrum.py (same recipe; reproduces online1 s0). On Taxi DEV, per
event t and memory mode m:
  - realized credit tail  T_{t,m} = lam_{t,m} - c_{t,m}  (exact adjoint minus arrived cotangent),
  - Laplace prediction    V_hat_{t,m} = sigma_m^2 * L_t(2 r_m) / (1 - L_t(2 r_m)),
    with L_t(s) = int f_t(tau) e^{-s tau} dtau from the forward head's own next-gap density f_t (exp(time log-likelihood) on a
    48-point log grid, trapezoid rule) and sigma_m^2 the TRAIN-free running DEV mean of |c_m|^2 up to t (causal).
Reports per-mode Spearman rank correlation of V_hat with |T|^2, global calibration slope, and the audit cost (note 165 Theorem
10.1) of allocations p ~ sqrt(V_hat) and p ~ |T| (oracle) relative to uniform at equal estimator variance. DEV only.
"""
import argparse, hashlib, json, math, sys, time
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'experiments/credit')); sys.path.insert(0, str(ROOT / 'experiments/tpp'))
import online_race as O  # noqa: E402
import credit_spectrum as CS  # noqa: E402
from race_tpp_v8 import load_split, batches  # noqa: E402

DT = torch.float64


def spearman(a, b):
    ra = np.argsort(np.argsort(a)); rb = np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def audit_cost_ratio(r_abs, weight):
    """Cost of p ~ weight at the variance of uniform p = mean(p), relative to uniform (Theorem 10.1 form, no saturation)."""
    w = np.maximum(weight, 1e-300); r2 = r_abs ** 2
    # uniform p0: variance V0 = sum r2 / p0. With p = a w, variance = sum r2/(a w) = V0 => a = sum(r2/w)/sum(r2) * p0;
    # cost ratio = sum(a w)/(N p0) = sum(r2/w) * sum(w) / (N * sum(r2))
    return float((r2 / w).sum() * w.sum() / (len(w) * r2.sum()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', required=True); ap.add_argument('--dataset', default='taxi')
    ap.add_argument('--modes', type=int, default=16); ap.add_argument('--d', type=int, default=32)
    ap.add_argument('--n-exp', type=int, default=4); ap.add_argument('--n-ln', type=int, default=8)
    ap.add_argument('--epochs', type=int, default=20); ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--batch', type=int, default=16); ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--max-train-seqs', type=int, default=0); ap.add_argument('--max-dev-seqs', type=int, default=0)
    ap.add_argument('--grid', type=int, default=48)
    a = ap.parse_args(); torch.set_num_threads(1); t0 = time.time()
    tr_s, dev_s = load_split(a.dataset, 'train'), load_split(a.dataset, 'dev')
    if a.max_train_seqs: tr_s = tr_s[:a.max_train_seqs]
    if a.max_dev_seqs: dev_s = dev_s[:a.max_dev_seqs]
    p, scale, hist = CS.train(a, tr_s, dev_s)
    rates = p['log_rate'].detach().exp().numpy()                        # r_m in scaled time units
    tau = torch.from_numpy(np.geomspace(1e-4, 1e3, a.grid))             # scaled gap grid
    T2, Vh, M = [], [], []
    sig_sum = np.zeros(a.modes); sig_n = 0
    with torch.no_grad():
        for t, m, mask in batches(dev_s, 32, False, None):
            t = t.to(DT); B, L = m.shape; n = a.modes
            dt = torch.diff(t, dim=1, prepend=t[:, :1]).clamp_min(0) / scale
            zr = torch.zeros(B, n, dtype=DT); zi = torch.zeros_like(zr)
            C = np.zeros((B, L, n), complex); A = np.zeros((B, L, n), complex); Lap = np.zeros((B, L, n))
            for i in range(L):
                u, f = O.embed(p, m[:, i], dt[:, i])
                if i > 0:
                    valid = mask[:, i]
                    with torch.enable_grad():
                        zr_l = zr_prev.detach().requires_grad_(True); zi_l = zi_prev.detach().requires_grad_(True)
                        tl, ml = O.event_ll(p, u_prev, zr_l, zi_l, dt[:, i].clamp_min(1e-9), m[:, i], a.n_exp, a.n_ln, a.K)
                        gr, gi = torch.autograd.grad(-((tl + ml) * valid).sum(), (zr_l, zi_l))
                    C[:, i - 1] = (gr + 1j * gi).numpy()
                br_, bi_, _ = O.write(p, u); ar, ai, _ = O.decay(p, dt[:, i])
                A[:, i] = (ar + 1j * ai).numpy()
                zr_prev, zi_prev, u_prev = (ar * zr - ai * zi + br_), (ar * zi + ai * zr + bi_), u
                zr, zi = zr_prev, zi_prev
                # Laplace transform of the predicted next-gap density, from the state AFTER event i (as run_batch scores)
                tl_grid = []
                for g in tau:
                    tl, _ = O.event_ll(p, u, zr, zi, g.expand(B).clone(), m[:, i], a.n_exp, a.n_ln, a.K)
                    tl_grid.append(tl)
                dens = torch.stack(tl_grid, 1).exp().numpy()                # [B, G] predicted gap density
                kern = np.exp(-2 * rates[None, :, None] * tau.numpy()[None, None, :])
                Lap[:, i] = np.trapezoid(dens[:, None, :] * kern, tau.numpy(), axis=-1)
            Mv = mask.numpy()
            for bb in range(B):
                last = int(Mv[bb].sum()) - 1
                lam = np.zeros((L + 1, n), complex)
                for s in range(last - 1, -1, -1):
                    lam[s] = C[bb, s] + np.conj(A[bb, s + 1]) * lam[s + 1]
                for s in range(0, last):
                    sig = sig_sum / max(sig_n, 1)                           # causal running mean of |c|^2 (DEV so far)
                    Lm = np.clip(Lap[bb, s], 0, 1 - 1e-9)
                    if sig_n:
                        T2.append(np.abs(lam[s] - C[bb, s]) ** 2); Vh.append(sig * Lm / (1 - Lm)); M.append(Lm)
                    sig_sum += np.abs(C[bb, s]) ** 2; sig_n += 1
    T2 = np.array(T2); Vh = np.array(Vh); Lm = np.array(M)
    per_mode = [dict(rate=float(rates[k]), spearman=spearman(Vh[:, k], T2[:, k]),
                     mean_pred=float(Vh[:, k].mean()), mean_real=float(T2[:, k].mean())) for k in range(a.modes)]
    flat_r = np.sqrt(T2.ravel()); flat_v = Vh.ravel()
    res = dict(status='completed', battle='ENABLER (theory note 167: Laplace credit-magnitude prior)', tag=a.tag, args=vars(a),
               dev_ll_history=hist, events=int(len(T2)), per_mode=per_mode,
               spearman_all=spearman(flat_v, flat_r ** 2),
               calibration_slope_log=float(np.polyfit(np.log(flat_v + 1e-30), np.log(flat_r ** 2 + 1e-30), 1)[0]),
               audit_cost_vs_uniform=dict(laplace=audit_cost_ratio(flat_r, np.sqrt(flat_v)),
                                          per_mode_constant=audit_cost_ratio(flat_r, np.sqrt(np.tile(T2.mean(0), len(T2)))),
                                          oracle=audit_cost_ratio(flat_r, flat_r)),
               scope='DEV only; renewal approximation; audit costs by the Theorem 10.1 identity without saturation',
               wall_s=time.time() - t0)
    res['source_sha256'] = {str(Path(f).resolve().relative_to(ROOT)): hashlib.sha256(Path(f).read_bytes()).hexdigest()
                            for f in (__file__, ROOT / 'experiments/credit/credit_spectrum.py', ROOT / 'experiments/credit/online_race.py',
                                      ROOT / 'experiments/tpp/race_tpp_v8.py')}
    (ROOT / 'experiments/results/credit' / f'{a.tag}.json').write_text(json.dumps(res, indent=1) + '\n')
    print(json.dumps({k: res[k] for k in ('spearman_all', 'calibration_slope_log', 'audit_cost_vs_uniform')}), flush=True)


if __name__ == '__main__':
    main()
