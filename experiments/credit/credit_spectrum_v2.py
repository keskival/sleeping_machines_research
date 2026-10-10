#!/usr/bin/env python3
"""ENABLER measurement for theory note 165 section 7b: credit spectrum of an ADDRESSED-write temporal memory.

credit_spectrum.py found extensive credit (k(eps) ~ n^0.8) for online_race's dense-write memory, where every event writes
every mode. This driver makes the writes sparse and addressed, the family's mechanism, while keeping online_race.py's
exact per-event trace learner unchanged:
  - the n = K * b modes form K slots of b modes; each event writes exactly ONE slot, chosen by content:
    slot = hash(mark, gap bucket) mod K (deterministic; 8 log-gap buckets);
  - implemented by masking the write gate (g -> g * mask). For a 0/1 mask this is exact for the write, its Jacobian and
    every trace derivative (sigma'(v) * mask = g_m (1 - g_m) when mask = 1, 0 when mask = 0), so Traces.step is unchanged.
Measures on DEV the exact memory adjoint lam_t and, per event, each mode's eligibility norm w_{t,m} = sum over memory
parameters of |S_{t,m}|^2 (forward traces). Reports spectra of (i) unweighted residual credit, (ii) eligibility-weighted
credit lam_{t,m} * sqrt(w_{t,m}) (note 165 Theorem E: coordinates with no eligibility need no credit). Prediction: the
weighted k(eps) is flat in K at fixed k = 1 slot per event. DEV only, no TEST, no selection.
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
_orig_embed, _orig_write = O.embed, O.write
CFG = dict(K=1, b=1, on=False)
_last = {}


def slot_of(m, dt):
    bucket = torch.clamp(torch.floor(torch.log1p(dt) * 2.0), 0, 7).long()
    return (m.long() * 8 + bucket) * 2654435761 % CFG['K']


def embed(p, m, dt):
    if CFG['on']:
        _last['slot'] = slot_of(m, dt)
    return _orig_embed(p, m, dt)


def write(p, u):
    wr, wi, (r_, i_, g) = None, None, (None, None, None)
    br, bi, parts = _orig_write(p, u)
    if not CFG['on']:
        return br, bi, parts
    wr, wi, g = parts
    mask = torch.zeros_like(g)
    slot = _last['slot']
    cols = slot[:, None] * CFG['b'] + torch.arange(CFG['b'])[None]
    mask.scatter_(1, cols, 1.0)
    gm = g * mask
    return wr * gm, wi * gm, (wr, wi, gm)


O.embed, O.write = embed, write


def eligibility_adjoints(p, seqs, a, scale):
    """lam (exact adjoint), c, window-4 prediction and per-mode eligibility norms w on DEV."""
    out = dict(lam=[], c=[], w4=[], w=[])
    for t, m, mask in batches(seqs, 32, False, None):
        t = t.to(DT); B, L = m.shape; n = a.modes
        dt = torch.diff(t, dim=1, prepend=t[:, :1]).clamp_min(0) / scale
        zr = torch.zeros(B, n, dtype=DT); zi = torch.zeros_like(zr); tr = O.Traces(B, n, a.d, a.K, DT)
        C = np.zeros((B, L, n), complex); A = np.zeros((B, L, n), complex); W = np.zeros((B, L, n))
        for i in range(L):
            u, f = O.embed(p, m[:, i], dt[:, i])
            if i > 0:
                valid = mask[:, i]
                zr_l = zr.detach().requires_grad_(True); zi_l = zi.detach().requires_grad_(True)
                up, _ = O.embed(p, m[:, i - 1], dt[:, i - 1])
                tl, ml = O.event_ll(p, up.detach(), zr_l, zi_l, dt[:, i].clamp_min(1e-9), m[:, i], a.n_exp, a.n_ln, a.K)
                gr, gi = torch.autograd.grad(-((tl + ml) * valid).sum(), (zr_l, zi_l))
                C[:, i - 1] = (gr + 1j * gi).numpy()
                u, f = O.embed(p, m[:, i], dt[:, i])                      # re-address event i before its write
            with torch.no_grad():
                br_, bi_, parts = O.write(p, u); ar, ai, r = O.decay(p, dt[:, i])
                tr.step(p, ar, ai, r, dt[:, i], zr, zi, u, f, m[:, i], parts)
                A[:, i] = (ar + 1j * ai).numpy()
                zr, zi = ar * zr - ai * zi + br_, ar * zi + ai * zr + bi_
                wn = torch.zeros(B, n, dtype=DT)
                for Sr, Si in tr.S.values():
                    wn += (Sr ** 2 + Si ** 2).reshape(B, n, -1).sum(-1)
                W[:, i] = wn.numpy()
        Mv = mask.numpy()
        for bb in range(B):
            last = int(Mv[bb].sum()) - 1
            lam = np.zeros((L + 1, n), complex)
            for s in range(last - 1, -1, -1):
                lam[s] = C[bb, s] + np.conj(A[bb, s + 1]) * lam[s + 1]
            for s in range(0, last):
                w4 = np.zeros(n, complex); prod = np.ones(n, complex)
                for j in range(4):
                    if s + j >= last: break
                    if j > 0: prod = prod * np.conj(A[bb, s + j])
                    w4 += prod * C[bb, s + j]
                out['lam'].append(lam[s]); out['c'].append(C[bb, s]); out['w4'].append(w4); out['w'].append(W[bb, s])
    return {k: np.array(v) for k, v in out.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', required=True); ap.add_argument('--dataset', default='taxi')
    ap.add_argument('--slots', type=int, nargs='+', default=[16, 64]); ap.add_argument('--slot-modes', type=int, default=4)
    ap.add_argument('--d', type=int, default=32); ap.add_argument('--n-exp', type=int, default=4); ap.add_argument('--n-ln', type=int, default=8)
    ap.add_argument('--epochs', type=int, default=20); ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--batch', type=int, default=16); ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--max-train-seqs', type=int, default=0); ap.add_argument('--max-dev-seqs', type=int, default=0)
    ap.add_argument('--contract', action='store_true')
    a = ap.parse_args(); torch.set_num_threads(1); t0 = time.time()
    if a.contract:   # masked addressed writes: summed per-event trace gradient == BPTT (online_race.contract, unchanged)
        CFG.update(K=4, b=2, on=True)
        ca = argparse.Namespace(modes=8, n_exp=a.n_exp, n_ln=a.n_ln)
        err = O.contract(ca)
        res = dict(status='completed' if err < 1e-9 else 'failed', tag=a.tag, contract='masked addressed writes (K=4 slots, b=2): trace gradient == BPTT, fixed weights, float64',
                   max_relative_difference=err,
                   source_sha256={str(Path(f).resolve().relative_to(ROOT)): hashlib.sha256(Path(f).read_bytes()).hexdigest()
                                  for f in (__file__, ROOT / 'experiments/credit/credit_spectrum.py', ROOT / 'experiments/credit/online_race.py', ROOT / 'experiments/tpp/race_tpp_v8.py')})
        (ROOT / 'experiments/results/credit' / f'{a.tag}.json').write_text(json.dumps(res, indent=1) + '\n')
        sys.exit(0 if err < 1e-9 else 1)
    tr_s, dev_s = load_split(a.dataset, 'train'), load_split(a.dataset, 'dev')
    if a.max_train_seqs: tr_s = tr_s[:a.max_train_seqs]
    if a.max_dev_seqs: dev_s = dev_s[:a.max_dev_seqs]
    res = dict(status='completed', battle='ENABLER (theory note 165 section 7b: addressed-write credit scaling)', tag=a.tag,
               args=vars(a), by_slots={})
    for K in a.slots:
        CFG.update(K=K, b=a.slot_modes, on=True)
        aa = argparse.Namespace(**{**vars(a), 'modes': K * a.slot_modes})
        p, scale, hist = CS.train(aa, tr_s, dev_s)
        X = eligibility_adjoints(p, dev_s, aa, scale)
        sw = np.sqrt(X['w'])
        res['by_slots'][str(K)] = dict(modes=K * a.slot_modes, dev_ll_history=hist, events=int(len(X['lam'])),
            mean_modes_with_eligibility_above_1pct=float(((X['w'] / X['w'].max(1, keepdims=True).clip(1e-300)) > 0.01).sum(1).mean()),
            unweighted=dict(P0_none=CS.spectrum_stats(X['lam']), P2_window4=CS.spectrum_stats(X['lam'] - X['w4'])),
            weighted=dict(P0_none=CS.spectrum_stats(X['lam'] * sw), P2_window4=CS.spectrum_stats((X['lam'] - X['w4']) * sw)))
        r = res['by_slots'][str(K)]
        print(json.dumps(dict(K=K, unweighted_k01=r['unweighted']['P0_none']['k_eps0.1'], weighted_k01=r['weighted']['P0_none']['k_eps0.1'],
                              weighted_w4_k01=r['weighted']['P2_window4']['k_eps0.1'], dev_ll=hist[-1]['dev_ll'])), flush=True)
    CFG['on'] = False
    res['wall_s'] = time.time() - t0
    res['source_sha256'] = {str(Path(f).resolve().relative_to(ROOT)): hashlib.sha256(Path(f).read_bytes()).hexdigest()
                            for f in (__file__, ROOT / 'experiments/credit/credit_spectrum.py', ROOT / 'experiments/credit/online_race.py',
                                      ROOT / 'experiments/tpp/race_tpp_v8.py')}
    od = ROOT / 'experiments/results/credit'; (od / f'{a.tag}.json').write_text(json.dumps(res, indent=1) + '\n')
    print('RESULT', a.tag, round(res['wall_s']), flush=True)


if __name__ == '__main__':
    main()
