"""B3 FAS native member: learned delay-race binding with addressed per-step evidence.

Mechanism (temporal computation, races and addressed memory, no item identities):
1. Delay clocks, learned from anonymous clean TRAIN logs. For every ordered event-type pair (j -> y), the log-lag
   histogram between a y event and the preceding j events (window W) is accumulated; a sharp mode marks a process step
   (1% timing noise). Each type y keeps its strongest P predecessor clocks (j, mu, sigma) by mode mass.
2. Binding by temporal race. A y event at time t races the clocks started by earlier candidate events of its
   predecessor types; the candidate whose age best matches its clock's delay (smallest |z|) wins. With consume=1 a
   predecessor event can bind at most one successor of each pair (anonymous at-most-once assignment).
3. Addressed evidence. The residual z of each bound step is standardized per step (pair) on a separate clean
   calibration draw and accumulated in that step's own CUSUM slot; the run score is the most anomalous slot.
Weights/clocks are fixed during scoring: a fault is growth of a step's delay within a run, which adaptation would absorb.
"""
import argparse, hashlib, json, sys, time
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT/'experiments/fas'))
from native import load
from sklearn.metrics import roc_auc_score

DATA = ROOT/'experiments/data/fas'
MAIN = 'fas_v2_K2_drop0.02_delta0_20261005'
CALIB = 'fas_v2_K2_drop0.02_delta0_calib_520261015'
K = 64; BINS = np.linspace(-3, 8, 441); CENTRE = 0.5*(BINS[1:] + BINS[:-1])
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()


def learn_clocks(runs, window=64, top=2, min_mass=5.):
    H = np.zeros((K, K, len(BINS) - 1))
    for ids, t in runs:
        for i in range(1, len(ids)):
            lo = max(0, i - window); a = np.log(np.maximum(t[i] - t[lo:i], 1e-3))
            np.add.at(H, (ids[lo:i], np.full(i - lo, ids[i]), np.clip(np.digitize(a, BINS) - 1, 0, len(BINS) - 2)), 1)
    clocks = {}
    for y in range(K):
        cands = []
        for j in range(K):
            h = H[j, y]
            if h.sum() < 200: continue
            k = h.argmax(); sl = slice(max(0, k - 3), k + 4); mass = h[sl].sum()/len(runs)
            if mass < min_mass: continue
            w = h[sl]; mu = (w*CENTRE[sl]).sum()/w.sum(); sd = max(np.sqrt((w*(CENTRE[sl] - mu)**2).sum()/w.sum()), 0.01)
            cands.append((float(mass), int(j), float(mu), float(sd)))
        cands.sort(reverse=True)
        if cands: clocks[y] = cands[:top]
    return clocks


def residuals(ids, t, clocks, N, window=64, consume=False):
    out = []; used = set()
    for i in range(1, min(len(ids), N)):
        y = int(ids[i])
        if y not in clocks: continue
        lo = max(0, i - window); best = None
        for mass, j, mu, sd in clocks[y]:
            ks = np.where(ids[lo:i] == j)[0] + lo
            if consume: ks = np.array([k for k in ks if (k, j, y) not in used], dtype=int)
            if len(ks) == 0: continue
            z = (np.log(np.maximum(t[i] - t[ks], 1e-3)) - mu)/sd; b = int(np.argmin(np.abs(z)))
            if best is None or abs(z[b]) < abs(best[1]): best = (j*K + y, float(z[b]), int(ks[b]), j)
        if best:
            out.append(best[:2])
            if consume: used.add((best[2], best[3], y))
    return out


def calibrate(runs, clocks, window, consume, clip):
    acc = {}
    for ids, t in runs:
        for key, z in residuals(ids, t, clocks, 10**9, window, consume): acc.setdefault(key, []).append(min(z, clip))
    return {k: (float(np.mean(v)), float(np.std(v)) + 1e-3) for k, v in acc.items() if len(v) > 30}


def score(runs, clocks, cal, N, kappa, window, consume, clip, two_sided=False):
    S = []
    for ids, t in runs:
        cus = {}; peak = 0.
        for key, z in residuals(ids, t, clocks, N, window, consume):
            m, s = cal.get(key, (0., 1.)); z = (min(z, clip) - m)/s; z = abs(z) if two_sided else z
            c = max(0., cus.get(key, 0.) + z - kappa); cus[key] = c; peak = max(peak, c)
        S.append(peak)
    return np.array(S)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--tag', required=True); ap.add_argument('--train-runs', type=str, default='0:1000', help='TRAIN slice start:stop for clock learning')
    ap.add_argument('--top', type=int, default=2); ap.add_argument('--window', type=int, default=64); ap.add_argument('--consume', type=int, default=0)
    ap.add_argument('--kappas', default='0.5,1,2'); ap.add_argument('--clip', type=float, default=8.); ap.add_argument('--two-sided', type=int, default=0)
    ap.add_argument('--calib-runs', type=int, default=1000); ap.add_argument('--eval-runs', type=int, default=1000); ap.add_argument('--prefixes', default='256,512,1024')
    a = ap.parse_args(); began = time.time(); out = ROOT/f'experiments/results/fas/{a.tag}.json'; assert not out.exists()
    s0, s1 = (int(x) for x in a.train_runs.split(':'))
    train, _ = load(DATA/MAIN/'train_clean.npz', 1100); train = train[s0:s1]
    clocks = learn_clocks(train, a.window, a.top); del train
    cal_runs, _ = load(DATA/CALIB/'val_clean.npz', 1100); cal = calibrate(cal_runs[:a.calib_runs], clocks, a.window, bool(a.consume), a.clip)
    vc, _ = load(DATA/MAIN/'val_clean.npz', 1100); vf, vk = load(DATA/MAIN/'val_faulty.npz', 1100)
    vc, vf, vk = vc[:a.eval_runs], vf[:a.eval_runs], np.asarray(vk[:a.eval_runs]); res = {}
    for N in (int(x) for x in a.prefixes.split(',')):
        for kappa in (float(x) for x in a.kappas.split(',')):
            sc = score(vc, clocks, cal, N, kappa, a.window, bool(a.consume), a.clip, bool(a.two_sided)); sf = score(vf, clocks, cal, N, kappa, a.window, bool(a.consume), a.clip, bool(a.two_sided))
            y = np.r_[np.zeros(len(sc)), np.ones(len(sf))]; row = dict(all=float(roc_auc_score(y, np.r_[sc, sf])))
            for k in np.unique(vk): row[str(int(k))] = float(roc_auc_score(np.r_[np.zeros(len(sc)), np.ones((vk == k).sum())], np.r_[sc, sf[vk == k]]))
            res.setdefault(str(N), {})[f'kappa{kappa:g}'] = row; print(json.dumps(dict(N=N, kappa=kappa, **row)), flush=True)
    record = dict(status='completed', battle='B3', tag=a.tag, args=vars(a), clocks={str(k): v for k, v in clocks.items()}, n_clock_types=len(clocks),
                  calibration_steps=len(cal), val_auroc=res, wall_s=time.time() - began, test_opened=False, identities_used=False,
                  data=dict(main=MAIN, calibration=CALIB), source_sha256={n: sha(ROOT/n) for n in ['experiments/fas/race_delay_binding.py', 'experiments/fas/native.py']},
                  scope='Validation-only development. Delay clocks from anonymous clean TRAIN logs; per-step standardization from a separate clean calibration draw; fault labels used only for validation AUROC.')
    out.write_text(json.dumps(record, indent=1) + '\n')


if __name__ == '__main__':
    main()
