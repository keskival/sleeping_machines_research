"""B3 FAS: addressed per-event-type evidence detector, applicable to every learned family.

A FAS fault lives on one module: its step durations grow with use on one interleaved line. A run-wide mean of
per-event surprise dilutes that evidence over ~46 event types and both lines. The addressed detector keeps one evidence
accumulator per next-event type (the address), standardizes each event's surprise by that type's clean mean and spread
measured on a separate clean calibration draw, and reports the most anomalous address. Weights are fixed during scoring:
adapting them inside a run would learn the fault as normal.

Statistics (finite declared family, identical for every model; prefix = first N merged process events):
  total            mean surprise over the prefix (the sealed v2 primary rule, reproduced)
  cusum_k{0,.5,1}  max over positions and types of a per-type CUSUM of standardized surprise with drift k
  excess           max over types of (sum of standardized surprise)/sqrt(count), types with >= 4 events
Each is computed on total surprise (type + gap) and on gap surprise alone.
"""
import argparse, hashlib, json, math, sys, time
from pathlib import Path
import numpy as np
import torch
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT/'experiments/fas')); sys.path.insert(0, str(ROOT/'experiments/tpp'))
from native import load
from sklearn.metrics import roc_auc_score

DATA = ROOT/'experiments/data/fas'
MAIN = 'fas_v2_K2_drop0.02_delta0_20261005'
CALIB = 'fas_v2_K2_drop0.02_delta0_calib_520261015'
PREFIXES = (256, 512, 1024)
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()


def per_event_native(result_json, runs, batch=8):
    import race_tpp_fas_v3 as R
    res = json.loads(Path(result_json).read_text()); a = argparse.Namespace(**res['args'])
    train, _ = load(DATA/a.data/'train_clean.npz', a.train_max_events); train = train[:a.fit_runs]
    gaps = np.concatenate([np.diff(ts) for _, ts in train[:500]]); pos = gaps[gaps > 0]; del train
    scale = float(np.median(pos)); qs = np.log(np.quantile(pos, np.linspace(0.1, 0.9, a.n_lognormal))).tolist()
    edges = R.cluster_windows(pos, a.n_window, a.seed) if a.n_window else None
    margs = (46, a.d, a.modes, a.layers, a.n_exp, a.n_lognormal, a.dv, a.dropout, scale, qs, a.cell_s, a.n_window, edges, 0)
    model = (R.KeyedRaceFAS(*margs, dk=a.dk, local=a.keyed == 2, pred_window=a.pred_window, keyed=bool(a.keyed)) if (a.keyed or a.pred_window) else R.RaceTPP(*margs))
    model.load_state_dict(torch.load(ROOT/res['checkpoint'])); model.eval()
    PT, PG, nll = R.per_position(model, runs, batch)
    return PT, PG, nll


def per_event_dense(result_json, runs, lanes=16):
    import dense as Dn
    res = json.loads(Path(result_json).read_text()); args = res['args'] if 'args' in res else res
    kind, d, layers = res.get('model', args.get('model')), args.get('d', 128), args.get('layers', 2)
    model = Dn.build(kind, d, layers); w = Path(result_json).parent/res['selected_weights']['path']; model.load_state_dict(torch.load(w)); model.eval()
    PT, PG = [], []; tot = 0.; cnt = 0
    with torch.no_grad():
        for b in range(0, len(runs), lanes):
            idx = list(range(b, min(b + lanes, len(runs)))); x, lg, tabs, dt, y, g, mask = Dn.tensors(runs, idx)
            pt, pg = Dn.nll(model(x, lg, tabs, dt), y, g, mask, parts=True); pt, pg = pt.numpy(), pg.numpy(); m = mask.numpy()
            pt[~m] = 0; pg[~m] = 0; tot += float((pt + pg)[m].sum()); cnt += int(m.sum())
            for r in range(len(idx)): PT.append(pt[r]); PG.append(pg[r])
    L = max(len(x) for x in PT); A = np.zeros((len(PT), L)); B = np.zeros((len(PT), L))
    for r in range(len(PT)): A[r, :len(PT[r])] = PT[r]; B[r, :len(PG[r])] = PG[r]
    return A, B, tot/max(cnt, 1)


def targets(runs, L):
    Y = np.full((len(runs), L), -1, dtype=np.int64)
    for r, (ids, _) in enumerate(runs):
        n = min(len(ids) - 1, L); Y[r, :n] = ids[1:n + 1]
    return Y


def calibrate(S, Y, n_types=64):
    mu = np.zeros(n_types); sd = np.ones(n_types)
    for k in range(n_types):
        v = S[Y == k]
        if len(v) > 20: mu[k] = v.mean(); sd[k] = v.std() + 1e-6
    return mu, sd


def statistics(S, Y, mu, sd, N):
    S, Y = S[:, :N], Y[:, :N]; valid = Y >= 0; out = {}
    out['total'] = np.where(valid, S, 0).sum(1)/np.maximum(valid.sum(1), 1)
    Z = np.where(valid, (S - mu[np.clip(Y, 0, None)])/sd[np.clip(Y, 0, None)], 0.)
    types = np.unique(Y[valid])
    for kappa in (0., .5, 1.):
        best = np.zeros(len(S))
        for k in types:
            m = (Y == k); c = np.zeros(len(S)); peak = np.zeros(len(S))
            for i in range(S.shape[1]):
                c = np.where(m[:, i], np.maximum(0., c + Z[:, i] - kappa), c); peak = np.maximum(peak, c)
            best = np.maximum(best, peak)
        out[f'cusum_k{kappa:g}'] = best
    ex = np.full(len(S), -np.inf)
    for k in types:
        m = (Y == k); n = m.sum(1); s = np.where(m, Z, 0).sum(1)
        ex = np.maximum(ex, np.where(n >= 4, s/np.sqrt(np.maximum(n, 1)), -np.inf))
    out['excess'] = ex
    return out


def detector_aurocs(clean, faulty, calib, kinds):
    """clean/faulty/calib: (PT, PG, Y). Returns AUROC per surprise channel, statistic and prefix."""
    res = {}
    for channel in ('total', 'gap'):
        pick = (lambda T, G: T + G) if channel == 'total' else (lambda T, G: G)
        mu, sd = calibrate(pick(calib[0], calib[1]), calib[2])
        for N in PREFIXES:
            sc = statistics(pick(clean[0], clean[1]), clean[2], mu, sd, N); sf = statistics(pick(faulty[0], faulty[1]), faulty[2], mu, sd, N)
            for name in sc:
                y = np.r_[np.zeros(len(sc[name])), np.ones(len(sf[name]))]; s = np.r_[sc[name], sf[name]]
                row = dict(all=float(roc_auc_score(y, s)))
                for kind in np.unique(kinds):
                    sel = np.r_[np.ones(len(sc[name]), bool), np.asarray(kinds) == kind]
                    row[str(kind)] = float(roc_auc_score(y[sel], s[sel]))
                res.setdefault(channel, {}).setdefault(name, {})[str(N)] = row
    return res


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--tag', required=True); ap.add_argument('--family', choices=['native', 'dense'], required=True)
    ap.add_argument('--result', required=True, help='fit result json (native race_tpp_fas_v3 or dense reference)'); ap.add_argument('--runs', type=int, default=1000)
    a = ap.parse_args(); torch.set_num_threads(1); torch.set_default_dtype(torch.float64 if a.family == 'native' else torch.float32); began = time.time()
    out = ROOT/f'experiments/results/fas/{a.tag}.json'; assert not out.exists()
    calib, _ = load(DATA/CALIB/'val_clean.npz', 1100); val_c, _ = load(DATA/MAIN/'val_clean.npz', 1100); val_f, val_k = load(DATA/MAIN/'val_faulty.npz', 1100)
    calib, val_c, val_f, val_k = calib[:a.runs], val_c[:a.runs], val_f[:a.runs], val_k[:a.runs]
    fn = per_event_native if a.family == 'native' else per_event_dense
    packs = {}
    for name, runs in (('calib', calib), ('clean', val_c), ('faulty', val_f)):
        PT, PG, nll = fn(a.result, runs); packs[name] = (PT, PG, targets(runs, PT.shape[1]), nll)
    res = detector_aurocs(packs['clean'][:3], packs['faulty'][:3], packs['calib'][:3], val_k)
    record = dict(status='completed', battle='B3', tag=a.tag, family=a.family, model_result=a.result, model_result_sha256=sha(ROOT/a.result),
                  calib_clean_nll=packs['calib'][3], val_clean_nll=packs['clean'][3], val_auroc=res, runs=a.runs, wall_s=time.time() - began,
                  data=dict(main=MAIN, calibration=CALIB), test_opened=False,
                  source_sha256={n: sha(ROOT/n) for n in ['experiments/fas/addressed_evidence.py', 'experiments/fas/native.py', 'experiments/fas/race_tpp_fas_v3.py', 'experiments/fas/dense.py', 'experiments/tpp/race_tpp_v16.py']},
                  scope='Validation-only detector development; per-type standardization from a separate clean calibration draw (seed base 520261015, disjoint from every v2 split). No TEST access; model weights fixed during scoring.')
    out.write_text(json.dumps(record, indent=1) + '\n')
    best = sorted(((v['1024']['all'], ch, st) for ch, d in res.items() for st, v in d.items()), reverse=True)[:6]
    print(json.dumps(dict(val_clean_nll=packs['clean'][3], best=best)))


if __name__ == '__main__':
    main()
