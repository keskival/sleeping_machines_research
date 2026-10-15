"""B10-P classical references on the identical tick-cell return targets: unconditional Student-t, realized-volatility
scaled Student-t (RiskMetrics-style, past 300 s realized variance) and a linear-feature Student-t whose location and
log-scale are linear in all seven causal inputs the neural models receive. Fitted by maximum likelihood on TRAIN windows
(density at the cell midpoint; cell width << scale), scored on DEV positions 128..1023 with exact Student-t cell
probabilities."""
import argparse, hashlib, json, math, resource, sys, time
from pathlib import Path
import numpy as np
import torch
from scipy.special import stdtr
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT/'experiments/market'))
from b10_depth_tpp_v1 import TRAIN, VAL, CTX
from b10p_data import split_windows, standardize, receipt, HORIZONS, NFEAT

sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()


def flatten(rows, start):
    f = np.concatenate([r['feats'][start:] for r in rows]); lo = np.concatenate([r['lo'][start:] for r in rows]); hi = np.concatenate([r['hi'][start:] for r in rows])
    return f, lo, hi


def design(kind, f):
    if kind == 'unconditional': return np.ones((len(f), 1)), np.ones((len(f), 1))
    if kind == 'rv_scaled': return np.ones((len(f), 1)), np.stack([np.ones(len(f)), f[:, 3]], 1)
    return np.concatenate([np.ones((len(f), 1)), f], 1), np.concatenate([np.ones((len(f), 1)), f], 1)


def fit(kind, f, mid, iters=300):
    X_loc, X_scale = (torch.tensor(x) for x in design(kind, f)); y = torch.tensor(mid)
    loc = torch.zeros(X_loc.shape[1], requires_grad=True); sc = torch.zeros(X_scale.shape[1], requires_grad=True)
    with torch.no_grad(): sc[0] = math.log(np.std(mid) + 1e-6)
    lognu = torch.tensor(math.log(4.0), requires_grad=True)
    opt = torch.optim.LBFGS([loc, sc, lognu], max_iter=iters, line_search_fn='strong_wolfe')
    def closure():
        opt.zero_grad(); nu = lognu.exp() + 2.0
        d = torch.distributions.StudentT(nu, X_loc @ loc, (X_scale @ sc).exp())
        loss = -d.log_prob(y).mean(); loss.backward(); return loss
    opt.step(closure)
    return dict(loc=loc.detach().numpy().tolist(), scale=sc.detach().numpy().tolist(), nu=float(lognu.exp() + 2.0))


def cell_nll(kind, params, f, lo, hi):
    X_loc, X_scale = design(kind, f); mu = X_loc @ np.array(params['loc']); s = np.exp(X_scale @ np.array(params['scale'])); nu = params['nu']
    a = (lo - mu)/s; b = (hi - mu)/s; upper = a > 0
    p = np.where(upper, stdtr(nu, -a) - stdtr(nu, -b), stdtr(nu, b) - stdtr(nu, a))
    return -np.log(np.clip(p, 1e-300, None))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--tag', required=True); a = ap.parse_args(); torch.set_num_threads(1); torch.set_default_dtype(torch.float64); began = time.monotonic()
    out = ROOT/f'experiments/results/market/{a.tag}.json'; assert not out.exists()
    train = split_windows(TRAIN, 20); dev = split_windows(VAL, 10); moments = standardize(train, dev)
    ft, lot, hit = flatten(train, 0); fd, lod, hid = flatten(dev, CTX)
    rows = {}
    for kind in ('unconditional', 'rv_scaled', 'linear_features'):
        rows[kind] = {}
        for h, H in enumerate(HORIZONS):
            vt = np.isfinite(lot[:, h]); vd = np.isfinite(lod[:, h])
            params = fit(kind, ft[vt], 0.5*(lot[vt, h] + hit[vt, h]))
            nll = cell_nll(kind, params, fd[vd], lod[vd, h], hid[vd, h])
            rows[kind][f'{int(H)}s'] = dict(nll=float(nll.mean()), targets=int(vd.sum()), params=params)
            print(kind, H, rows[kind][f'{int(H)}s']['nll'], flush=True)
        rows[kind]['return_nll_sum'] = sum(rows[kind][f'{int(H)}s']['nll'] for H in HORIZONS)
    record = dict(status='completed', battle='B10', task='B10-P price', references=rows, feature_moments=moments,
                  source_sha256={n: sha(ROOT/n) for n in ['experiments/market/b10p_references.py', 'experiments/market/b10p_data.py']},
                  data_sha256=receipt(list(TRAIN)+list(VAL)), wall_s=time.monotonic()-began, max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                  september_opened=False, scope='Classical references fitted on all TRAIN window events, scored on the same DEV positions 128..1023 and exact 0.01 USDT tick cells as the neural B10-P models.')
    out.write_text(json.dumps(record, indent=2)+'\n')


if __name__ == '__main__':
    main()
