#!/usr/bin/env python3
"""Re-score final B1 models on TEST with the EasyTPP / S2P2-fork estimator (evaluation only).

Their per-event log-likelihood (easy_tpp/model/torch_model/torch_basemodel.py, compute_loglikelihood, read at commit
c3933240 of UCIDataLab/state_space_point_process): for events 2..N,
    event term     = log λ_{k_i}(t_i)                       (intensity at the event, left limit)
    non-event term = Δt_i · mean_{g=1..G} Σ_k λ_k(t_{i-1} + u_g Δt_i),  u_g ~ U(0, 1), G = 10 (use_mc_samples: True)
    per-event LL   = Σ (event − non-event) / number of scored events.
We compute the same quantity from our model's intensities and compare with our exact compensator. The Monte Carlo
estimate is unbiased for the compensator; repeated draws show its spread.
"""
import argparse, importlib, json, math, random, sys
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))


def build(r):
    a = r['args']; mod = importlib.import_module(Path(next(iter(r['source_sha256']))).stem)
    train = mod.load_split(a['dataset'], 'train')
    gaps = np.concatenate([np.diff(t) for t, _ in train]); pos = gaps[gaps > 0]
    qs = np.log(np.quantile(pos, np.linspace(0.1, 0.9, a['n_lognormal']))).tolist()
    kw = {'floor_cell': a['floor_cell']} if 'floor_cell' in a else {}
    if a.get('n_window', 0):
        kw.update(n_window=a['n_window'], window_edges=([0.1] * a['n_window'], [0.2] * a['n_window']))
    if a.get('state_modes', 0):
        kw.update(state_modes=a['state_modes'])
    m = mod.RaceTPP(r['K'], a['d'], a['modes'], a['layers'], a['n_exp'], a['n_lognormal'], a['dv'], 0.0, r['scale'], qs, **kw)
    m.load_state_dict(torch.load(ROOT / r['checkpoint'])); m.eval()
    return mod, m


def total_log_intensity(m, cache, tau_q):
    """log Σ_k λ_k at elapsed times tau_q [B, L, Q] after each event."""
    h, slots, params = cache[0], cache[1], cache[2]
    lh, _ = m.clock_terms(tau_q, *(p.unsqueeze(-2) for p in params[:4]))
    out = torch.logsumexp(lh, -1)
    if getattr(m, 'ns', 0):
        zr, zi = cache[5], cache[6]
        tq = m.state_tau(tau_q) if hasattr(m, 'state_tau') else tau_q
        sl, _ = m.state_eval(h[:, :-1], zr, zi, tq, marks=False)
        out = torch.logaddexp(out, sl)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results', nargs='+', required=True)
    ap.add_argument('--samples', type=int, default=10)
    ap.add_argument('--repeats', type=int, default=5)
    ap.add_argument('--write', action='store_true', help='append rows to the tracked results file')
    a = ap.parse_args()
    torch.set_num_threads(1); torch.set_default_dtype(torch.float64)
    rows = []
    for path in a.results:
        r = json.loads((ROOT / path).read_text()); mod, m = build(r)
        test = mod.load_split(r['args']['dataset'], 'test')
        exact_tot, n = 0.0, 0; mc = np.zeros(a.repeats)
        with torch.no_grad():
            for t, mk, mask in mod.batches(test, 16, False, random.Random(0)):
                if hasattr(m, 'event_terms'):
                    time_lp, joint_lp, cache = m.event_terms(t, mk, mask)
                else:                                                    # v5: closed-form clocks only
                    h, slots = m.encode(t, mk, mask)
                    params = m.clocks(h[:, :-1], slots[:, :-1])
                    tau = (t[:, 1:] - t[:, :-1]).clamp_min(0)
                    log_h, log_s = m.clock_terms(tau, *params[:4])
                    surv = log_s.sum(-1)
                    time_lp = torch.logsumexp(log_h, -1) + surv
                    joint_lp = torch.logsumexp(log_h + params[4].gather(-1, mk[:, 1:, None, None].expand(-1, -1, m.M, 1)).squeeze(-1), -1) + surv
                    cache = (h, slots, params, tau, mask[:, 1:])
                tau, valid = cache[3], cache[4]
                log_lam_tau = total_log_intensity(m, cache, tau.unsqueeze(-1))[..., 0]
                exact_surv = time_lp - log_lam_tau
                event_term = joint_lp - exact_surv                         # log λ_k(t_i)
                exact_tot += float((joint_lp * valid).sum()); n += int(valid.sum())
                for rep in range(a.repeats):
                    g = torch.Generator().manual_seed(1000 * rep + n)
                    u = torch.rand(*tau.shape, a.samples, generator=g)
                    lam = total_log_intensity(m, cache, tau.unsqueeze(-1) * u).exp().mean(-1)
                    mc[rep] += float(((event_term - tau * lam) * valid).sum())
        row = dict(result=path, dataset=r['args']['dataset'], exact=exact_tot / n, easytpp_mc_mean=float(mc.mean() / n),
                   easytpp_mc_sd=float(mc.std(ddof=1) / n), recorded_test=r['test']['ll'], events=n)
        rows.append(row); print(json.dumps(row), flush=True)
    if a.write:
        out = ROOT / 'experiments/results/tpp/b1_easytpp_estimator_check.json'
        old = json.loads(out.read_text()) if out.exists() else []
        out.write_text(json.dumps(old + rows, indent=1) + '\n')


if __name__ == '__main__':
    main()
