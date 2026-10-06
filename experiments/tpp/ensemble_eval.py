#!/usr/bin/env python3
"""B1: evaluate a predictive mixture of trained race models (evaluation only).

At every scored event the next-event density is the uniform mixture of the members' densities,
p(τ, k | history) = (1/J) Σ_j λ^j_k(τ) S^j(τ), which is itself a valid marked point process (the predictive
distribution of a model average). Per-event log-likelihood is log of that mixture; time LL uses the mixture of the
members' time densities and mark LL is the difference. Members are the seeds of one selected configuration.
"""
import argparse, importlib, json, math, random, sys
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))


def member_terms(mod, r, seqs):
    a = r['args']
    train = mod.load_split(a['dataset'], 'train')
    gaps = np.concatenate([np.diff(t) for t, _ in train]); pos = gaps[gaps > 0]
    qs = np.log(np.quantile(pos, np.linspace(0.1, 0.9, a['n_lognormal']))).tolist()
    kw = {'floor_cell': a['floor_cell']} if 'floor_cell' in a else {}
    if a.get('n_window', 0):
        qq = np.quantile(pos, np.linspace(0.02, 0.98, a['n_window'] + 1))
        kw.update(n_window=a['n_window'], window_edges=(qq[:-1].tolist(), qq[1:].tolist()))
    if a.get('state_modes', 0):
        kw.update(state_modes=a['state_modes'])
    m = mod.RaceTPP(r['K'], a['d'], a['modes'], a['layers'], a['n_exp'], a['n_lognormal'], a['dv'], 0.0, r['scale'],
                    qs, **kw)
    m.load_state_dict(torch.load(ROOT / r['checkpoint'])); m.eval()
    times, totals = [], []
    with torch.no_grad():
        for t, mk, mask in mod.batches(seqs, 64, False, random.Random(0)):
            if hasattr(m, 'event_terms'):
                lt, lk, _ = m.event_terms(t, mk, mask); v = mask[:, 1:]
                times.append(lt[v]); totals.append(lk[v]); continue
            h, slots = m.encode(t, mk, mask)
            params = m.clocks(h[:, :-1], slots[:, :-1])
            tau = (t[:, 1:] - t[:, :-1]).clamp_min(0)
            log_h, log_s = m.clock_terms(tau, *params[:4])
            lt = torch.logsumexp(log_h, -1) + log_s.sum(-1)
            lk = torch.logsumexp(log_h + params[4].gather(-1, mk[:, 1:, None, None].expand(-1, -1, m.M, 1)).squeeze(-1), -1) + log_s.sum(-1)
            v = mask[:, 1:]
            times.append(lt[v]); totals.append(lk[v])
    return torch.cat(times), torch.cat(totals)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results', nargs='+', required=True)
    ap.add_argument('--split', default='dev', choices=['dev', 'test'])
    ap.add_argument('--out')
    a = ap.parse_args()
    torch.set_num_threads(1); torch.set_default_dtype(torch.float64)
    rs = [json.loads((ROOT / p).read_text()) for p in a.results]
    ds = rs[0]['args']['dataset']; assert all(r['args']['dataset'] == ds for r in rs)
    mod = importlib.import_module(Path(next(iter(rs[0]['source_sha256']))).stem)
    seqs = mod.load_split(ds, a.split)
    T, L = zip(*(member_terms(importlib.import_module(Path(next(iter(r['source_sha256']))).stem), r, seqs) for r in rs))
    T, L = torch.stack(T), torch.stack(L)                       # [J, events]
    J = len(rs)
    mix_time = torch.logsumexp(T, 0) - math.log(J); mix_total = torch.logsumexp(L, 0) - math.log(J)
    out = dict(dataset=ds, split=a.split, members=[r['tag'] for r in rs], events=int(T.shape[1]),
               member_ll=[float(x.mean()) for x in L], ensemble=dict(ll=float(mix_total.mean()),
               time_ll=float(mix_time.mean()), mark_ll=float((mix_total - mix_time).mean())),
               macs_per_event=None)
    print(json.dumps(out, indent=1))
    if a.out:
        (ROOT / a.out).write_text(json.dumps(out, indent=1) + '\n')


if __name__ == '__main__':
    main()
