#!/usr/bin/env python3
"""R1 recall diagnosis (evaluation only): where does keyed recall fail?

For a trained recall_tpp(_v2) checkpoint, on the DEV split:
- recall accuracy by pair age (queries answered by the k-th most recent pair) and by elapsed time since the pair;
- information availability: is the preceding key linearly decodable from the state h at each context value event
  (the h the keyed memory writes)? A ridge readout is fitted on half of DEV and scored on the other half;
- match quality: rank of the correct value slot under the query score q · key alone.
"""
import argparse
import importlib
import json
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--result', required=True, help='result JSON of the trained run')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    torch.set_num_threads(1); torch.set_default_dtype(torch.float64)
    res = json.loads((ROOT / a.result).read_text()); args = res['args']
    driver = next(k for k in res['source_sha256'] if 'recall_tpp' in k)
    mod = importlib.import_module(Path(driver).stem)
    kk, kv, n = args['kk'], args['kv'], args['pairs']
    T, M, R = mod.make_split(args['dev'], n, kk, kv, np.random.default_rng(2000 + args['seed']))[:3]
    K = kk + kv
    gaps = torch.diff(mod.make_split(args['train'], n, kk, kv, np.random.default_rng(1000 + args['seed']))[0], dim=1)
    pos = gaps[gaps > 0].numpy(); scale = float(np.median(pos))
    qs = np.log(np.quantile(pos, np.linspace(0.1, 0.9, 4))).tolist()
    margs = (K, args['d'], args['modes'], args['layers'], 2, 4, args['dv'], 0.0, scale, qs)
    kw = dict(dk=args['dk'])
    if args['keyed'] == 2:
        kw['local'] = True
    for f in ('prev_msg', 'qk_norm'):
        if args.get(f):
            kw[f] = True
    model = mod.KeyedRaceTPP(*margs, **kw) if args['keyed'] else mod.RaceTPP(*margs)
    model.load_state_dict(torch.load(ROOT / res['checkpoint'])); model.eval()
    mask = torch.ones_like(M, dtype=torch.bool)
    with torch.no_grad():
        _, _, _, logp = mod.mark_posterior(model, T, M, mask)
        h, _ = model.encode(T, M, mask)
    pred = logp[..., kk:].argmax(-1) + kk; tgt = M[:, 1:]; ok = (pred == tgt).numpy()
    # pair age and elapsed time for each query value
    by_age, by_dt = {}, {}
    for s in range(len(T)):
        keys_ctx = M[s, 0:2 * n:2].tolist()
        for j in range(n, 2 * n):
            qi = 2 * j; key = int(M[s, qi]); p = keys_ctx.index(key)
            age = n - 1 - p + (j - n)                                    # pairs shown since this pair (incl. queries)
            dt = float(T[s, qi] - T[s, 2 * p + 1])
            c = bool(ok[s, qi])                                           # prediction for event qi+1 sits at index qi
            by_age.setdefault(age, []).append(c)
            by_dt.setdefault(int(np.floor(np.log10(max(dt, 1e-3)))), []).append(c)
    out = dict(result=a.result, recall_acc=float(np.mean([c for v in by_age.values() for c in v])),
               by_pair_age={k: dict(n=len(v), acc=float(np.mean(v))) for k, v in sorted(by_age.items())},
               by_log10_elapsed={k: dict(n=len(v), acc=float(np.mean(v))) for k, v in sorted(by_dt.items())})
    # decodability of the preceding key from h at context value events
    X = h[:, 1:2 * n:2].reshape(-1, h.shape[-1]).numpy(); y = M[:, 0:2 * n:2].reshape(-1).numpy()
    half = len(X) // 2; Y = np.eye(kk)[y]
    Xa = np.c_[X, np.ones(len(X))]
    W = np.linalg.solve(Xa[:half].T @ Xa[:half] + 1e-2 * np.eye(Xa.shape[1]), Xa[:half].T @ Y[:half])
    out['prev_key_decodable_acc'] = float(((Xa[half:] @ W).argmax(1) == y[half:]).mean())
    out['prev_key_chance'] = 1 / kk
    if args['keyed']:
        with torch.no_grad():
            score = torch.einsum('bld,blkd->blk', model._query, model._keyed)[..., kk:]
        ranks = []
        for s in range(len(T)):
            for j in range(n, 2 * n):
                qi = 2 * j; v = int(M[s, qi + 1]) - kk
                ranks.append(int((score[s, qi] > score[s, qi, v]).sum()))
        ranks = np.array(ranks)
        out['match_rank_top1'] = float((ranks == 0).mean()); out['match_rank_mean'] = float(ranks.mean())
    p = ROOT / a.out; p.parent.mkdir(parents=True, exist_ok=True); p.write_text(json.dumps(out, indent=1) + '\n')
    print(json.dumps(out), flush=True)


if __name__ == '__main__':
    main()
