#!/usr/bin/env python3
"""B2: average the predicted probabilities of several trained models (same dataset and split), evaluation only.

Each member is rebuilt from its result JSON (driver module from source_sha256, args) with the same TRAIN-only
normalization as its driver, scored on the chosen split, and the probabilities are averaged.
"""
import argparse, importlib, json, sys
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, average_precision_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'experiments/irts'))


def prepared(dataset, split):
    D = np.load(ROOT / f'data/raindrop/cache/{dataset}.npz')
    tr = D[f'split{split}_train']; vals, mask = D['vals'], D['mask']; C = vals.shape[2]
    obs = mask[tr]; v = vals[tr]
    nonneg = np.array([(v[..., c][obs[..., c]] >= 0).all() for c in range(C)])
    tv = np.where(nonneg, np.log1p(np.clip(vals, 0, None)), vals)
    mu = np.array([tv[tr][..., c][obs[..., c]].mean() if obs[..., c].any() else 0 for c in range(C)])
    sd = np.array([tv[tr][..., c][obs[..., c]].std() + 1e-6 if obs[..., c].any() else 1 for c in range(C)])
    z = np.clip((tv - mu) / sd, -6, 6) * mask
    st = D['static']; stz = (st - st[tr].mean(0)) / (st[tr].std(0) + 1e-6)
    return D, torch.from_numpy(D['times']), torch.from_numpy(z.astype(np.float32)), torch.from_numpy(mask), \
        torch.from_numpy(D['lens']), torch.from_numpy(stz.astype(np.float32))


def member_probs(r, D, t, Z, M, L, S, idx):
    a = r['args']; src = [s for s in r['source_sha256'] if 'irts' in s][0]
    mod = importlib.import_module(Path(src).stem)
    C = Z.shape[2]
    if hasattr(mod, 'G1Model'):
        net = mod.G1Model(C, S.shape[1], a['d'], 16, 2, 4, 4, 0.0)
    else:
        net = mod.RaceIRTS(C, S.shape[1], a['d'], a.get('modes', 16), a.get('layers', 2), a.get('J', 4), a.get('dv', 4), 0.0, 1)
    net.load_state_dict(torch.load(ROOT / r['checkpoint'])); net.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(idx), 256):
            b = torch.from_numpy(idx[i:i + 256]); T = int(L[b].max())
            out.append(torch.sigmoid(net(t[b, :T], Z[b, :T], M[b, :T], L[b], S[b])[:, 0]))
    return torch.cat(out).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results', nargs='+', required=True)
    ap.add_argument('--part', default='val', choices=['val', 'test'])
    a = ap.parse_args()
    torch.set_num_threads(1)
    rs = [json.loads((ROOT / p).read_text()) for p in a.results]
    ds, split = rs[0]['args']['dataset'], rs[0]['args']['split']
    assert all(r['args']['dataset'] == ds and r['args']['split'] == split for r in rs)
    D, t, Z, M, L, S = prepared(ds, split)
    idx = D[f'split{split}_{a.part}']; y = D['y'][idx]
    P = [member_probs(r, D, t, Z, M, L, S, idx) for r in rs]
    for r, p in zip(rs, P):
        print(f"{r['tag']:40s} AUROC {roc_auc_score(y, p):.4f} AUPRC {average_precision_score(y, p):.4f}")
    pm = np.mean(P, 0)
    print(f"{'ENSEMBLE (' + str(len(P)) + ')':40s} AUROC {roc_auc_score(y, pm):.4f} AUPRC {average_precision_score(y, pm):.4f}")


if __name__ == '__main__':
    main()
