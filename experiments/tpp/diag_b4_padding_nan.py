import json, sys, random, numpy as np, torch
sys.path.insert(0, '/workspace/experiments/tpp')
import race_tpp_v19 as base
from race_tpp_b4 import load, recording_cell, n_components
torch.set_default_dtype(torch.float64); torch.set_num_threads(1)
r = json.load(open('/workspace/experiments/results/tpp_b4/b4g_github_s1.json')); cfg = r['config']
train, K = load('github_filtered', 1, 'train')
gaps = np.concatenate([np.diff(t) for t, _ in train]); pos = gaps[gaps > 0]
cell = recording_cell(pos); comps = n_components(pos); nw = 4 if comps >= 2 else 0
qs = np.log(np.quantile(pos, np.linspace(0.1, 0.9, cfg['n_lognormal']))).tolist()
edges = base.cluster_windows(pos, nw, 0) if nw else None
m = base.RaceTPP(K + 1, cfg['d'], cfg['modes'], cfg['layers'], cfg['n_exp'], cfg['n_lognormal'], cfg['dv'], cfg['dropout'], float(np.median(pos)), qs, cell, nw, edges, cfg['state_modes'])
m.load_state_dict(torch.load('/workspace/' + r['checkpoint'])); m.eval()
L = sorted(range(len(train)), key=lambda i: -len(train[i][1]))
print('longest train sequences:', [len(train[i][1]) for i in L[:3]], 'cell', cell)
for i in L[:2]:
    t, mk = train[i]
    tt = torch.tensor(t)[None]; mm = torch.tensor(mk)[None]; mask = torch.ones_like(mm, dtype=torch.bool)
    m.zero_grad()
    with torch.autograd.set_detect_anomaly(True):
        try:
            tl, ml, n, _ = m.loglik(tt, mm, mask, cell)
            loss = -(tl + ml) / n
            print('seq', i, 'len', len(mk), 'loss', float(loss), 'zero gaps', int((np.diff(t) == 0).sum()))
            loss.backward()
            bad = [k for k, p in m.named_parameters() if p.grad is not None and not torch.isfinite(p.grad).all()]
            print('  non-finite grads:', bad[:6])
        except RuntimeError as e:
            print('  anomaly:', str(e).split('\n')[0][:300])
print('--- padded batch of the 32 longest sequences ---')
idx = L[:32]
n_ = max(len(train[i][1]) for i in idx)
T = np.zeros((32, n_)); M = np.zeros((32, n_), np.int64); MK = np.zeros((32, n_), bool)
for r_, i in enumerate(idx):
    t, mk = train[i]; T[r_, :len(t)] = t; M[r_, :len(t)] = mk; MK[r_, :len(t)] = True; T[r_, len(t):] = t[-1]
tt, mm, mask = torch.tensor(T), torch.tensor(M), torch.tensor(MK)
for mode in ('eval', 'train'):
    getattr(m, mode)(); m.zero_grad(); torch.manual_seed(0)
    tl, ml, n, _ = m.loglik(tt, mm, mask, cell); loss = -(tl + ml) / n; loss.backward()
    bad = [k for k, p in m.named_parameters() if p.grad is not None and not torch.isfinite(p.grad).all()]
    print(mode, 'events', int(mask.sum()), 'loss', float(loss), 'non-finite grads', len(bad), bad[:5])
m.eval(); m.zero_grad()
with torch.autograd.set_detect_anomaly(True):
    try:
        tl, ml, n, _ = m.loglik(tt, mm, mask, cell); (-(tl + ml) / n).backward(); print('anomaly: none')
    except RuntimeError as e:
        print('anomaly:', str(e).split('\n')[0][:400])
