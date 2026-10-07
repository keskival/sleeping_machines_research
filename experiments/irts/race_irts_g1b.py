#!/usr/bin/env python3
"""G1b: cross-cohort self-supervised pretraining (P19 sepsis cohort, unlabeled) -> P12 mortality with few labels.

G1 pretrained on the same P12 TRAIN records it later fine-tuned on: it added an objective but no new data, and its
gain stayed within split spread (AUROC mean +0.008 over three splits at 10% labels). G1b pretrains the same encoder
(G1Model, unchanged) on P19's TRAIN stays (31,042 records from other hospitals; sepsis labels never read), mapped onto
P12's 36-variable layout, then fine-tunes on P12 exactly as G1 does. Each cohort is normalized with its own TRAIN
statistics (log1p for non-negative channels, z-score, clip ±6). P19 variables without a P12 counterpart are dropped;
P12 variables without a P19 counterpart are unobserved during pretraining. Both cohorts record time in hours.
--pretrain-source P19+P12 adds P12's own TRAIN records (unlabeled) to the pretraining pool.
Controls: the existing G1 scratch runs (g1_p12_s{k}_scratch_lf{f}). TEST is not scored (development).
"""
import argparse, hashlib, json, random, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'experiments/irts'))
from race_irts_g1 import G1Model  # noqa: E402

# Raindrop variable orders (P12 processed_data/ts_params.npy; P19 processed_data/labels_ts.npy without SepsisLabel)
P12_VARS = ['ALP', 'ALT', 'AST', 'Albumin', 'BUN', 'Bilirubin', 'Cholesterol', 'Creatinine', 'DiasABP', 'FiO2', 'GCS',
            'Glucose', 'HCO3', 'HCT', 'HR', 'K', 'Lactate', 'MAP', 'MechVent', 'Mg', 'NIDiasABP', 'NIMAP', 'NISysABP',
            'Na', 'PaCO2', 'PaO2', 'Platelets', 'RespRate', 'SaO2', 'SysABP', 'Temp', 'TroponinI', 'TroponinT', 'Urine',
            'WBC', 'pH']
P19_VARS = ['HR', 'O2Sat', 'Temp', 'SBP', 'MAP', 'DBP', 'Resp', 'EtCO2', 'BaseExcess', 'HCO3', 'FiO2', 'pH', 'PaCO2',
            'SaO2', 'AST', 'BUN', 'Alkalinephos', 'Calcium', 'Chloride', 'Creatinine', 'Bilirubin_direct', 'Glucose',
            'Lactate', 'Magnesium', 'Phosphate', 'Potassium', 'Bilirubin_total', 'TroponinI', 'Hct', 'Hgb', 'PTT', 'WBC',
            'Fibrinogen', 'Platelets']
P19_TO_P12 = {'HR': 'HR', 'Temp': 'Temp', 'SBP': 'SysABP', 'MAP': 'MAP', 'DBP': 'DiasABP', 'Resp': 'RespRate',
              'HCO3': 'HCO3', 'FiO2': 'FiO2', 'pH': 'pH', 'PaCO2': 'PaCO2', 'SaO2': 'SaO2', 'AST': 'AST', 'BUN': 'BUN',
              'Alkalinephos': 'ALP', 'Creatinine': 'Creatinine', 'Glucose': 'Glucose', 'Lactate': 'Lactate',
              'Magnesium': 'Mg', 'Potassium': 'K', 'Bilirubin_total': 'Bilirubin', 'TroponinI': 'TroponinI',
              'Hct': 'HCT', 'WBC': 'WBC', 'Platelets': 'Platelets'}


def normalized(D, tr):
    vals, mask = D['vals'], D['mask']; C = vals.shape[2]
    obs = mask[tr]; v = vals[tr]
    nonneg = np.array([(v[..., c][obs[..., c]] >= 0).all() for c in range(C)])
    tv = np.where(nonneg, np.log1p(np.clip(vals, 0, None)), vals)
    mu = np.array([tv[tr][..., c][obs[..., c]].mean() if obs[..., c].any() else 0 for c in range(C)])
    sd = np.array([tv[tr][..., c][obs[..., c]].std() + 1e-6 if obs[..., c].any() else 1 for c in range(C)])
    return np.clip((tv - mu) / sd, -6, 6) * mask


def p19_in_p12_layout(split):
    D = np.load(ROOT / 'data/raindrop/cache/P19.npz'); tr = D[f'split{split}_train']
    assert D['vals'].shape[2] == len(P19_VARS)
    z = normalized(D, tr)[tr]; m = D['mask'][tr]
    Z = np.zeros(z.shape[:2] + (len(P12_VARS),), np.float32); M = np.zeros(Z.shape, bool)
    for src, dst in P19_TO_P12.items():
        i, j = P19_VARS.index(src), P12_VARS.index(dst)
        Z[..., j] = z[..., i]; M[..., j] = m[..., i]
    return torch.from_numpy(D['times'][tr]), torch.from_numpy(Z), torch.from_numpy(M), torch.from_numpy(D['lens'][tr])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--split', type=int, default=0, help='P12 split for fine-tuning; P19 TRAIN of the same index')
    ap.add_argument('--tag', required=True)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--label-frac', type=float, default=0.1)
    ap.add_argument('--pretrain-source', default='P19', choices=['P19', 'P19+P12'])
    ap.add_argument('--pretrain-epochs', type=int, default=10)
    ap.add_argument('--finetune-epochs', type=int, default=40)
    ap.add_argument('--patience', type=int, default=12)
    ap.add_argument('--lr', type=float, default=2e-3)
    ap.add_argument('--batch', type=int, default=128)
    ap.add_argument('--d', type=int, default=32)
    ap.add_argument('--dropout', type=float, default=0.2)
    a = ap.parse_args()
    torch.set_num_threads(1); random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    D = np.load(ROOT / 'data/raindrop/cache/P12.npz')
    tr, va = D[f'split{a.split}_train'], D[f'split{a.split}_val']
    assert D['vals'].shape[2] == len(P12_VARS)
    z = normalized(D, tr); st = D['static']; stz = (st - st[tr].mean(0)) / (st[tr].std(0) + 1e-6)
    t = torch.from_numpy(D['times']); Z = torch.from_numpy(z.astype(np.float32)); M = torch.from_numpy(D['mask'])
    Lens = torch.from_numpy(D['lens']); S = torch.from_numpy(stz.astype(np.float32)); y = D['y']
    rng = np.random.default_rng(1000 + a.seed)                    # the same labelled subset as G1
    lab = np.sort(rng.choice(tr, max(16, int(round(a.label_frac * len(tr)))), replace=False)) if a.label_frac < 1 else tr
    model = G1Model(len(P12_VARS), S.shape[1], a.d, 16, 2, 4, 4, a.dropout)
    out_dir = ROOT / 'experiments/results/irts'; out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = out_dir / f'{a.tag}.pt'; start = time.time(); pre_hist = []

    pools = [p19_in_p12_layout(a.split)]
    if a.pretrain_source == 'P19+P12':
        pools.append((t[tr], Z[tr], M[tr], Lens[tr]))
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    for ep in range(a.pretrain_epochs):                            # labels never read
        model.train(); tot = n = 0
        plan = [(k, i) for k, (pt, _, _, _) in enumerate(pools) for i in range(0, len(pt), a.batch)]
        perms = [np.random.permutation(len(p[0])) for p in pools]
        random.shuffle(plan)
        for k, i in plan:
            pt, pz, pm, pl = pools[k]; b = torch.from_numpy(perms[k][i:i + a.batch]); T = int(pl[b].max())
            loss, _ = model.pretrain_loss(pt[b, :T], pz[b, :T], pm[b, :T], pl[b])
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            tot += loss.item() * len(b); n += len(b)
        model.eval()
        with torch.no_grad():                                      # P12 validation records, target cohort
            vb = torch.from_numpy(va[:512]); T = int(Lens[vb].max())
            vl, vparts = model.pretrain_loss(t[vb, :T], Z[vb, :T], M[vb, :T], Lens[vb])
        pre_hist.append(dict(epoch=ep, train_nll=tot / n, p12_val_nll=float(vl), **{f'p12_val_{k}': x for k, x in vparts.items()}))
        print(json.dumps(pre_hist[-1]), flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    pos_weight = torch.tensor((y[lab] == 0).sum() / max(1, (y[lab] == 1).sum()), dtype=torch.float32)

    def evaluate(idx):
        model.eval(); outs = []
        with torch.no_grad():
            for i in range(0, len(idx), 256):
                b = torch.from_numpy(idx[i:i + 256]); T = int(Lens[b].max())
                outs.append(model(t[b, :T], Z[b, :T], M[b, :T], Lens[b], S[b]))
        p = torch.sigmoid(torch.cat(outs)[:, 0]).numpy()
        return dict(auroc=float(roc_auc_score(y[idx], p)), auprc=float(average_precision_score(y[idx], p)))

    best, best_ep, hist = -1.0, -1, []
    for ep in range(a.finetune_epochs):
        model.train(); perm = np.random.permutation(lab)
        for i in range(0, len(perm), a.batch):
            b = torch.from_numpy(perm[i:i + a.batch]); T = int(Lens[b].max())
            out = model(t[b, :T], Z[b, :T], M[b, :T], Lens[b], S[b])[:, 0]
            loss = F.binary_cross_entropy_with_logits(out, torch.from_numpy(y[b.numpy()]).float(), pos_weight=pos_weight)
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        vm = evaluate(va); hist.append(dict(epoch=ep, **{f'val_{k}': x for k, x in vm.items()}))
        print(json.dumps(hist[-1]), flush=True)
        if vm['auroc'] > best:
            best, best_ep = vm['auroc'], ep; torch.save(model.state_dict(), ckpt)
        if ep - best_ep >= a.patience:
            break
    model.load_state_dict(torch.load(ckpt))
    src = Path(__file__).resolve()
    res = dict(status='completed', track='G1b', tag=a.tag, args=vars(a), labelled=int(len(lab)), val=evaluate(va),
               best_epoch=best_ep, pretrain_records=int(sum(len(p[0]) for p in pools)), pretrain_history=pre_hist,
               history=hist, wall_s=time.time() - start, parameters=sum(p.numel() for p in model.parameters()),
               checkpoint=str(ckpt.relative_to(ROOT)),
               source_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                              for p in (src, ROOT / 'experiments/irts/race_irts_g1.py', ROOT / 'experiments/irts/race_irts_v3.py',
                                        ROOT / 'experiments/tpp/race_tpp_v8.py')})
    (out_dir / f'{a.tag}.json').write_text(json.dumps(res, indent=1) + '\n')
    print('VAL', json.dumps(res['val']), flush=True)


if __name__ == '__main__':
    main()
