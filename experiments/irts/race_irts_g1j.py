#!/usr/bin/env python3
"""G1j: joint silence-aware supervision — the supervised classifier also predicts every next measurement event.

G1 pretrained the next-measurement objective and then fine-tuned (no robust gain). G1j keeps the generative objective
DURING supervised training with weight --aux: every time step supervises the shared temporal memory with when the next
measurement comes, which channels are measured and their values (dense supervision), while the label supervises the
readout. Motivation: P12 classifiers peak by epoch 2-5 and then overfit the one label bit per record.

Original G1 description:

Stage 1 (--pretrain-epochs > 0, no labels used): the B2 encoder (race_irts_v3: temporal memory layers shared with the
EasyTPP model, addressed channel memory) predicts, from the state after step i, the next measurement event at step i+1:
  - when: a mixture of log-normal densities over the gap (time log-likelihood),
  - which channels are measured: independent Bernoulli per channel,
  - their values: Gaussian on the normalized value, observed channels only.
Stage 2: the classifier head is trained on a label fraction of TRAIN (seeded subset), the whole network fine-tuned;
selection by validation AUROC. The scratch control runs stage 2 alone for pretrain + finetune epochs (equal total
epochs). TEST is not scored here (development); G1's protocol scores TEST once per configuration later.
"""
import argparse, hashlib, json, math, random, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'experiments/irts'))
from race_irts_v3 import RaceIRTS  # noqa: E402


class G1Model(RaceIRTS):
    def __init__(self, C, S, d, modes, layers, J, dv, dropout, n_mix=4):
        super().__init__(C, S, d, modes, layers, J, dv, dropout, 1)
        self.n_mix = n_mix
        self.next_time = nn.Linear(d, 3 * n_mix)          # mixture logits, means, log-scales of log-gap
        self.next_obs = nn.Linear(d, C)                   # which channels are measured next
        self.next_val = nn.Linear(d, 2 * C)               # their normalized values (mean, log-scale)

    def steps(self, t, z, mask, lens):
        """Per-step state after each event (the encoder's temporal memory path)."""
        B, T, C = z.shape
        valid = torch.arange(T).unsqueeze(0) < lens.unsqueeze(1)
        dt = torch.diff(t, dim=1, prepend=t[:, :1]).clamp_min(0) * valid
        per = self.typed(z, mask)
        x = per.sum(2) + self.gap(torch.stack([torch.log1p(dt), mask.float().mean(-1)], -1))
        for layer in self.layers:
            x = layer(x, dt) * valid.unsqueeze(-1)
        return x, dt, valid

    def pretrain_loss(self, t, z, mask, lens):
        x, dt, valid = self.steps(t, z, mask, lens)
        h = x[:, :-1]; nxt_dt = dt[:, 1:]; nxt_mask = mask[:, 1:].float(); nxt_z = z[:, 1:]
        tgt = valid[:, 1:] & (nxt_dt > 0)
        # time: mixture of log-normals over the gap (density in the data's time units)
        p = self.next_time(h); K = self.n_mix
        logw = F.log_softmax(p[..., :K], -1); mu = p[..., K:2 * K]; ls = p[..., 2 * K:].clamp(-5, 3)
        lg = torch.log(nxt_dt.clamp_min(1e-6)).unsqueeze(-1)
        comp = logw - lg - ls - 0.5 * math.log(2 * math.pi) - 0.5 * ((lg - mu) / ls.exp()) ** 2
        time_ll = torch.logsumexp(comp, -1)
        obs_ll = -F.binary_cross_entropy_with_logits(self.next_obs(h), nxt_mask, reduction='none').sum(-1)
        v = self.next_val(h); vm, vls = v[..., :z.shape[2]], v[..., z.shape[2]:].clamp(-5, 3)
        val_ll = ((-vls - 0.5 * math.log(2 * math.pi) - 0.5 * ((nxt_z - vm) / vls.exp()) ** 2) * nxt_mask).sum(-1)
        w = tgt.float(); n = w.sum().clamp_min(1)
        return -((time_ll + obs_ll + val_ll) * w).sum() / n, dict(time=float((time_ll * w).sum() / n),
                                                                     obs=float((obs_ll * w).sum() / n), val=float((val_ll * w).sum() / n))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default='P12', choices=['P12', 'P19'])
    ap.add_argument('--split', type=int, default=0)
    ap.add_argument('--tag', required=True)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--label-frac', type=float, default=1.0)
    ap.add_argument('--pretrain-epochs', type=int, default=0)
    ap.add_argument('--finetune-epochs', type=int, default=40)
    ap.add_argument('--patience', type=int, default=12)
    ap.add_argument('--lr', type=float, default=2e-3)
    ap.add_argument('--batch', type=int, default=128)
    ap.add_argument('--d', type=int, default=32)
    ap.add_argument('--dropout', type=float, default=0.2)
    ap.add_argument('--aux', type=float, default=0.0, help='weight of the next-measurement log-loss during supervised training')
    a = ap.parse_args()
    torch.set_num_threads(1); random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    D = np.load(ROOT / f'data/raindrop/cache/{a.dataset}.npz')
    tr, va = D[f'split{a.split}_train'], D[f'split{a.split}_val']
    vals, mask = D['vals'], D['mask']; C = vals.shape[2]
    obs = mask[tr]; v = vals[tr]
    nonneg = np.array([(v[..., c][obs[..., c]] >= 0).all() for c in range(C)])
    tv = np.where(nonneg, np.log1p(np.clip(vals, 0, None)), vals)
    mu = np.array([tv[tr][..., c][obs[..., c]].mean() if obs[..., c].any() else 0 for c in range(C)])
    sd = np.array([tv[tr][..., c][obs[..., c]].std() + 1e-6 if obs[..., c].any() else 1 for c in range(C)])
    z = np.clip((tv - mu) / sd, -6, 6) * mask
    st = D['static']; stz = (st - st[tr].mean(0)) / (st[tr].std(0) + 1e-6)
    t = torch.from_numpy(D['times']); Z = torch.from_numpy(z.astype(np.float32)); M = torch.from_numpy(mask)
    Lens = torch.from_numpy(D['lens']); S = torch.from_numpy(stz.astype(np.float32)); y = D['y']
    rng = np.random.default_rng(1000 + a.seed)
    lab = np.sort(rng.choice(tr, max(16, int(round(a.label_frac * len(tr)))), replace=False)) if a.label_frac < 1 else tr
    model = G1Model(C, S.shape[1], a.d, 16, 2, 4, 4, a.dropout)
    out_dir = ROOT / 'experiments/results/irts'; out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = out_dir / f'{a.tag}.pt'; start = time.time(); pre_hist = []

    def batches(idx, bs):
        perm = np.random.permutation(idx)
        for i in range(0, len(perm), bs):
            b = torch.from_numpy(perm[i:i + bs]); T = int(Lens[b].max())
            yield b, T

    if a.pretrain_epochs:
        opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
        for ep in range(a.pretrain_epochs):           # all TRAIN records, labels never read
            model.train(); tot = n = 0
            for b, T in batches(tr, a.batch):
                loss, parts = model.pretrain_loss(t[b, :T], Z[b, :T], M[b, :T], Lens[b])
                opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
                tot += loss.item() * len(b); n += len(b)
            model.eval()
            with torch.no_grad():
                vb = torch.from_numpy(va[:512]); T = int(Lens[vb].max())
                vl, vparts = model.pretrain_loss(t[vb, :T], Z[vb, :T], M[vb, :T], Lens[vb])
            pre_hist.append(dict(epoch=ep, train_nll=tot / n, val_nll=float(vl), **{f'val_{k}': x for k, x in vparts.items()}))
            print(json.dumps(pre_hist[-1]), flush=True)
    total_ft = a.finetune_epochs + (0 if a.pretrain_epochs else 0)
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
    for ep in range(total_ft):
        model.train()
        for b, T in batches(lab, a.batch):
            out = model(t[b, :T], Z[b, :T], M[b, :T], Lens[b], S[b])[:, 0]
            loss = F.binary_cross_entropy_with_logits(out, torch.from_numpy(y[b.numpy()]).float(), pos_weight=pos_weight)
            if a.aux > 0:
                gen, _ = model.pretrain_loss(t[b, :T], Z[b, :T], M[b, :T], Lens[b])
                loss = loss + a.aux * gen
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        vm = evaluate(va); hist.append(dict(epoch=ep, **{f'val_{k}': x for k, x in vm.items()}))
        print(json.dumps(hist[-1]), flush=True)
        if vm['auroc'] > best:
            best, best_ep = vm['auroc'], ep; torch.save(model.state_dict(), ckpt)
        if ep - best_ep >= a.patience:
            break
    model.load_state_dict(torch.load(ckpt))
    src = Path(__file__).resolve()
    res = dict(status='completed', track='G1j', tag=a.tag, args=vars(a), labelled=int(len(lab)), val=evaluate(va),
               best_epoch=best_ep, pretrain_history=pre_hist, history=hist, wall_s=time.time() - start,
               parameters=sum(p.numel() for p in model.parameters()), checkpoint=str(ckpt.relative_to(ROOT)),
               source_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                              for p in (src, ROOT / 'experiments/irts/race_irts_v3.py', ROOT / 'experiments/tpp/race_tpp_v8.py')})
    (out_dir / f'{a.tag}.json').write_text(json.dumps(res, indent=1) + '\n')
    print('VAL', json.dumps(res['val']), flush=True)


if __name__ == '__main__':
    main()
