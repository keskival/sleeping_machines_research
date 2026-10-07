#!/usr/bin/env python3
"""B2: event-native classifier for irregular multivariate time series (P12, P19), Raindrop splits.

Every time step is an event carrying the channels observed then. Each observed value first passes typed comparisons
(learned per-channel thresholds, soft step features) before neural mixing; the event message is the sum over observed
channels of channel-specific projections. Persistent state: stacked complex-diagonal temporal memories that decay and
rotate with the real elapsed time (B1's layer), plus an addressed channel memory — one slot per channel, written only
when that channel is measured, decaying with time since its last measurement. Silence is information: the readout sees
how long each channel has gone unmeasured. Readout: final state, mean state, channel slots, staleness and statics.
Selection on the validation split (AUROC); TEST scored once at the selected checkpoint with --score-test.

v8 (7 Oct): + per-channel ordering latency (time of first measurement; never-measured channels sit past the record
end) and measurement rate, + record duration and step count. Diagnosis: P12 validation AUROC is 0.809 on the sparsest
third of stays vs 0.898 on the densest; in sparse records which tests are ordered, and when, carries the signal.
"""
import argparse, hashlib, json, math, random, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_fscore_support

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'experiments/tpp'))
from race_tpp_v8 import TemporalMemoryLayer  # noqa: E402  (same persistent temporal memory as B1)


class TypedChannelEncoder(nn.Module):
    """Per-channel typed comparisons: value -> [z, σ((z − θ_cj)/s_cj) for j] -> channel-specific projection."""

    def __init__(self, C, J, d):
        super().__init__()
        self.theta = nn.Parameter(torch.linspace(-1.5, 1.5, J).repeat(C, 1))
        self.log_s = nn.Parameter(torch.full((C, J), math.log(0.5)))
        self.proj = nn.Parameter(torch.randn(C, J + 1, d) / math.sqrt(J + 1))
        self.chan = nn.Parameter(torch.randn(C, d) * 0.1)

    def forward(self, z, mask):
        """z, mask: [B, T, C] -> per-channel features [B, T, C, d] (zero where unobserved)."""
        steps = torch.sigmoid((z.unsqueeze(-1) - self.theta) / self.log_s.exp())
        f = torch.cat([z.unsqueeze(-1), steps], -1)
        x = torch.einsum('btcj,cjd->btcd', f, self.proj) + self.chan
        return x * mask.unsqueeze(-1)


class RaceIRTS(nn.Module):
    def __init__(self, C, S, d, modes, layers, J, dv, dropout, n_classes):
        super().__init__()
        self.C, self.dv = C, dv
        self.typed = TypedChannelEncoder(C, J, d)
        self.gap = nn.Linear(2, d)
        self.layers = nn.ModuleList(TemporalMemoryLayer(d, modes, dropout) for _ in range(layers))
        self.slot_value = nn.Linear(d, dv)
        self.slot_log_rate = nn.Parameter(torch.logspace(-2, 0.5, dv).log())
        self.static = nn.Linear(S, d)
        # statistic-valued channel slots (THEORY note 59): count, mean, min, max, first, last, trend, staleness
        # v8: + ordering latency (log time of first measurement) and measurement rate per channel
        self.stat_norm = nn.LayerNorm(C * 12)
        self.stat_proj = nn.Linear(C * 12, 2 * d)
        self.head = nn.Sequential(nn.Linear(3 * d + C * (dv + 2) + 2 * d + 2, 2 * d), nn.GELU(), nn.Dropout(dropout),
                                  nn.Linear(2 * d, n_classes))

    def forward(self, t, z, mask, lens, static):
        B, T, C = z.shape
        valid = torch.arange(T).unsqueeze(0) < lens.unsqueeze(1)
        dt = torch.diff(t, dim=1, prepend=t[:, :1]).clamp_min(0) * valid
        per = self.typed(z, mask)                                                    # [B, T, C, d]
        x = per.sum(2) + self.gap(torch.stack([torch.log1p(dt), mask.float().mean(-1)], -1))
        for layer in self.layers:
            x = layer(x, dt) * valid.unsqueeze(-1)
        # addressed channel memory: slot c written only when channel c is observed; decays with elapsed time
        vals = F.softplus(self.slot_value(per))                                      # [B, T, C, dv]
        decay = torch.exp(-self.slot_log_rate.exp() * dt.unsqueeze(-1))              # [B, T, dv]
        slots = x.new_zeros(B, C, self.dv); since = x.new_full((B, C), 48.0); seen = x.new_zeros(B, C)
        cnt = x.new_zeros(B, C); ssum = x.new_zeros(B, C); vmin = x.new_full((B, C), 6.0); vmax = x.new_full((B, C), -6.0)
        first = x.new_zeros(B, C); last = x.new_zeros(B, C); ssq = x.new_zeros(B, C); absd = x.new_zeros(B, C)
        t_first = x.new_full((B, C), -1.0); elapsed = x.new_zeros(B, 1)
        for i in range(T):
            m = mask[:, i].float() * valid[:, i:i + 1].float()
            elapsed = elapsed + dt[:, i:i + 1]
            t_first = torch.where((t_first < 0) & (m > 0), elapsed.expand(B, C), t_first)
            slots = slots * decay[:, i].unsqueeze(1) * (1 - m).unsqueeze(-1) + vals[:, i] * m.unsqueeze(-1)
            since = (since + dt[:, i:i + 1]) * (1 - m)
            zi = z[:, i]
            seen_prev = seen.clone()
            first = torch.where((seen == 0) & (m > 0), zi, first)
            seen = torch.maximum(seen, m)
            absd = absd + (zi - last).abs() * m * seen_prev
            cnt = cnt + m; ssum = ssum + zi * m; ssq = ssq + zi * zi * m
            vmin = torch.where(m > 0, torch.minimum(vmin, zi), vmin); vmax = torch.where(m > 0, torch.maximum(vmax, zi), vmax)
            last = torch.where(m > 0, zi, last)
        mean = ssum / cnt.clamp_min(1)
        vmin = vmin * seen; vmax = vmax * seen
        var = (ssq / cnt.clamp_min(1) - mean ** 2).clamp_min(0)
        mad = absd / (cnt - 1).clamp_min(1)
        latency = torch.where(t_first < 0, elapsed.expand(B, C) + 1.0, t_first)        # never measured: past the record end
        rate = cnt / (elapsed + 1.0)
        stats = torch.stack([torch.log1p(cnt), mean, vmin, vmax, first, last, last - first, torch.log1p(since),
                             torch.sqrt(var + 1e-6), mad, torch.log1p(latency), torch.log1p(rate)], -1)
        stat_feat = F.gelu(self.stat_proj(self.stat_norm(stats.flatten(1))))
        last_h = x[torch.arange(B), (lens - 1).clamp_min(0)]
        mean_h = (x * valid.unsqueeze(-1)).sum(1) / lens.clamp_min(1).unsqueeze(-1)
        span = torch.cat([torch.log1p(elapsed), torch.log1p(lens.float()).unsqueeze(-1)], -1)   # record duration, steps
        feats = torch.cat([last_h, mean_h, self.static(static), slots.flatten(1), torch.log1p(since), seen, stat_feat, span], -1)
        return self.head(feats)


def metrics(y, logits):
    if logits.shape[1] == 1:
        p = torch.sigmoid(logits[:, 0]).numpy()
        return dict(auroc=float(roc_auc_score(y, p)), auprc=float(average_precision_score(y, p)))
    pred = logits.argmax(1).numpy()
    p, r, f, _ = precision_recall_fscore_support(y, pred, average='macro', zero_division=0)
    return dict(acc=float((pred == y).mean()), precision=float(p), recall=float(r), f1=float(f))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True, choices=['P12', 'P19', 'PAM'])
    ap.add_argument('--split', type=int, default=0)
    ap.add_argument('--tag', required=True)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--d', type=int, default=32)
    ap.add_argument('--modes', type=int, default=16)
    ap.add_argument('--layers', type=int, default=2)
    ap.add_argument('--J', type=int, default=4)
    ap.add_argument('--dv', type=int, default=4)
    ap.add_argument('--dropout', type=float, default=0.2)
    ap.add_argument('--lr', type=float, default=2e-3)
    ap.add_argument('--wd', type=float, default=1e-4)
    ap.add_argument('--batch', type=int, default=128)
    ap.add_argument('--epochs', type=int, default=60)
    ap.add_argument('--patience', type=int, default=12)
    ap.add_argument('--score-test', action='store_true')
    ap.add_argument('--ema', type=float, default=0.0, help='select/evaluate an exponential moving average of weights')
    ap.add_argument('--crop', type=float, default=0.0, help='train on random contiguous crops of this fraction of each record (0 = off)')
    ap.add_argument('--jitter', type=float, default=0.0, help='train-time multiplicative amplitude jitter (sd) on observed values')
    a = ap.parse_args()
    torch.set_num_threads(1); random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    D = np.load(ROOT / f'data/raindrop/cache/{a.dataset}.npz')
    tr, va, te = (D[f'split{a.split}_{p}'] for p in ('train', 'val', 'test'))
    vals, mask = D['vals'], D['mask']
    # per-channel normalization from TRAIN observations only (log1p for non-negative heavy-tailed channels)
    obs = mask[tr]; v = vals[tr]
    C = vals.shape[2]
    nonneg = np.array([(v[..., c][obs[..., c]] >= 0).all() for c in range(C)])
    tv = np.where(nonneg, np.log1p(np.clip(vals, 0, None)), vals)
    mu = np.array([tv[tr][..., c][obs[..., c]].mean() if obs[..., c].any() else 0 for c in range(C)])
    sd = np.array([tv[tr][..., c][obs[..., c]].std() + 1e-6 if obs[..., c].any() else 1 for c in range(C)])
    z = np.clip((tv - mu) / sd, -6, 6) * mask
    st = D['static']; smu, ssd = st[tr].mean(0), st[tr].std(0) + 1e-6; stz = (st - smu) / ssd
    t = torch.from_numpy(D['times']); Z = torch.from_numpy(z.astype(np.float32)); M = torch.from_numpy(mask)
    Lens = torch.from_numpy(D['lens']); S = torch.from_numpy(stz.astype(np.float32)); y = D['y']
    n_classes = int(y.max()) + 1
    multi = n_classes > 2
    model = RaceIRTS(C, S.shape[1], a.d, a.modes, a.layers, a.J, a.dv, a.dropout, n_classes if multi else 1)
    sel_key = 'acc' if multi else 'auroc'
    params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
    pos_weight = torch.tensor((y[tr] == 0).sum() / max(1, (y[tr] == 1).sum()), dtype=torch.float32)
    out_dir = ROOT / 'experiments/results/irts'; out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = out_dir / f'{a.tag}.pt'

    def run(idx):
        net = scored(); net.eval(); outs = []
        with torch.no_grad():
            for i in range(0, len(idx), 256):
                b = torch.from_numpy(idx[i:i + 256]); T = int(Lens[b].max())
                outs.append(net(t[b, :T], Z[b, :T], M[b, :T], Lens[b], S[b]))
        return metrics(y[idx], torch.cat(outs))

    ema = torch.optim.swa_utils.AveragedModel(model, multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(a.ema)) if a.ema > 0 else None
    scored = lambda: ema.module if ema is not None else model
    best, best_epoch, history, start = -1.0, -1, [], time.time()
    for epoch in range(a.epochs):
        model.train(); e0 = time.time(); perm = np.random.permutation(tr); tot = 0.0
        for i in range(0, len(perm), a.batch):
            b = torch.from_numpy(perm[i:i + a.batch]); T = int(Lens[b].max())
            tb, zb, mb, lb = t[b, :T], Z[b, :T], M[b, :T], Lens[b]
            if a.crop > 0:                                   # random contiguous crop per record, re-based to start at 0
                Lc = max(2, int(round(a.crop * T)))
                st = torch.randint(0, max(1, T - Lc + 1), (1,)).item()
                tb, zb, mb = tb[:, st:st + Lc], zb[:, st:st + Lc], mb[:, st:st + Lc]
                tb = tb - tb[:, :1]
                lb = (lb - st).clamp(1, Lc)
            if a.jitter > 0:
                zb = zb * (1 + a.jitter * torch.randn(zb.shape[0], 1, zb.shape[2])) * mb
            out = model(tb, zb, mb, lb, S[b])
            yb = torch.from_numpy(y[b.numpy()])
            loss = F.cross_entropy(out, yb) if multi else F.binary_cross_entropy_with_logits(out[:, 0], yb.float(), pos_weight=pos_weight)
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            if ema is not None:
                ema.update_parameters(model)
            tot += loss.item() * len(b)
        vm = run(va)
        history.append(dict(epoch=epoch, train_loss=tot / len(tr), **{f'val_{k}': v for k, v in vm.items()}, epoch_s=time.time() - e0))
        print(json.dumps(history[-1]), flush=True)
        if vm[sel_key] > best:
            best, best_epoch = vm[sel_key], epoch; torch.save(scored().state_dict(), ckpt)
        if epoch - best_epoch >= a.patience:
            break
    scored().load_state_dict(torch.load(ckpt))
    res = dict(status='completed', battle='B2', tag=a.tag, dataset=a.dataset, split=a.split, args=vars(a),
               parameters=params, best_epoch=best_epoch, epochs_run=len(history), wall_s=time.time() - start,
               val=run(va), history=history, checkpoint=str(ckpt.relative_to(ROOT)))
    if a.score_test:
        res['test'] = run(te)
    src = Path(__file__).resolve(); dep = ROOT / 'experiments/tpp/race_tpp_v8.py'
    res['source_sha256'] = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in (src, dep)}
    (out_dir / f'{a.tag}.json').write_text(json.dumps(res, indent=1) + '\n')
    print('VAL', json.dumps(res['val']), flush=True)
    if a.score_test:
        print('TEST', json.dumps(res['test']), flush=True)


if __name__ == '__main__':
    main()
