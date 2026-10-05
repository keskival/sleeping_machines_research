"""Dense and state-space references for the FAS interlaced-event benchmark (experiments/FAS_BENCHMARK.md).

Same data, head, loss, selection and scoring as experiments/fas/native.py: process (non-TICK) events with timestamps,
next-event type cross-entropy plus Gaussian NLL of log(next gap + 10 ms), selection by validation-clean NLL only, prefix
AUROC.  Every model receives the event id and the time since the previous event (log gap); models differ only in the
sequence core:
  lstm         LSTM over [event embedding, log gap]  (the RMTPP input pattern)
  transformer  causal Transformer, event embedding + log gap + sinusoidal encoding of absolute time (the Transformer
               Hawkes Process input pattern; Gaussian gap head instead of an intensity integral)
  lru          stacked diagonal complex linear recurrences (LRU-style), time-invariant, FFT convolution
  s5t          stacked diagonal complex recurrences discretized by each event's real gap, h <- exp(Lambda*dt) h + B x
               (S5-style variable step for irregular sampling), sequential scan
  mamba        minimal selective SSM blocks (Mamba/S6: input-dependent dt, B, C; causal depthwise conv; gating),
               sequential scan
Training work: shape estimate (2 FLOPs per multiply-add, backward = 2x forward, Adam 14/param/step), as
experiments/lm_training_flops.py; stated per model.  Dense neural training is reserved for AWS (AGENTS.md); local use is
limited to tiny smoke checks.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import platform
import sys
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'experiments/fas'))
from baselines import PREFIXES  # noqa: E402
from native import LOG_EPS, V, aurocs, load, nll, prefix_scores, RULES  # noqa: E402

OUT = ROOT / 'experiments/results/fas'      # redirected by experiments/aws_benchmark.py


def tensors(runs, idx):
    T = max(len(runs[j][0]) for j in idx)
    x = torch.zeros(len(idx), T, dtype=torch.long); lg = torch.zeros(len(idx), T); tabs = torch.zeros(len(idx), T)
    dt = torch.zeros(len(idx), T); y = torch.zeros(len(idx), T, dtype=torch.long)
    g = torch.zeros(len(idx), T, dtype=torch.float64); mask = torch.zeros(len(idx), T, dtype=torch.bool)
    for r, j in enumerate(idx):
        ids, t = runs[j]; n = len(ids)
        x[r, :n] = torch.from_numpy(ids); tabs[r, :n] = torch.from_numpy(t - t[0]).float()
        gap_prev = np.diff(t, prepend=t[0]); dt[r, :n] = torch.from_numpy(gap_prev).float()
        lg[r, :n] = torch.from_numpy(np.log(gap_prev + LOG_EPS)).float()
        y[r, :n - 1] = torch.from_numpy(ids[1:]); g[r, :n - 1] = torch.from_numpy(np.log(np.diff(t) + LOG_EPS))
        mask[r, :n - 1] = True
    return x, lg, tabs, dt, y, g, mask


class Inputs(nn.Module):
    def __init__(self, d, time_encoding=False):
        super().__init__()
        self.emb = nn.Embedding(V, d); self.gap = nn.Linear(1, d); self.te = time_encoding
        if time_encoding:
            self.register_buffer('freq', torch.exp(torch.linspace(math.log(1e-3), math.log(1.), d // 2)))
            self.tproj = nn.Linear(d, d)

    def forward(self, x, lg, tabs):
        h = self.emb(x) + self.gap(lg[..., None])
        if self.te:
            ang = tabs[..., None] * self.freq
            h = h + self.tproj(torch.cat([ang.sin(), ang.cos()], -1))
        return h


class LSTMRef(nn.Module):
    def __init__(self, d, layers):
        super().__init__(); self.inp = Inputs(d); self.rnn = nn.LSTM(d, d, layers, batch_first=True); self.head = nn.Linear(d, V + 2)

    def forward(self, x, lg, tabs, dt):
        return self.head(self.rnn(self.inp(x, lg, tabs))[0])


class TransformerRef(nn.Module):
    def __init__(self, d, layers, heads=4):
        super().__init__(); self.inp = Inputs(d, time_encoding=True)
        layer = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout=0., batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False); self.head = nn.Linear(d, V + 2)

    def forward(self, x, lg, tabs, dt):
        T = x.shape[1]; causal = torch.triu(torch.full((T, T), float('-inf')), 1)
        return self.head(self.enc(self.inp(x, lg, tabs), mask=causal, is_causal=True))


class DiagSSM(nn.Module):
    """Diagonal complex SSM layer.  variable=False: time-invariant LRU-style recurrence by FFT convolution;
    variable=True: lambda^dt per event (continuous-time discretization), sequential scan."""
    def __init__(self, d, n, variable):
        super().__init__(); self.variable = variable
        self.nu = nn.Parameter(torch.log(-torch.log(torch.rand(n) * .099 + .9)))       # |lambda| in (0.9, 0.999)
        self.theta = nn.Parameter(torch.rand(n) * 2 * math.pi * .1)
        self.B = nn.Parameter(torch.randn(n, d, dtype=torch.cfloat) / math.sqrt(2 * d))
        self.C = nn.Parameter(torch.randn(d, n, dtype=torch.cfloat) / math.sqrt(n)); self.D = nn.Parameter(torch.randn(d) * .1)

    def forward(self, u, dt):
        Bu = torch.einsum('nd,btd->btn', self.B, u.to(torch.cfloat))
        log_lam = -torch.exp(self.nu) + 1j * self.theta                                   # (n,)
        if not self.variable:
            T = u.shape[1]; k = torch.exp(log_lam[None] * torch.arange(T)[:, None])       # (T, n)
            L = 2 * T
            h = torch.fft.ifft(torch.fft.fft(Bu, L, dim=1) * torch.fft.fft(k, L, dim=0)[None], L, dim=1)[:, :T]
        else:
            a = torch.exp(log_lam[None, None] * dt[..., None].clamp(max=1e3))              # (B, T, n)
            hs = []; h = torch.zeros(u.shape[0], Bu.shape[-1], dtype=torch.cfloat)
            for t in range(u.shape[1]):
                h = a[:, t] * h + Bu[:, t]; hs.append(h)
            h = torch.stack(hs, 1)
        return torch.einsum('dn,btn->btd', self.C, h).real + self.D * u


class SSMRef(nn.Module):
    def __init__(self, d, layers, n=64, variable=False):
        super().__init__(); self.inp = Inputs(d)
        self.ssm = nn.ModuleList(DiagSSM(d, n, variable) for _ in range(layers))
        self.norm = nn.ModuleList(nn.LayerNorm(d) for _ in range(layers))
        self.mlp = nn.ModuleList(nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, d)) for _ in range(layers))
        self.head = nn.Linear(d, V + 2)

    def forward(self, x, lg, tabs, dt):
        h = self.inp(x, lg, tabs)
        for s, nrm, m in zip(self.ssm, self.norm, self.mlp):
            h = h + m(s(nrm(h), dt))
        return self.head(h)


class MambaBlock(nn.Module):
    def __init__(self, d, n=16, expand=2, k=4):
        super().__init__(); e = expand * d; self.n = n
        self.norm = nn.LayerNorm(d); self.inp = nn.Linear(d, 2 * e); self.conv = nn.Conv1d(e, e, k, groups=e, padding=k - 1)
        self.xp = nn.Linear(e, 1 + 2 * n); self.dtp = nn.Linear(1, e)
        self.A_log = nn.Parameter(torch.log(torch.arange(1, n + 1).float()).repeat(e, 1)); self.Dp = nn.Parameter(torch.ones(e))
        self.out = nn.Linear(e, d)

    def forward(self, h):
        Bsz, T, _ = h.shape
        u, z = self.inp(self.norm(h)).chunk(2, -1)
        u = F.silu(self.conv(u.transpose(1, 2))[..., :T].transpose(1, 2))
        p = self.xp(u); delta = F.softplus(self.dtp(p[..., :1])); Bm, Cm = p[..., 1:1 + self.n], p[..., 1 + self.n:]
        A = -torch.exp(self.A_log)                                                          # (e, n)
        s = torch.zeros(Bsz, u.shape[-1], self.n); ys = []
        for t in range(T):
            s = torch.exp(delta[:, t, :, None] * A) * s + delta[:, t, :, None] * Bm[:, t, None, :] * u[:, t, :, None]
            ys.append((s * Cm[:, t, None, :]).sum(-1))
        y = torch.stack(ys, 1) + self.Dp * u
        return h + self.out(y * F.silu(z))


class MambaRef(nn.Module):
    def __init__(self, d, layers):
        super().__init__(); self.inp = Inputs(d); self.blocks = nn.ModuleList(MambaBlock(d) for _ in range(layers))
        self.norm = nn.LayerNorm(d); self.head = nn.Linear(d, V + 2)

    def forward(self, x, lg, tabs, dt):
        h = self.inp(x, lg, tabs)
        for b in self.blocks:
            h = b(h)
        return self.head(self.norm(h))


def build(kind, d, layers):
    return dict(lstm=lambda: LSTMRef(d, layers), transformer=lambda: TransformerRef(d, layers),
                lru=lambda: SSMRef(d, layers), s5t=lambda: SSMRef(d, layers, variable=True),
                mamba=lambda: MambaRef(d, layers))[kind]()


def flops_per_event(model, kind, d, layers, T):
    """forward multiply-add estimate per event (x2 FLOPs), shape-based."""
    matrix = sum(p.numel() for n, p in model.named_parameters() if p.dim() >= 2 and 'emb' not in n)
    extra = layers * 4 * T * d / 2 if kind == 'transformer' else 0          # QK^T and AV over the causal half
    return 2 * (matrix + extra)


@torch.no_grad()
def scores(model, runs, lanes):
    parts = []; total = 0.; count = 0
    model.eval()
    for b in range(0, len(runs), lanes):
        idx = list(range(b, min(b + lanes, len(runs))))
        x, lg, tabs, dt, y, g, mask = tensors(runs, idx)
        pt, pg = nll(model(x, lg, tabs, dt), y, g, mask, parts=True)
        pt, pg = pt.numpy(), pg.numpy(); total += pt.sum() + pg.sum(); count += int(mask.sum())
        parts.append(prefix_scores(pt, pg, [len(runs[j][0]) for j in idx]))
    return {r: np.concatenate([q[r] for q in parts]) for r in RULES}, total / max(count, 1)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', required=True); p.add_argument('--tag', required=True)
    p.add_argument('--model', choices=('lstm', 'transformer', 'lru', 's5t', 'mamba'), required=True)
    p.add_argument('--d', type=int, default=128); p.add_argument('--layers', type=int, default=2)
    p.add_argument('--epochs', type=int, default=3); p.add_argument('--lanes', type=int, default=16)
    p.add_argument('--lr', type=float, default=1e-3); p.add_argument('--clip', type=float, default=1.)
    p.add_argument('--fit-runs', type=int, default=10000); p.add_argument('--max-events', type=int, default=1100)
    p.add_argument('--eval-runs', type=int, default=1000); p.add_argument('--seed', type=int, default=0)
    p.add_argument('--threads', type=int, default=1); p.add_argument('--max-windows', type=int, default=0)
    a = p.parse_args()
    out = Path(OUT) / f'{a.tag}.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    if Path(a.tag).name != a.tag or out.exists():
        raise ValueError('Unique unused tag required')
    torch.set_num_threads(a.threads); torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed); started = time.perf_counter()
    d = ROOT / 'experiments/data/fas' / a.data
    train, _ = load(d / 'train_clean.npz', a.max_events); train = train[:a.fit_runs]
    val_c, _ = load(d / 'val_clean.npz', a.max_events); val_f, val_k = load(d / 'val_faulty.npz', a.max_events)
    val_c, val_f, val_k = val_c[:a.eval_runs], val_f[:a.eval_runs], val_k[:a.eval_runs]
    model = build(a.model, a.d, a.layers); params = sum(q.numel() for q in model.parameters())
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    total = a.epochs * math.ceil(len(train) / a.lanes)
    if a.max_windows:
        total = min(total, a.max_windows)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(opt, total)
    mean_T = float(np.mean([len(r[0]) for r in train]))
    fwd = flops_per_event(model, a.model, a.d, a.layers, mean_T)
    curve = []; best = (math.inf, None, None); w = 0; events = 0
    for epoch in range(1, a.epochs + 1):
        order = rng.permutation(len(train)); t0 = time.perf_counter(); ls = 0.; ns = 0
        for b in range(0, len(order), a.lanes):
            if w >= total:
                break
            x, lg, tabs, dt, y, g, mask = tensors(train, order[b:b + a.lanes])
            model.train(); opt.zero_grad(set_to_none=True)
            loss = nll(model(x, lg, tabs, dt), y, g, mask).sum() / mask.sum()
            loss.float().backward(); nn.utils.clip_grad_norm_(model.parameters(), a.clip); opt.step(); schedule.step()
            w += 1; events += int((x != 0).sum()) + int(mask.sum() > 0); ls += float(loss.detach()) * int(mask.sum()); ns += int(mask.sum())
            if w % 50 == 0:
                print(json.dumps(dict(window=w, of=total, train_nll=float(loss.detach()),
                                      events_per_s=events / (time.perf_counter() - started))), flush=True)
        sc_c, val_nll = scores(model, val_c, 64); sc_f, _ = scores(model, val_f, 64)
        curve.append(dict(epoch=epoch, train_nll=ls / max(ns, 1), val_clean_nll=val_nll, val_auroc=aurocs(sc_c, sc_f, val_k),
                          epoch_s=time.perf_counter() - t0))
        print(json.dumps(curve[-1]), flush=True)
        if val_nll < best[0]:
            best = (val_nll, epoch, {k: v.detach().clone() for k, v in model.state_dict().items()})
        if w >= total:
            break
    model.load_state_dict(best[2])
    test_c, _ = load(d / 'test_clean.npz', a.max_events); test_f, test_k = load(d / 'test_faulty.npz', a.max_events)
    if a.max_windows:
        test_c, test_f, test_k = test_c[:a.eval_runs], test_f[:a.eval_runs], test_k[:a.eval_runs]
    sc_c, test_nll = scores(model, test_c, 64); sc_f, _ = scores(model, test_f, 64)
    result = dict(status='smoke' if a.max_windows else 'completed', args=vars(a), parameters=params, curve=curve,
                  selected_epoch=best[1], selection='validation-clean NLL only', test_clean_nll=test_nll,
                  test_auroc=aurocs(sc_c, sc_f, test_k),
                  work=dict(method='shape estimate', forward_flops_per_event=fwd, inference_flops_per_event=fwd,
                            fitting_events=events, whole_fit_flops_estimate=3 * fwd * events + 14 * params * w),
                  data_manifest_sha256=hashlib.sha256((d / 'manifest.json').read_bytes()).hexdigest(),
                  source_sha256={n: hashlib.sha256((ROOT / n).read_bytes()).hexdigest() for n in
                                 ('experiments/fas/dense.py', 'experiments/fas/native.py', 'experiments/fas/baselines.py')},
                  hardware=dict(platform=platform.platform(), torch=torch.__version__, threads=a.threads),
                  wall_s=time.perf_counter() - started)
    out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(dict(model=a.model, params=params, test_auroc=result['test_auroc'], selected_epoch=best[1])), flush=True)


if __name__ == '__main__':
    main()
