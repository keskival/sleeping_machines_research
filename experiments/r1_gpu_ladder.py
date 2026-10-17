#!/usr/bin/env python3
"""R1 GPU scaling ladder: the plan-B deep keyed temporal-memory flagship, trained one pass over fresh FineWeb tokens on a
CUDA GPU, with the diagnostics needed to find where scaling breaks (experiments/runbooks/GPU_SCALING_LADDER.md).

Model = r1_deep_keyed_lm.DeepKeyedLM: same modules, parameter names and mathematics (checkpoints load into the CPU
drivers and the public-protocol scorer), executed with exact parallel kernels:
- temporal memory: complex decay/rotation per step as a causal kernel, fully parallel over chunks (intra-chunk kernel
  product + chunk-to-chunk carry kernel + carry injection, three batched products; no Python loop over time); fp32.
- per-layer keyed race read: scaled-dot-product attention with the learned per-head distance bias -r_h (t - j) as an
  additive mask (fp32 by default: the bias spans hundreds of nats).
- output keyed token-slot bonus and the 50,257-way softmax: row chunks under activation checkpointing, so [B, T, V]
  logits never exist at once.
Matmuls run in bf16 autocast; normalisation, memory kernel, attention and loss in fp32. --mode contract checks the fast
model against DeepKeyedLM (loss and every gradient, fp32 CPU).

Variants (--variant): flagship (temporal memory + per-layer keyed reads + MLP + output slot bonus); memory_only (no
per-layer reads: the cheapest member on event hardware, O(1) state per step); no_bonus (no output slot bonus); and the
diagnostic read_only (no temporal memory). Unused modules are frozen and excluded from parameter and FLOP counts.

Data: llm.c GPT-2 shards fineweb_train_*.bin (in name order) cut into lanes of --lane tokens from the start of the first
shard; the first train_tokens/lane lanes are visited once in a seeded permutation. Validation windows of 1024 tokens from
fineweb_val_000000: selection slice (16,384 targets from 10,485,760) at log-spaced steps; development slice (65,536 from
20,971,520) and the flagship windows (8,192 from 25,165,824) at milestones and at the end. The public range
[0, 10,485,761) is scored only by --mode public for an explicitly selected run.

Diagnostics (streamed to <tag>.diag.jsonl): loss/grad-norm/LR/throughput/MFU every --log-every steps; loss spikes and
non-finite steps; at log-spaced steps: per-layer residual RMS and sublayer update ratios, per-head attention entropy and
mean attended distance, learned memory/read/bonus decay rates, temporal-memory gate activity, logit-lens loss from every
layer, slot-bonus contribution, per-module weight/gradient/update norms; at milestones: development loss by history
bucket and token role (word-initial / continuation / punctuation / newline).
"""
import argparse, contextlib, glob, hashlib, json, math, os, platform, queue, signal, subprocess, sys, threading, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as ckpt
ROOT = Path(__file__).resolve().parents[1]; sys.path[:0] = [str(ROOT/'experiments'), str(ROOT/'experiments/tpp')]
from r1_deep_keyed_lm import DeepKeyedLM  # noqa: E402

V = 50257
SELECT, DEV, FLAGSHIP, PUBLIC = (10485760, 16384), (20971520, 65536), (25165824, 8192), (0, 10485760)
WINDOW = 1024
VARIANTS = dict(flagship=dict(tm=1, read=1, bonus=1), memory_only=dict(tm=1, read=0, bonus=1), no_bonus=dict(tm=1, read=1, bonus=0), read_only=dict(tm=0, read=1, bonus=1))
PEAK_BF16_TFLOPS = {'RTX PRO 6000 Blackwell Max-Q': 438.9, 'RTX PRO 6000 Blackwell': 503.8, 'H100 80GB HBM3': 989.4, 'H100': 756.0, 'H200': 989.4, 'B200': 2250.0, 'L40S': 362.1, 'A100': 312.0}
EXECUTION_ONLY = ('compile', 'max_steps', 'log_every', 'checkpoint_every', 'mode', 'run', 'peak_tflops', 'eval_batch', 'loss_chunk', 'data_dir', 'vocab', 'prefetch')


# ---------------------------------------------------------------- exact parallel kernels (DeepKeyedLM parameters)
def tm_parallel(tm, u, chunk=64):
    """ParallelTemporalMemory.forward, exact, parallel over chunks: z = K_intra a + C(carry); fp32."""
    with torch.autocast(u.device.type, enabled=False):
        u = u.float(); B, L, _ = u.shape; n = tm.n
        g = torch.sigmoid(tm.gate(u)); w = tm.write(u); wr, wi = w[..., :n]*g, w[..., n:]*g
        C = min(chunk, L); pad = (-L) % C
        if pad: wr, wi = F.pad(wr, (0, 0, 0, pad)), F.pad(wi, (0, 0, 0, pad))
        nC = wr.shape[1]//C; ar, ai = wr.view(B, nC, C, n), wi.view(B, nC, C, n)
        rate, freq = tm.log_rate.exp(), tm.freq; dev = u.device
        k = torch.arange(C, device=dev, dtype=torch.float32); lag = k[:, None] - k[None, :]; lagc = lag.clamp_min(0).unsqueeze(-1)
        mag = torch.exp(-rate*lagc)*(lag >= 0).float().unsqueeze(-1); Kr, Ki = mag*torch.cos(freq*lagc), mag*torch.sin(freq*lagc)        # [C, C, n]
        zr = torch.einsum('ijn,bcjn->bcin', Kr, ar) - torch.einsum('ijn,bcjn->bcin', Ki, ai)
        zi = torch.einsum('ijn,bcjn->bcin', Kr, ai) + torch.einsum('ijn,bcjn->bcin', Ki, ar)
        if nC > 1:                                                     # state entering chunk c: sum_{c'<c} (c^C)^(c-1-c') end_{c'}
            er, ei = zr[:, :, -1], zi[:, :, -1]; cc = torch.arange(nC, device=dev, dtype=torch.float32); m = cc[:, None] - cc[None, :] - 1
            mc = m.clamp_min(0).unsqueeze(-1); Dm = torch.exp(-rate*C*mc)*(m >= 0).float().unsqueeze(-1); Dr, Di = Dm*torch.cos(freq*C*mc), Dm*torch.sin(freq*C*mc)
            sr = torch.einsum('cdn,bdn->bcn', Dr, er) - torch.einsum('cdn,bdn->bcn', Di, ei); si = torch.einsum('cdn,bdn->bcn', Dr, ei) + torch.einsum('cdn,bdn->bcn', Di, er)
            st = (k + 1).unsqueeze(-1); cr, ci = torch.exp(-rate*st)*torch.cos(freq*st), torch.exp(-rate*st)*torch.sin(freq*st)   # c^(i+1), [C, n]
            zr = zr + cr*sr[:, :, None] - ci*si[:, :, None]; zi = zi + cr*si[:, :, None] + ci*sr[:, :, None]
        out = torch.cat([zr, zi], -1).reshape(B, nC*C, 2*n)[:, :L]
        return out, g


def decay_bias(log_rate, L, device):
    """Per-head additive race bias -r_h (t - j) for j <= t, -inf above the diagonal: [H, L, L], fp32."""
    t = torch.arange(L, device=device, dtype=torch.float32); dist = t[:, None] - t[None, :]
    return (-log_rate.exp().view(-1, 1, 1)*dist.clamp_min(0)).masked_fill(dist < 0, float('-inf'))


def read_sdpa(r, x, e, fp32=True, want_probs=False):
    """KeyedRead.forward via scaled-dot-product attention (exact; dk^-1/2 scale is SDPA's default)."""
    B, L, _ = x.shape; prev = torch.cat([torch.zeros_like(e[:, :1]), e[:, :-1]], 1); kv_in = torch.cat([x, prev], -1)
    q = r.q(x).view(B, L, r.h, r.dk).transpose(1, 2); k = r.k(kv_in).view(B, L, r.h, r.dk).transpose(1, 2); v = r.v(kv_in).view(B, L, r.h, r.dk).transpose(1, 2)
    bias = decay_bias(r.log_rate, L, x.device); probs = None
    with (torch.autocast(x.device.type, enabled=False) if fp32 else contextlib.nullcontext()):
        if fp32: q, k, v = q.float(), k.float(), v.float()
        if want_probs:                                                 # diagnostics only: explicit probabilities
            probs = torch.softmax(q @ k.transpose(-1, -2)/math.sqrt(r.dk) + bias, -1); o = probs @ v
        else:
            o = F.scaled_dot_product_attention(q, k, v, attn_mask=bias.to(q.dtype).unsqueeze(0))
    return r.o(o.transpose(1, 2).reshape(B, L, -1)), probs


class FastDeepKeyedLM(nn.Module):
    """DeepKeyedLM with exact GPU kernels and variant switches. state_dict() is DeepKeyedLM's (prefix 'net.')."""
    def __init__(self, d, modes, layers, heads, dk, dropout, out_heads, out_dk, variant, chunk=64, attn_fp32=True):
        super().__init__(); self.net = DeepKeyedLM(d, modes, layers, heads, dk, dropout, out_heads, out_dk)
        self.use = VARIANTS[variant]; self.chunk, self.attn_fp32 = chunk, attn_fp32
        for b in self.net.blocks:
            if not self.use['tm']: b.n1.requires_grad_(False); b.tm.requires_grad_(False); b.tm_out.requires_grad_(False)
            if not self.use['read']: b.n2.requires_grad_(False); b.read.requires_grad_(False)
        if not self.use['bonus']:
            for p in (self.net.key_w, self.net.query_w): p.requires_grad_(False)
            self.net.log_rate.requires_grad_(False)
        self.block_fn = self._block; self.ckpt_loss = True

    def _block(self, b, h, e):
        if self.use['tm']: y, _ = tm_parallel(b.tm, b.n1(h), self.chunk); h = h + b.drop(b.tm_out(y))
        if self.use['read']: h = h + b.drop(read_sdpa(b.read, b.n2(h), e, self.attn_fp32)[0])
        return h + b.drop(b.mlp(b.n3(h)))

    def states(self, x, collect=None):
        n = self.net; e = n.embed(x); h = e
        for i, b in enumerate(n.blocks):
            h = self.block_fn(b, h, e)
            if collect is not None: collect.append(h)
        return n.out_norm(h), e

    def loss_sum(self, h, e, x, y, row_chunk=256, bonus=None):
        """Summed NLL of targets y [B, L] given states h and inputs x, by row chunks under activation checkpointing."""
        n = self.net; B, L, _ = h.shape; use_bonus = self.use['bonus'] if bonus is None else bonus
        if use_bonus:
            prev = torch.cat([torch.zeros_like(e[:, :1]), e[:, :-1]], 1)
            K = n.key_w(torch.cat([h, prev], -1)).view(B, L, n.heads, n.dk).transpose(1, 2).float(); Q = n.query_w(h).view(B, L, n.heads, n.dk).transpose(1, 2).float()
        total = h.new_zeros((), dtype=torch.float32)
        for r0 in range(0, L, row_chunk):
            r1 = min(r0 + row_chunk, L)
            args = (h[:, r0:r1], y[:, r0:r1]) + ((Q[:, :, r0:r1], K[:, :, :r1], x[:, :r1], torch.tensor(r0)) if use_bonus else ())
            total = total + (ckpt.checkpoint(self._chunk_nll, *args, use_reentrant=False) if (torch.is_grad_enabled() and self.ckpt_loss) else self._chunk_nll(*args))
        return total

    @torch.no_grad()
    def nll_rows(self, h, e, x, y, row_chunk=256):
        """Per-target NLL [B, L] (evaluation)."""
        n = self.net; B, L, _ = h.shape; outs = []
        if self.use['bonus']:
            prev = torch.cat([torch.zeros_like(e[:, :1]), e[:, :-1]], 1)
            K = n.key_w(torch.cat([h, prev], -1)).view(B, L, n.heads, n.dk).transpose(1, 2).float(); Q = n.query_w(h).view(B, L, n.heads, n.dk).transpose(1, 2).float()
        for r0 in range(0, L, row_chunk):
            r1 = min(r0 + row_chunk, L); extra = (Q[:, :, r0:r1], K[:, :, :r1], x[:, :r1], torch.tensor(r0)) if self.use['bonus'] else ()
            outs.append(self._chunk_nll(h[:, r0:r1], y[:, r0:r1], *extra, reduction='none').view(B, r1 - r0))
        return torch.cat(outs, 1)

    def _chunk_nll(self, hc, yc, Qc=None, Kc=None, xc=None, r0=None, reduction='sum'):
        n = self.net; logits = (hc @ n.embed.weight.T).float() + n.bias
        if Qc is not None:
            R, J = hc.shape[1], Kc.shape[2]; t = torch.arange(int(r0), int(r0) + R, device=hc.device, dtype=torch.float32)[:, None]; j = torch.arange(J, device=hc.device, dtype=torch.float32)[None, :]
            dist = t - j; decay = torch.exp(-n.log_rate.exp().view(-1, 1, 1)*dist.clamp_min(0))*(dist >= 0).float()          # [H, R, J]
            S = ((Qc @ Kc.transpose(-1, -2))*decay).sum(1)/math.sqrt(n.dk)                                                     # [B, R, J]
            logits = logits.scatter_add(2, xc[:, None, :].expand(-1, R, -1), S)
        return F.cross_entropy(logits.reshape(-1, V), yc.reshape(-1), reduction=reduction)


# ---------------------------------------------------------------- data
class Lanes:
    """The first n lanes of `lane` tokens across fineweb_train_*.bin shards (name order, each shard from offset 0)."""
    def __init__(self, data_dir, lane, n, stride=None):
        """stride < lane gives overlapping lanes (stride = lane - 1: consecutive lanes form one contiguous input stream, for carried memory)."""
        self.lane, self.shards, self.index = lane, [], []; self.stride = stride or lane
        for p in sorted(glob.glob(str(Path(data_dir)/'fineweb_train_*.bin'))):
            hdr = np.fromfile(p, dtype=np.int32, count=256); assert hdr[0] == 20240520, p
            mm = np.memmap(p, dtype=np.uint16, mode='r', offset=1024, shape=(int(hdr[2]),)); k = (len(mm) - lane)//self.stride + 1
            self.shards.append(mm); self.index.append(k)
            if sum(self.index) >= n: break
        self.cum = np.cumsum([0] + self.index); assert self.cum[-1] >= n, f'need {n} lanes, shards hold {self.cum[-1]}: download more shards'
        self.n = n

    def get(self, ids):
        out = np.empty((len(ids), self.lane), dtype=np.int64)
        for r, i in enumerate(ids):
            s = int(np.searchsorted(self.cum, i, side='right') - 1); o = (int(i) - int(self.cum[s]))*self.stride; out[r] = self.shards[s][o:o + self.lane]
        return out


def val_windows(val, start, targets):
    return np.stack([np.asarray(val[p:p + WINDOW + 1], dtype=np.int64) for p in range(start, start + targets, WINDOW)])


# ---------------------------------------------------------------- evaluation and diagnostics
def vocab_roles(path):
    """GPT-2 token roles from encoder.json: 0 word-initial, 1 continuation, 2 punctuation, 3 newline, 4 other."""
    if not path or not Path(path).exists(): return None
    enc = json.loads(Path(path).read_text()); tok = [None]*len(enc)
    for s, i in enc.items(): tok[i] = s
    kind = np.full(len(tok), 4, dtype=np.int8)
    for i, s in enumerate(tok):
        core = s.lstrip('Ġ')
        if 'Ċ' in s: kind[i] = 3
        elif core and all(not c.isalnum() for c in core): kind[i] = 2
        elif s.startswith('Ġ') and core[:1].isalnum(): kind[i] = 0
        elif core[:1].isalnum(): kind[i] = 1
    kind[50256] = 4
    return kind


@torch.no_grad()
def eval_windows(model, wins, dev, batch, per_token=False):
    """Mean NLL over 1024-token windows (state reset at window starts); optionally the per-target NLL [N, 1024]."""
    model.eval(); tot, cnt, rows = 0.0, 0, []
    for i in range(0, len(wins), batch):
        w = torch.from_numpy(wins[i:i + batch]).to(dev); x, y = w[:, :-1], w[:, 1:]
        with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == 'cuda'):
            h, e = model.states(x); nll = model.nll_rows(h, e, x, y).float()
        tot += nll.sum().item(); cnt += nll.numel()
        if per_token: rows.append(nll.cpu())
    model.train()
    return (tot/cnt, torch.cat(rows).numpy()) if per_token else tot/cnt


def breakdown(nll, wins, kind):
    """Development NLL by history length (targets at window offsets) and by token role."""
    pos = np.arange(WINDOW); out = dict(history={})
    for lo, hi in ((0, 16), (16, 64), (64, 256), (256, 1024)): out['history'][f'{lo}-{hi}'] = float(nll[:, lo:hi].mean())
    if kind is not None:
        x, y = wins[:, :-1], wins[:, 1:]; r = kind[y].copy(); r[(r == 1) & (kind[x] == 3)] = 0
        out['roles'] = {name: dict(nll=float(nll[r == i].mean()), share=float((r == i).mean())) for i, name in enumerate(('word_initial', 'continuation', 'punctuation', 'newline', 'other')) if (r == i).any()}
    return out


@torch.no_grad()
def deep_diagnostics(model, wins, dev):
    """Layer-wise structure on a fixed development batch: residual/sublayer RMS, attention entropy and reach, gate
    activity, learned rates, logit-lens loss from every layer, slot-bonus contribution."""
    model.eval(); n = model.net; w = torch.from_numpy(wins).to(dev); x, y = w[:, :-1], w[:, 1:]; e = n.embed(x); h = e; layers = []
    with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == 'cuda'):
        for b in n.blocks:
            row = dict(resid_in_rms=h.float().pow(2).mean().sqrt().item()); base = row['resid_in_rms']
            if model.use['tm']:
                yv, g = tm_parallel(b.tm, b.n1(h), model.chunk); u = b.tm_out(yv); h = h + u
                row.update(tm_update_ratio=u.float().pow(2).mean().sqrt().item()/base, tm_gate_mean=g.mean().item(), tm_gate_frac_gt_0_5=(g > 0.5).float().mean().item(),
                           tm_rate=np.quantile(b.tm.log_rate.exp().float().cpu().numpy(), [0, .5, 1]).tolist(), tm_abs_freq_median=b.tm.freq.abs().median().item())
            if model.use['read']:
                u, p = read_sdpa(b.read, b.n2(h), e, True, want_probs=True); h = h + u
                L = p.shape[-1]; dist = (torch.arange(L, device=dev)[:, None] - torch.arange(L, device=dev)[None, :]).clamp_min(0).float()
                ent = -(p*torch.log(p.clamp_min(1e-12))).sum(-1).mean((0, 2)); reach = (p*dist).sum(-1).mean((0, 2))
                row.update(read_update_ratio=u.float().pow(2).mean().sqrt().item()/base, read_entropy_per_head=ent.tolist(), read_mean_distance_per_head=reach.tolist(),
                           read_rate_per_head=b.read.log_rate.exp().tolist())
            u = b.mlp(b.n3(h)); h = h + u; row['mlp_update_ratio'] = u.float().pow(2).mean().sqrt().item()/base
            row['resid_out_max_abs'] = h.float().abs().max().item(); hn = n.out_norm(h)
            row['logit_lens_nll'] = (model.loss_sum(hn, e, x, y, bonus=False)/y.numel()).item(); layers.append(row)
        hf = n.out_norm(h); out = dict(layers=layers, final_nll_no_bonus=(model.loss_sum(hf, e, x, y, bonus=False)/y.numel()).item())
        if model.use['bonus']:
            out['final_nll'] = (model.loss_sum(hf, e, x, y, bonus=True)/y.numel()).item(); out['bonus_gain'] = out['final_nll_no_bonus'] - out['final_nll']
            out['bonus_rate_per_head'] = n.log_rate.exp().tolist()
    model.train(); return out


def module_norms(model, before=None):
    """Weight, gradient and last-update norms per module group (embed, block i sublayer, output)."""
    groups = {}
    for name, p in model.net.named_parameters():
        if not p.requires_grad: continue
        parts = name.split('.'); key = '.'.join(parts[:3]) if parts[0] == 'blocks' else parts[0]
        g = groups.setdefault(key, dict(w2=0.0, g2=0.0, u2=0.0))
        g['w2'] += p.detach().float().pow(2).sum().item()
        if p.grad is not None: g['g2'] += p.grad.detach().float().pow(2).sum().item()
        if before is not None: g['u2'] += (p.detach().float() - before[name]).pow(2).sum().item()
    return {k: dict(weight=math.sqrt(v['w2']), grad=math.sqrt(v['g2']), update_ratio=math.sqrt(v['u2'])/max(math.sqrt(v['w2']), 1e-12)) for k, v in groups.items()}


def train_flops_per_token(model, lane, dev):
    from torch.utils.flop_counter import FlopCounterMode
    x = torch.randint(0, V, (1, lane), device=dev); model.train(); model.ckpt_loss = False   # model FLOPs: no loss-chunk recompute (CPU ladder convention)
    with FlopCounterMode(display=False) as fc, torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == 'cuda'):
        h, e = model.states(x[:, :-1]); model.loss_sum(h, e, x[:, :-1], x[:, 1:]).backward()
    model.ckpt_loss = True; model.zero_grad(set_to_none=True); return fc.get_total_flops()/(lane - 1)


# ---------------------------------------------------------------- modes
def build(a):
    heads = a.heads or max(4, a.d//64); dk = a.dk or (64 if a.d >= 256 else 16); modes = a.modes or max(32, a.d//2)
    return FastDeepKeyedLM(a.d, modes, a.layers, heads, dk, a.dropout, a.out_heads, a.out_dk, a.variant, a.tm_chunk, bool(a.attn_fp32)), dict(heads=heads, dk=dk, modes=modes)


def contract(a):
    """Fast kernels vs DeepKeyedLM on CPU in fp32: loss and every gradient (flagship, lane not a chunk multiple)."""
    torch.manual_seed(0); res = {}
    for lane, chunk in ((300, 64), (257, 32)):
        m = FastDeepKeyedLM(64, 32, 2, 4, 16, 0.0, 4, 16, 'flagship', chunk); ref = m.net; x = torch.randint(0, V, (2, lane + 1))
        h, e = ref.states(x[:, :-1]); lr = F.cross_entropy(ref.logits(h, e, x[:, :-1], list(range(lane))).reshape(-1, V), x[:, 1:].reshape(-1), reduction='sum'); lr.backward()
        gref = {k: p.grad.clone() for k, p in ref.named_parameters()}; ref.zero_grad()
        h2, e2 = m.states(x[:, :-1]); lf = m.loss_sum(h2, e2, x[:, :-1], x[:, 1:], row_chunk=128); lf.backward()
        scale = max(g.abs().max().item() for g in gref.values())          # key biases have analytically zero gradient (softmax shift invariance): judge them on the global scale
        gd = max(((p.grad - gref[k]).abs().max()/max(gref[k].abs().max().item(), 1e-4*scale)).item() for k, p in ref.named_parameters())
        res[f'lane{lane}_chunk{chunk}'] = dict(loss_ref=lr.item()/(2*lane), loss_fast=lf.item()/(2*lane), loss_abs_diff=abs(lr.item() - lf.item())/(2*lane), grad_max_rel_diff=gd)
    ok = all(v['loss_abs_diff'] < 1e-5 and v['grad_max_rel_diff'] < 1e-3 for v in res.values())
    for variant in ('memory_only', 'no_bonus', 'read_only'):
        m = FastDeepKeyedLM(64, 32, 2, 4, 16, 0.0, 4, 16, variant); x = torch.randint(0, V, (2, 129)); h, e = m.states(x[:, :-1]); l = m.loss_sum(h, e, x[:, :-1], x[:, 1:]); l.backward()
        frozen_grads = [k for k, p in m.net.named_parameters() if not p.requires_grad and p.grad is not None]
        res[variant] = dict(loss=l.item()/256, finite=bool(math.isfinite(l.item())), frozen_with_grad=frozen_grads); ok = ok and res[variant]['finite'] and not frozen_grads
    res['status'] = 'passed' if ok else 'failed'; return res


def main():
    ap = argparse.ArgumentParser(); A = ap.add_argument
    A('--mode', choices=('train', 'contract', 'gpu_check', 'plan', 'public'), default='train'); A('--tag', default=''); A('--run', default='')
    A('--variant', choices=tuple(VARIANTS), default='flagship'); A('--d', type=int, default=256); A('--layers', type=int, default=8)
    A('--heads', type=int, default=0); A('--dk', type=int, default=0); A('--modes', type=int, default=0); A('--out-heads', type=int, default=4); A('--out-dk', type=int, default=32)
    A('--dropout', type=float, default=0.0); A('--tm-chunk', type=int, default=64); A('--attn-fp32', type=int, default=1)
    A('--train-tokens', type=int, default=0); A('--budget-flops', type=float, default=0.0, help='set train tokens from measured FLOPs/token')
    A('--lane', type=int, default=1024); A('--batch', type=int, default=64, help='lanes per micro-batch'); A('--accum', type=int, default=1)
    A('--lr', type=float, default=0.0, help='0: 3e-3 * (256/d)^0.5'); A('--wd', type=float, default=0.1); A('--warmup', type=float, default=0.02); A('--final-lr-frac', type=float, default=0.1)
    A('--clip', type=float, default=1.0); A('--seed', type=int, default=0)
    A('--evals', type=int, default=24, help='log-spaced selection evaluations'); A('--diags', type=int, default=12, help='log-spaced deep diagnostics')
    A('--log-every', type=int, default=50); A('--checkpoint-every', type=int, default=500); A('--max-steps', type=int, default=0)
    A('--compile', type=int, default=1); A('--peak-tflops', type=float, default=0.0); A('--eval-batch', type=int, default=16); A('--loss-chunk', type=int, default=256); A('--prefetch', type=int, default=4)
    A('--data-dir', default=str(ROOT/'data/fineweb_gpt2')); A('--vocab', default=str(ROOT/'data/gpt2/encoder.json')); A('--out-dir', default=str(ROOT/'experiments/results/gpu_ladder'))
    a = ap.parse_args()
    if a.mode == 'contract':
        torch.set_num_threads(min(8, os.cpu_count() or 1)); res = dict(battle='R1', tag=a.tag, **contract(a)); print(json.dumps(res, indent=1))
        if a.tag:
            with (ROOT/f'experiments/results/diagnostics/{a.tag}.json').open('x') as f: json.dump(res, f, indent=1); f.write('\n')
        return
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu'); torch.manual_seed(a.seed)
    if dev.type == 'cuda': torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
    model, shape = build(a); model.to(dev)
    if a.mode == 'plan':
        f = train_flops_per_token(model, a.lane, dev); print(json.dumps(dict(variant=a.variant, d=a.d, layers=a.layers, **shape, lane=a.lane, train_flops_per_token=f,
                              parameters=sum(p.numel() for p in model.parameters() if p.requires_grad), parameters_non_embedding=sum(p.numel() for n_, p in model.named_parameters() if p.requires_grad and 'embed' not in n_)))); return
    if a.mode == 'public':
        return public(a, model, dev)
    if a.mode == 'gpu_check':
        return gpu_check(a, model, dev)
    return train(a, model, shape, dev)


def train(a, model, shape, dev):
    out_dir = Path(a.out_dir); out_dir.mkdir(parents=True, exist_ok=True); out = out_dir/f'{a.tag}.json'; state_path = out_dir/f'{a.tag}.resume.pt'; diag_path = out_dir/f'{a.tag}.diag.jsonl'
    assert a.tag and not out.exists(), 'tag required; never overwrite a completed run'
    gpu = torch.cuda.get_device_name(0) if dev.type == 'cuda' else platform.processor()
    peak = a.peak_tflops or next((v for k, v in PEAK_BF16_TFLOPS.items() if k in gpu), 0.0)
    fpt = train_flops_per_token(model, a.lane, dev); params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    tokens_per_step = a.batch*a.accum*a.lane
    train_tokens = a.train_tokens or int(a.budget_flops/fpt); steps = max(1, train_tokens//tokens_per_step); n_lanes = steps*a.batch*a.accum
    lr = a.lr or 3e-3*math.sqrt(256/a.d); warm = max(1, int(a.warmup*steps))
    data = Lanes(a.data_dir, a.lane, n_lanes); order = np.random.default_rng(a.seed).permutation(n_lanes)
    val = np.memmap(Path(a.data_dir)/'fineweb_val_000000.bin', dtype=np.uint16, mode='r', offset=1024)
    sel_w, dev_w, flag_w = val_windows(val, *SELECT), val_windows(val, *DEV), val_windows(val, *FLAGSHIP); diag_w = dev_w[:8]; kind = vocab_roles(a.vocab)
    decay, other = [p for p in model.parameters() if p.requires_grad and p.dim() >= 2], [p for p in model.parameters() if p.requires_grad and p.dim() < 2]
    opt = torch.optim.AdamW([dict(params=decay, weight_decay=a.wd), dict(params=other, weight_decay=0.0)], lr=lr, betas=(0.9, 0.95), fused=dev.type == 'cuda')
    lr_at = lambda s: lr*min(1.0, (s + 1)/warm)*(a.final_lr_frac + (1 - a.final_lr_frac)*0.5*(1 + math.cos(math.pi*min(1.0, max(0, s - warm)/max(1, steps - warm)))))
    logspace = lambda k: sorted(set(int(round(steps**(i/(k - 1)))) for i in range(k)) | {steps})
    eval_steps, diag_steps = set(logspace(a.evals)), set(logspace(a.diags)); milestone_steps = {max(1, steps//4), max(1, steps//2), steps}
    ident = {k: v for k, v in vars(a).items() if k not in EXECUTION_ONLY}
    st = dict(step=0, losses=[], evals=[], milestones=[], spikes=[], nonfinite=[], clipped=0, wall=0.0, resumes=0)
    if state_path.exists():
        s = torch.load(state_path, map_location=dev, weights_only=False); assert s['ident'] == ident, 'resume with different settings'
        model.net.load_state_dict(s['model']); opt.load_state_dict(s['opt']); torch.set_rng_state(s['rng']); st = s['st']; st['resumes'] += 1
        if dev.type == 'cuda' and s.get('cuda_rng') is not None: torch.cuda.set_rng_state(s['cuda_rng'])
        print(json.dumps(dict(resumed_at_step=st['step'])), flush=True)
    if a.compile and dev.type == 'cuda': model.block_fn = torch.compile(model._block, dynamic=False)
    stop = {'flag': False}; signal.signal(signal.SIGTERM, lambda *_: stop.update(flag=True)); signal.signal(signal.SIGINT, lambda *_: stop.update(flag=True))

    def save():
        tmp = state_path.with_suffix('.tmp'); torch.save(dict(model=model.net.state_dict(), opt=opt.state_dict(), rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state() if dev.type == 'cuda' else None,
                                                            st=st, ident=ident), tmp); os.replace(tmp, state_path)

    def log(rec):
        with diag_path.open('a') as f: f.write(json.dumps(rec) + '\n')

    q = queue.Queue(maxsize=a.prefetch)
    def loader(s0):                                                     # background batch assembly (memmap gather + pinned copy)
        for s in range(s0, steps):
            for m in range(a.accum):
                ids = order[(s*a.accum + m)*a.batch:(s*a.accum + m + 1)*a.batch]; t = torch.from_numpy(data.get(np.sort(ids)))
                q.put(t.pin_memory() if dev.type == 'cuda' else t)
    threading.Thread(target=loader, args=(st['step'],), daemon=True).start()
    log(dict(event='start', step=st['step'], steps=steps, tokens_per_step=tokens_per_step, train_tokens=steps*tokens_per_step, train_flops_per_token=fpt, train_flops=fpt*steps*tokens_per_step,
             parameters=params, shape=shape, lr=lr, gpu=gpu, peak_bf16_tflops=peak, torch=torch.__version__, cuda=torch.version.cuda, args=vars(a)))
    began = time.time() - st['wall']; t_log = time.time(); ema = var = None; before = None
    while st['step'] < steps and not stop['flag']:
        s = st['step']; [g.__setitem__('lr', lr_at(s)) for g in opt.param_groups]; total = 0.0
        if s + 1 in diag_steps: before = {k: p.detach().float().clone() for k, p in model.net.named_parameters() if p.requires_grad}
        for m in range(a.accum):
            w = q.get().to(dev, non_blocking=True); x, y = w[:, :-1], w[:, 1:]
            with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == 'cuda'):
                h, e = model.states(x); loss = model.loss_sum(h, e, x, y, a.loss_chunk)/(y.numel()*a.accum)
            loss.backward(); total += loss.item()
        gn = nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], a.clip).item()
        if not (math.isfinite(total) and math.isfinite(gn)):
            st['nonfinite'].append(dict(step=s, loss=total, grad_norm=gn)); opt.zero_grad(set_to_none=True); log(dict(event='nonfinite', step=s, loss=total, grad_norm=gn))
            if len(st['nonfinite']) > 20: raise RuntimeError('repeated non-finite steps; see diag log')
            st['step'] += 1; continue
        st['clipped'] += gn > a.clip; opt.step(); st['step'] += 1; s1 = st['step']; st['losses'].append(total)
        if ema is not None and total > ema + 5*math.sqrt(var) + 0.05: st['spikes'].append(dict(step=s1, loss=total, ema=ema)); log(dict(event='spike', step=s1, loss=total, ema=ema))
        ema = total if ema is None else 0.98*ema + 0.02*total; var = 0.01 if var is None else 0.98*var + 0.02*(total - ema)**2
        if s1 in diag_steps:
            log(dict(event='diag', step=s1, tokens=s1*tokens_per_step, modules=module_norms(model, before), grad_norm=gn, **deep_diagnostics(model, diag_w, dev))); before = None
        opt.zero_grad(set_to_none=True)
        if s1 % a.log_every == 0 or s1 == steps:
            dt = time.time() - t_log; tps = a.log_every*tokens_per_step/dt if s1 % a.log_every == 0 else float('nan'); t_log = time.time()
            rec = dict(event='log', step=s1, of=steps, tokens=s1*tokens_per_step, loss=float(np.mean(st['losses'][-a.log_every:])), lr=lr_at(s1 - 1), grad_norm=gn, clipped_frac=st['clipped']/s1,
                       tokens_per_s=tps, mfu=fpt*tps/(peak*1e12) if peak else None, gpu_mem_gb=torch.cuda.max_memory_allocated()/2**30 if dev.type == 'cuda' else None, elapsed_s=round(time.time() - began))
            log(rec); print(json.dumps(rec), flush=True)
        if s1 in eval_steps:
            st['evals'].append(dict(step=s1, tokens=s1*tokens_per_step, flops=fpt*s1*tokens_per_step, select=eval_windows(model, sel_w, dev, a.eval_batch), train_loss_recent=float(np.mean(st['losses'][-max(1, a.log_every):]))))
            log(dict(event='eval', **st['evals'][-1])); print(json.dumps(st['evals'][-1]), flush=True)
        if s1 in milestone_steps:
            dn, per = eval_windows(model, dev_w, dev, a.eval_batch, per_token=True)
            st['milestones'].append(dict(step=s1, tokens=s1*tokens_per_step, flops=fpt*s1*tokens_per_step, dev=dn, **breakdown(per, dev_w, kind))); log(dict(event='milestone', **st['milestones'][-1]))
        if s1 % a.checkpoint_every == 0 or s1 == steps: st['wall'] = time.time() - began; save()
        if a.max_steps and s1 >= a.max_steps: break
    st['wall'] = time.time() - began; save()
    if st['step'] < steps:
        print(json.dumps(dict(paused_at_step=st['step'], resumable=str(state_path))), flush=True); sys.exit(75 if stop['flag'] else 0)
    torch.save(model.net.state_dict(), out_dir/f'{a.tag}.pt'); n8 = max(1, steps//8)
    result = dict(status='completed', battle='R1', stage='GPU scaling ladder: plan-B deep keyed member, one pass', tag=a.tag, variant=a.variant, args=vars(a), shape=shape, parameters=params,
                  parameters_non_embedding=sum(p.numel() for n_, p in model.named_parameters() if p.requires_grad and 'embed' not in n_), train_flops_per_token=fpt,
                  train_tokens=steps*tokens_per_step, train_flops=fpt*steps*tokens_per_step, steps=steps, lr=lr, train_wall_s=st['wall'], resumes=st['resumes'], gpu=gpu, peak_bf16_tflops=peak,
                  mean_mfu=fpt*steps*tokens_per_step/st['wall']/(peak*1e12) if peak else None, evals=st['evals'], milestones=st['milestones'], spikes=st['spikes'], nonfinite=st['nonfinite'],
                  clipped_frac=st['clipped']/steps, prequential_by_eighth=[float(np.mean(st['losses'][i:i + n8])) for i in range(0, steps, n8)],
                  select_w1024=eval_windows(model, sel_w, dev, a.eval_batch), dev_w1024=st['milestones'][-1]['dev'], flagship_w1024=eval_windows(model, flag_w, dev, a.eval_batch),
                  checkpoint=str((out_dir/f'{a.tag}.pt').relative_to(ROOT)) if str(out_dir).startswith(str(ROOT)) else str(out_dir/f'{a.tag}.pt'),
                  git_commit=subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, capture_output=True, text=True).stdout.strip(),
                  source_sha256={n_: hashlib.sha256((ROOT/n_).read_bytes()).hexdigest() for n_ in ('experiments/r1_gpu_ladder.py', 'experiments/r1_deep_keyed_lm.py')})
    out.write_text(json.dumps(result, indent=1) + '\n'); state_path.unlink()
    print('RESULT', json.dumps({k: result[k] for k in ('select_w1024', 'dev_w1024', 'flagship_w1024', 'train_flops', 'parameters', 'mean_mfu')}), flush=True)


def gpu_check(a, model, dev):
    """On the worker GPU: fast kernels in fp32 and bf16 (eager and compiled) against the CPU fp32 DeepKeyedLM reference at
    the intended shape, loss and gradient agreement, plus throughput of one training step."""
    torch.manual_seed(0); x = torch.randint(0, V, (2, a.lane + 1)); res = dict(gpu=torch.cuda.get_device_name(0) if dev.type == 'cuda' else 'cpu', d=a.d, layers=a.layers, variant=a.variant)
    sd = {k: v.detach().cpu() for k, v in model.net.state_dict().items()}; refm, _ = build(a); refm.net.load_state_dict(sd); refm = refm.float()
    h, e = refm.net.states(x[:, :-1]); lr = F.cross_entropy(refm.net.logits(h, e, x[:, :-1], list(range(a.lane))).reshape(-1, V), x[:, 1:].reshape(-1)) if model.use == VARIANTS['flagship'] else None
    if lr is not None: lr.backward(); gref = {k: p.grad.clone() for k, p in refm.net.named_parameters() if p.grad is not None}
    xd = x.to(dev)
    for name, amp, comp in (('fp32_eager', False, False), ('bf16_eager', True, False), ('bf16_compiled', True, True)):
        model.zero_grad(set_to_none=True); model.block_fn = torch.compile(model._block, dynamic=False) if comp else model._block
        with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=amp):
            h2, e2 = model.states(xd[:, :-1]); l2 = model.loss_sum(h2, e2, xd[:, :-1], xd[:, 1:])/(2*a.lane)
        l2.backward(); row = dict(loss=l2.item())
        if lr is not None:
            row['loss_abs_diff'] = abs(l2.item() - lr.item()); scale = max(g.abs().max().item() for g in gref.values())
            row['grad_max_rel_diff'] = max(((p.grad.float().cpu() - gref[k]).abs().max()/max(gref[k].abs().max().item(), 1e-4*scale)).item() for k, p in model.net.named_parameters() if k in gref)
        res[name] = row; print(json.dumps({name: row}), flush=True)
    model.block_fn = model._block; xb = torch.randint(0, V, (a.batch, a.lane + 1), device=dev); opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4, fused=dev.type == 'cuda')
    for i in range(6):
        if i == 2 and dev.type == 'cuda': torch.cuda.synchronize(); t0 = time.time()
        with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == 'cuda'):
            h3, e3 = model.states(xb[:, :-1]); l3 = model.loss_sum(h3, e3, xb[:, :-1], xb[:, 1:])/xb[:, 1:].numel()
        l3.backward(); opt.step(); opt.zero_grad(set_to_none=True)
    if dev.type == 'cuda': torch.cuda.synchronize(); res['eager_tokens_per_s'] = 4*a.batch*a.lane/(time.time() - t0); res['gpu_mem_gb'] = torch.cuda.max_memory_allocated()/2**30
    res['status'] = 'passed' if lr is None or (res['fp32_eager']['loss_abs_diff'] < 1e-4 and res['fp32_eager']['grad_max_rel_diff'] < 1e-2 and res['bf16_eager']['loss_abs_diff'] < 3e-2 and res['bf16_compiled']['loss_abs_diff'] < 3e-2) else 'failed'
    print(json.dumps(res), flush=True)
    if a.tag:
        with (Path(a.out_dir)/f'{a.tag}.json').open('x') as f: json.dump(res, f, indent=1); f.write('\n')


def public(a, model, dev):
    """Sealed public protocol (targets 1..10,485,760 in 1024-token windows) for one explicitly selected run; never overwrites."""
    rec = json.loads((Path(a.out_dir)/f'{a.run}.json').read_text()); assert rec['status'] == 'completed'
    path = Path(a.out_dir)/f'{a.run}_public.json'; assert not path.exists(), 'public protocol already scored for this run'
    model.net.load_state_dict(torch.load(ROOT/rec['checkpoint'] if not Path(rec['checkpoint']).is_absolute() else rec['checkpoint'], map_location=dev))
    val = np.memmap(Path(a.data_dir)/'fineweb_val_000000.bin', dtype=np.uint16, mode='r', offset=1024); t0 = time.time()
    nll = eval_windows(model, val_windows(val, *PUBLIC), dev, a.eval_batch)
    res = dict(status='completed', battle='R1', run=a.run, public_w1024=dict(nll=nll, targets=PUBLIC[1], window=WINDOW, target_start=1), train_flops=rec['train_flops'], train_tokens=rec['train_tokens'],
               reference='modded-nanogpt record curves (experiments/results/public_reference/published_record_curves_20261016_v1.json)', wall_s=time.time() - t0)
    with path.open('x') as f: json.dump(res, f, indent=1); f.write('\n')
    print(json.dumps(res), flush=True)


if __name__ == '__main__':
    main()
