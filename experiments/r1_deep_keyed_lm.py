#!/usr/bin/env python3
"""R1 depth plan B: a deep keyed temporal-memory language model built to scale with layer count.

Same data, lanes, selection and 65,528-target scoring protocol as r1_token_keyed_lm.py (the strongest native language
member, 5.601/5.624 at 4M tokens). Changes, each aimed at a measured depth failure of the P24 race core:
- Pre-norm residual blocks x <- x + a(read(TM(LN x))) and x <- x + a(MLP(LN x)): depth cannot overwrite what earlier
  layers computed (the race core is a strict serial chain whose content cut costs the same 1.56 nats at any stage).
  Output projections start at 1/sqrt(2 * depth) of their usual scale (depth-scaled residual initialisation).
- Per-layer keyed predecessor read (race attention over past events): each layer's query races decayed keys built from
  each past event's state and its predecessor's message, values from the same pair; every layer gets addressed access to
  history (O(depth) credit paths to any earlier event instead of one serial path).
- Full backpropagation over each 256-token lane (the P24 core's 16-token credit window makes depth useless for long range).
- Exact parallel temporal memory: at unit time steps each mode's decay/rotation is constant, so the recurrence is a
  causal Toeplitz kernel applied in one matrix product (contract: equals the step-by-step recurrence).
The output keyed token-slot bonus of the strongest member is kept.
"""
import argparse, hashlib, json, math, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT/'experiments')); sys.path.insert(0, str(ROOT/'experiments/tpp'))
from r1_token_keyed_lm import load, evaluate as evaluate_keyed, lane_nll, V  # noqa: E402


class ParallelTemporalMemory(nn.Module):
    """Complex-diagonal memory that decays and rotates per unit step, written by gated messages; exact causal kernel."""
    def __init__(self, d, modes):
        super().__init__(); self.n = modes
        self.log_rate = nn.Parameter(torch.logspace(-2, 1, modes).log()); self.freq = nn.Parameter(torch.randn(modes)*0.5)
        self.write = nn.Linear(d, 2*modes); self.gate = nn.Linear(d, modes)

    CHUNK = 256

    def forward(self, u):
        """Exact causal kernel computed in chunks of CHUNK steps with the complex state carried across chunk boundaries."""
        B, L, _ = u.shape; g = torch.sigmoid(self.gate(u)); w = self.write(u); wr, wi = w[..., :self.n]*g, w[..., self.n:]*g
        rate, freq = self.log_rate.exp(), self.freq; C = min(self.CHUNK, L)
        k = torch.arange(C, dtype=u.dtype); lag = k[:, None] - k[None, :]; causal = (lag >= 0).to(u.dtype); lagc = lag.clamp_min(0).unsqueeze(-1)
        mag = torch.exp(-rate*lagc)*causal.unsqueeze(-1); ang = freq*lagc; kr, ki = mag*torch.cos(ang), mag*torch.sin(ang)   # [C, C, n]
        steps = (k + 1).unsqueeze(-1); cr, ci = torch.exp(-rate*steps)*torch.cos(freq*steps), torch.exp(-rate*steps)*torch.sin(freq*steps)  # c^(i+1)
        sr = u.new_zeros(B, self.n); si = u.new_zeros(B, self.n); outs = []
        for c0 in range(0, L, C):
            ar, ai = wr[:, c0:c0 + C], wi[:, c0:c0 + C]; n_ = ar.shape[1]; Kr, Ki = kr[:n_, :n_], ki[:n_, :n_]
            zr = torch.einsum('ijn,bjn->bin', Kr, ar) - torch.einsum('ijn,bjn->bin', Ki, ai); zi = torch.einsum('ijn,bjn->bin', Kr, ai) + torch.einsum('ijn,bjn->bin', Ki, ar)
            zr = zr + cr[:n_]*sr.unsqueeze(1) - ci[:n_]*si.unsqueeze(1); zi = zi + cr[:n_]*si.unsqueeze(1) + ci[:n_]*sr.unsqueeze(1)
            outs.append(torch.cat([zr, zi], -1)); sr, si = zr[:, -1], zi[:, -1]
        return torch.cat(outs, 1)


class KeyedRead(nn.Module):
    """Race attention: per head, softmax over past events j <= t of q_t.k_j/sqrt(dk) - r_h (t - j); keys/values from [h_j; e(x_{j-1})]."""
    def __init__(self, d, heads, dk):
        super().__init__(); self.h, self.dk = heads, dk
        self.q = nn.Linear(d, heads*dk); self.k = nn.Linear(2*d, heads*dk); self.v = nn.Linear(2*d, heads*dk); self.o = nn.Linear(heads*dk, d)
        self.log_rate = nn.Parameter(torch.linspace(math.log(1e-3), math.log(0.3), heads))

    def forward(self, x, e):
        B, L, _ = x.shape; prev = torch.cat([torch.zeros_like(e[:, :1]), e[:, :-1]], 1); kv_in = torch.cat([x, prev], -1)
        q = self.q(x).view(B, L, self.h, self.dk).transpose(1, 2); k = self.k(kv_in).view(B, L, self.h, self.dk).transpose(1, 2); v = self.v(kv_in).view(B, L, self.h, self.dk).transpose(1, 2)
        out = []
        for r0 in range(0, L, 512):                                                 # exact: query rows in blocks, all causal keys
            rows = torch.arange(r0, min(r0 + 512, L)); dist = (rows[:, None] - torch.arange(L)[None, :]).to(x.dtype)
            bias = -self.log_rate.exp().view(1, -1, 1, 1)*dist.clamp_min(0); bias = bias.masked_fill(dist < 0, float('-inf'))
            a = torch.softmax(q[:, :, rows] @ k.transpose(-1, -2)/math.sqrt(self.dk) + bias, -1); out.append(a @ v)
        return self.o(torch.cat(out, 2).transpose(1, 2).reshape(B, L, -1))


class Block(nn.Module):
    def __init__(self, d, modes, heads, dk, dropout, depth):
        super().__init__()
        self.n1, self.n2, self.n3 = nn.LayerNorm(d), nn.LayerNorm(d), nn.LayerNorm(d)
        self.tm = ParallelTemporalMemory(d, modes); self.tm_out = nn.Linear(2*modes, d)
        self.read = KeyedRead(d, heads, dk)
        self.mlp = nn.Sequential(nn.Linear(d, 4*d), nn.GELU(), nn.Linear(4*d, d)); self.drop = nn.Dropout(dropout)
        s = 1/math.sqrt(2*depth)
        with torch.no_grad():
            for lin in (self.tm_out, self.read.o, self.mlp[2]): lin.weight.mul_(s); lin.bias.zero_()

    def forward(self, x, e):
        x = x + self.drop(self.tm_out(self.tm(self.n1(x))))
        x = x + self.drop(self.read(self.n2(x), e))
        return x + self.drop(self.mlp(self.n3(x)))


class DeepKeyedLM(nn.Module):
    def __init__(self, d, modes, layers, heads, dk, dropout, out_heads=4, out_dk=16):
        super().__init__(); self.keyed, self.heads, self.dk = 1, out_heads, out_dk
        self.embed = nn.Embedding(V, d); nn.init.normal_(self.embed.weight, std=0.02)
        self.blocks = nn.ModuleList(Block(d, modes, heads, dk, dropout, layers) for _ in range(layers))
        self.out_norm = nn.LayerNorm(d); self.bias = nn.Parameter(torch.zeros(V))
        self.key_w = nn.Linear(2*d, out_heads*out_dk); self.query_w = nn.Linear(d, out_heads*out_dk)
        self.log_rate = nn.Parameter(torch.linspace(math.log(1e-3), math.log(0.3), out_heads))

    checkpoint_blocks = False

    def states(self, x):
        e = self.embed(x); h = e
        for b in self.blocks:
            # Exact activation checkpointing: identical gradients, activations recomputed in backward (memory bound).
            h = torch.utils.checkpoint.checkpoint(b, h, e, use_reentrant=False) if (self.checkpoint_blocks and self.training) else b(h, e)
        return self.out_norm(h), e

    # Output keyed token-slot bonus and logits: identical to the strongest member.
    from r1_token_keyed_lm import KeyedTokenLM as _K
    keyed_scores = _K.keyed_scores; logits = _K.logits


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--tag', required=True); ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--train-tokens', type=int, default=1048576); ap.add_argument('--lane', type=int, default=256); ap.add_argument('--d', type=int, default=64)
    ap.add_argument('--modes', type=int, default=32); ap.add_argument('--layers', type=int, default=2); ap.add_argument('--heads', type=int, default=4); ap.add_argument('--dk', type=int, default=16)
    ap.add_argument('--dropout', type=float, default=0.1); ap.add_argument('--lr', type=float, default=3e-3); ap.add_argument('--wd', type=float, default=0.01)
    ap.add_argument('--batch', type=int, default=4); ap.add_argument('--epochs', type=int, default=20); ap.add_argument('--patience', type=int, default=2)
    ap.add_argument('--max-wall-s', type=float, default=10800); ap.add_argument('--max-steps', type=int, default=0); ap.add_argument('--score', type=int, default=1); ap.add_argument('--checkpoint-blocks', type=int, default=0)
    a = ap.parse_args(); torch.set_num_threads(1); torch.manual_seed(a.seed); np.random.seed(a.seed)
    train_all = load(ROOT/'data/fineweb_gpt2/fineweb_train_000001.bin'); val = load(ROOT/'data/fineweb_gpt2/fineweb_val_000000.bin')
    n_lanes = a.train_tokens//a.lane; train = np.asarray(train_all[:n_lanes*a.lane]).reshape(n_lanes, a.lane)
    model = DeepKeyedLM(a.d, a.modes, a.layers, a.heads, a.dk, a.dropout); model.checkpoint_blocks = bool(a.checkpoint_blocks); params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
    total = math.ceil(n_lanes/a.batch)*a.epochs; sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=total, pct_start=0.05)
    out_dir = ROOT/'experiments/results/token_language'; ckpt = out_dir/f'{a.tag}.pt'; out = out_dir/f'{a.tag}.json'; assert not out.exists()
    rng = np.random.default_rng(a.seed); best, best_epoch, history, start, steps = math.inf, -1, [], time.time(), 0
    for epoch in range(a.epochs):
        model.train(); e0 = time.time(); tl = tn = 0.0; perm = rng.permutation(n_lanes)
        for bi, i in enumerate(range(0, n_lanes, a.batch)):
            x = torch.from_numpy(train[np.sort(perm[i:i + a.batch])].astype(np.int64)); h, e = model.states(x[:, :-1])
            lg = model.logits(h, e, x[:, :-1], list(range(a.lane - 1))); loss = F.cross_entropy(lg.reshape(-1, V), x[:, 1:].reshape(-1))
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step(); steps += 1
            tl += loss.item()*x[:, 1:].numel(); tn += x[:, 1:].numel()
            if bi % 100 == 0: print(json.dumps(dict(epoch=epoch, step=bi, train_nll=tl/tn, elapsed_s=round(time.time() - start))), flush=True)
            if time.time() - start > a.max_wall_s or (a.max_steps and steps >= a.max_steps): break
        sel = evaluate_keyed(model, val, 10485760, 8, 2048); history.append(dict(epoch=epoch, train_nll=tl/tn, select_nll=sel['nll'], epoch_s=round(time.time() - e0))); print(json.dumps(history[-1]), flush=True)
        if sel['nll'] < best: best, best_epoch = sel['nll'], epoch; torch.save(model.state_dict(), ckpt)
        if epoch - best_epoch >= a.patience or time.time() - start > a.max_wall_s or (a.max_steps and steps >= a.max_steps): break
    model.load_state_dict(torch.load(ckpt))
    result = dict(status='completed', battle='R1', stage='depth plan B: deep keyed temporal-memory ladder', tag=a.tag, args=vars(a), parameters=params, best_epoch=best_epoch,
                  epochs_run=len(history), steps=steps, wall_s=time.time() - start, history=history, select=evaluate_keyed(model, val, 10485760, 8, 2048), checkpoint=str(ckpt.relative_to(ROOT)),
                  reference=dict(keyed_L2_1M_score_65528=6.019141, keyed_L2_4M_score_65528=5.521061, note='r1_token_keyed_lm seed 0 on the same 65,528-target slice'))
    if a.score: result['score_65528'] = evaluate_keyed(model, val, 20971520, 8, 8192)
    result['source_sha256'] = {n: hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in ['experiments/r1_deep_keyed_lm.py', 'experiments/r1_token_keyed_lm.py', 'experiments/tpp/race_tpp_v5.py']}
    out.write_text(json.dumps(result, indent=1) + '\n'); print('RESULT', json.dumps({k: result.get(k) for k in ('select', 'score_65528', 'parameters', 'best_epoch')}), flush=True)


if __name__ == '__main__':
    main()
