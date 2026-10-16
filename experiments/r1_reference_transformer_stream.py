#!/usr/bin/env python3
"""R1 matched-protocol Transformer reference (user authorization 16 Oct 2026; experiments/SCALING_SUPREMACY_PROTOCOL.md step 3).

A small decoder-only GPT (pre-LayerNorm, learned positions, causal scaled-dot-product attention, GELU MLP 4x, tied
input/output embedding, GPT-2 initialisation with residual projections scaled by 1/sqrt(2L), no dropout: one pass) trained
exactly like the native one-pass ladder (r1_deep_keyed_stream.py): the same first --train-tokens of fineweb_train_000001,
visited once in a seeded permutation, 1024 tokens per update, AdamW with a one-cycle schedule, exact resume. Its natural
context is a 1024-token sequence (batch 1 x 1024) where the native member trains on 4 x 256-token lanes; --seq/--batch can
match the native shape instead. The model exposes states()/logits() like the native members, so the same scorers apply:
selection and development slices in disjoint 1024-token windows (public-protocol scorer), the flagship protocol, and
measured training FLOPs per token.
"""
import argparse, hashlib, json, math, os, resource, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.flop_counter import FlopCounterMode
ROOT = Path(__file__).resolve().parents[1]; sys.path[:0] = [str(ROOT/'experiments'), str(ROOT/'experiments/tpp'), str(ROOT/'analysis')]
from r1_token_keyed_lm import load, V  # noqa: E402
import r1_deep_keyed_stream as stream  # noqa: E402
from public_protocol_score_keyed import score as window_score  # noqa: E402

SELECT, DEV = (10485760, 16384), (20971520, 65536)      # (start, targets) in 1024-token windows; public range [0, 10485761) stays sealed


class Block(nn.Module):
    def __init__(self, d, heads):
        super().__init__(); self.heads = heads; self.n1 = nn.LayerNorm(d); self.qkv = nn.Linear(d, 3*d); self.proj = nn.Linear(d, d)
        self.n2 = nn.LayerNorm(d); self.fc = nn.Linear(d, 4*d); self.out = nn.Linear(4*d, d)

    def forward(self, x):
        B, T, D = x.shape; q, k, v = self.qkv(self.n1(x)).view(B, T, 3, self.heads, D//self.heads).permute(2, 0, 3, 1, 4)
        x = x + self.proj(F.scaled_dot_product_attention(q, k, v, is_causal=True).transpose(1, 2).reshape(B, T, D))
        return x + self.out(F.gelu(self.fc(self.n2(x))))


class ReferenceGPT(nn.Module):
    def __init__(self, d, layers, heads, ctx=1024):
        super().__init__(); self.wte = nn.Embedding(V, d); self.wpe = nn.Embedding(ctx, d)
        self.blocks = nn.ModuleList(Block(d, heads) for _ in range(layers)); self.nf = nn.LayerNorm(d)
        for n, p in self.named_parameters():
            if p.dim() == 2: nn.init.normal_(p, std=0.02/math.sqrt(2*layers) if n.endswith(('proj.weight', 'out.weight')) else 0.02)
            elif n.endswith('bias'): nn.init.zeros_(p)

    def states(self, x):
        h = self.wte(x) + self.wpe(torch.arange(x.shape[1]))
        for b in self.blocks: h = b(h)
        return self.nf(h), None

    def logits(self, h, e, x, rows):
        return h[:, rows] @ self.wte.weight.T


def train_flops_per_target(d, layers, heads, seq):
    m = ReferenceGPT(d, layers, heads); x = torch.randint(0, V, (1, seq))
    with FlopCounterMode(display=False) as fc:
        h, _ = m.states(x[:, :-1]); F.cross_entropy(m.logits(h, None, None, slice(None)).reshape(-1, V), x[:, 1:].reshape(-1)).backward()
    return fc.get_total_flops()/(seq - 1)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--tag', required=True); ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--train-tokens', type=int, default=4194304); ap.add_argument('--seq', type=int, default=1024); ap.add_argument('--batch', type=int, default=1)
    ap.add_argument('--d', type=int, default=64); ap.add_argument('--layers', type=int, default=8); ap.add_argument('--heads', type=int, default=4)
    ap.add_argument('--lr', type=float, default=3e-3); ap.add_argument('--wd', type=float, default=0.1); ap.add_argument('--checkpoint-every', type=int, default=256)
    ap.add_argument('--max-steps', type=int, default=0, help='smoke only: stop after this many updates (no scoring)')
    a = ap.parse_args(); torch.set_num_threads(1); torch.manual_seed(a.seed)
    train_all = load(ROOT/'data/fineweb_gpt2/fineweb_train_000001.bin'); val = load(ROOT/'data/fineweb_gpt2/fineweb_val_000000.bin')
    n_seq = a.train_tokens//a.seq; train = np.asarray(train_all[:n_seq*a.seq]).reshape(n_seq, a.seq)
    order = np.random.default_rng(a.seed).permutation(n_seq); total = math.ceil(n_seq/a.batch); quarter = math.ceil(total/4)
    model = ReferenceGPT(a.d, a.layers, a.heads); params = sum(p.numel() for p in model.parameters())
    decay = [p for p in model.parameters() if p.dim() == 2]; other = [p for p in model.parameters() if p.dim() < 2]
    opt = torch.optim.AdamW([dict(params=decay, weight_decay=a.wd), dict(params=other, weight_decay=0.0)], lr=a.lr, betas=(0.9, 0.95))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=total, pct_start=0.05)
    out_dir = ROOT/'experiments/results/token_language'; out = out_dir/f'{a.tag}.json'; state_path = out_dir/f'{a.tag}.resume.pt'; assert not out.exists()
    step, losses, milestones, wall, resumes = 0, [], [], 0.0, 0
    if state_path.exists():
        s = torch.load(state_path, weights_only=False); model.load_state_dict(s['model']); opt.load_state_dict(s['opt']); sched.load_state_dict(s['sched'])
        torch.set_rng_state(s['torch_rng']); step, losses, milestones, wall, resumes = s['step'], s['losses'], s['milestones'], s['wall'], s['resumes'] + 1
        assert s['args'] == {k: v for k, v in vars(a).items() if k != 'max_steps'}, 'resume with different settings'
        print(json.dumps(dict(resumed_at_step=step)), flush=True)
    began = time.time() - wall
    def save():
        tmp = state_path.with_suffix('.tmp'); torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(), torch_rng=torch.get_rng_state(), step=step,
                   losses=losses, milestones=milestones, wall=time.time() - began, resumes=resumes, args={k: v for k, v in vars(a).items() if k != 'max_steps'}), tmp); os.replace(tmp, state_path)
    while step < total:
        model.train(); x = torch.from_numpy(train[np.sort(order[step*a.batch:(step + 1)*a.batch])].astype(np.int64)); h, _ = model.states(x[:, :-1])
        loss = F.cross_entropy(model.logits(h, None, None, slice(None)).reshape(-1, V), x[:, 1:].reshape(-1))
        opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step(); step += 1
        losses.append(loss.item())                                    # fresh sequences: loss before this update is held out
        if step % 100 == 0: print(json.dumps(dict(step=step, of=total, prequential_last100=float(np.mean(losses[-100:])), elapsed_s=round(time.time() - began))), flush=True)
        if step % quarter == 0 or step == total:
            milestones.append(dict(step=step, tokens=step*a.batch*a.seq, select_w1024=window_score(model, val, *SELECT, 2)['nll'], prequential_segment=float(np.mean(losses[-quarter:])), elapsed_s=round(time.time() - began)))
            print(json.dumps(milestones[-1]), flush=True)
        if step % a.checkpoint_every == 0 or step == total: save()
        if a.max_steps and step >= a.max_steps:
            print(json.dumps(dict(smoke_steps=step, s_per_step=(time.time() - began)/step, max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)), flush=True); return
    seg = max(1, total//8)
    result = dict(status='completed', battle='R1', stage='matched-protocol Transformer reference, one pass', tag=a.tag, args=vars(a), parameters=params, steps=step,
                  presentations=step*a.batch*(a.seq - 1), train_wall_s=time.time() - began, resumes=resumes, milestones=milestones,
                  prequential_by_eighth=[float(np.mean(losses[i:i + seg])) for i in range(0, len(losses), seg)], prequential_last_quarter=float(np.mean(losses[-quarter:])),
                  select_w1024=window_score(model, val, *SELECT, 2), dev_w1024=window_score(model, val, *DEV, 2), flagship_protocol=stream.flagship(model, val),
                  train_flops_per_target=train_flops_per_target(a.d, a.layers, a.heads, a.seq), max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    torch.save(model.state_dict(), out_dir/f'{a.tag}.pt'); result['checkpoint'] = f'experiments/results/token_language/{a.tag}.pt'
    result['source_sha256'] = {n: hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in ['experiments/r1_reference_transformer_stream.py', 'analysis/public_protocol_score_keyed.py', 'experiments/r1_deep_keyed_stream.py']}
    out.write_text(json.dumps(result, indent=1) + '\n'); state_path.unlink()
    print('RESULT', json.dumps({k: result[k] for k in ('select_w1024', 'dev_w1024', 'flagship_protocol', 'prequential_last_quarter', 'parameters', 'train_flops_per_target')}), flush=True)


if __name__ == '__main__':
    main()
