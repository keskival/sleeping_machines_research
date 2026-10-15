#!/usr/bin/env python3
"""R1 decision depth: race-tree output cascade on the deep keyed temporal-memory stack (race-graph instance, theory note
DEPTH_PLAN_C_ASYNC_HIERARCHY.md section 6).

The next token is the winner of four successive 16-way races down a TRAIN-built vocabulary tree
(build_vocab_race_tree.py; siblings are distributionally similar tokens). Round k scores the 16 children of the node
chosen so far against a query that has been refined by every earlier choice (q_{k+1} = LN(q_k + W_k e(child))), so later
rounds decide with more specific information. An exact copy race (keyed scores over tokens already in the lane) competes
with the tree through a learned gate: p(y) = (1 - g) p_tree(y) + g p_copy(y), exactly normalised. Training maximises the
exact path likelihood, so every losing branch at every round receives its exact gradient (closed-form counterfactual
credit). Output work per token: 4 x 16 child scores plus 3 refinements, versus 50,257 dense dot products.
Same data, lanes, selection and 65,528-target scoring protocol as the keyed member and plan B.
"""
import argparse, hashlib, json, math, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT/'experiments')); sys.path.insert(0, str(ROOT/'experiments/tpp'))
from r1_token_keyed_lm import load, V, KeyedTokenLM  # noqa: E402
from r1_deep_keyed_lm import DeepKeyedLM  # noqa: E402

TREE = 'experiments/results/token_language/vocab_race_tree_b16d4_20261015_v1.npz'
BR, DEPTH = 16, 4


class RaceTreeHead(nn.Module):
    def __init__(self, d, path):
        super().__init__()
        self.register_buffer('path', torch.as_tensor(path, dtype=torch.long))                     # [V, DEPTH]
        node = torch.zeros(V, DEPTH, dtype=torch.long)                                             # parent node id at each round
        for k in range(1, DEPTH): node[:, k] = node[:, k - 1]*BR + self.path[:, k - 1]
        self.register_buffer('node', node)
        self.keys = nn.ModuleList(nn.Embedding(BR**(k + 1), d) for k in range(DEPTH)); self.bias = nn.ParameterList(nn.Parameter(torch.zeros(BR**(k + 1))) for k in range(DEPTH))
        for e in self.keys: nn.init.normal_(e.weight, std=d**-.5)
        self.refine = nn.ModuleList(nn.Embedding(BR**(k + 1), d) for k in range(DEPTH - 1)); self.norms = nn.ModuleList(nn.LayerNorm(d) for _ in range(DEPTH - 1))
        for e in self.refine: nn.init.normal_(e.weight, std=0.02)
        valid = [torch.zeros(BR**(k + 1), dtype=torch.bool) for k in range(DEPTH)]                 # children that lead to at least one token
        for k in range(DEPTH): valid[k][self.node[:, k]*BR + self.path[:, k]] = True
        for k in range(DEPTH): self.register_buffer(f'valid{k}', valid[k])

    def log_prob(self, h, y):
        """h: [N, d] states, y: [N] targets -> exact log p_tree(y)."""
        q = h; total = h.new_zeros(h.shape[0])
        for k in range(DEPTH):
            parent = self.node[y, k]; kids = parent.unsqueeze(-1)*BR + torch.arange(BR)                # [N, 16]
            logits = (self.keys[k](kids)*q.unsqueeze(1)).sum(-1) + self.bias[k][kids]
            logits = logits.masked_fill(~getattr(self, f'valid{k}')[kids], -1e9)
            total = total + F.log_softmax(logits, -1).gather(-1, self.path[y, k].unsqueeze(-1)).squeeze(-1)
            if k < DEPTH - 1: q = self.norms[k](q + self.refine[k](kids.gather(-1, self.path[y, k].unsqueeze(-1)).squeeze(-1)))
        return total


class RaceTreeLM(nn.Module):
    def __init__(self, d, modes, layers, heads, dk, dropout, path):
        super().__init__(); self.body = DeepKeyedLM(d, modes, layers, heads, dk, dropout); self.head = RaceTreeHead(d, path)
        self.copy_gate = nn.Linear(d, 1); nn.init.zeros_(self.copy_gate.weight); nn.init.constant_(self.copy_gate.bias, -3.0)
        self.keyed, self.heads, self.dk = 1, self.body.heads, self.body.dk

    @property
    def checkpoint_blocks(self): return self.body.checkpoint_blocks

    @checkpoint_blocks.setter
    def checkpoint_blocks(self, v): self.body.checkpoint_blocks = v

    def target_log_prob(self, x_in, y, rows, states=None):
        """x_in: [B, L] inputs, y: [B, L] next tokens; rows: positions to score; states: optional precomputed (h, e).
        Returns [B, len(rows)] exact log p(y)."""
        h, e = states if states is not None else self.body.states(x_in); hr = h[:, rows]; B = h.shape[0]
        lt = self.head.log_prob(hr.reshape(-1, hr.shape[-1]), y[:, rows].reshape(-1)).view(B, len(rows))
        S = KeyedTokenLM.keyed_scores(self.body, h, e, rows)                                         # [B, T, L] causal copy scores
        L = h.shape[1]; causal = torch.as_tensor(rows)[:, None] >= torch.arange(L)[None, :]
        a = torch.softmax(S.masked_fill(~causal, -1e9), -1); hit = (x_in.unsqueeze(1) == y[:, rows].unsqueeze(-1)) & causal
        pc = (a*hit).sum(-1); g = torch.sigmoid(self.copy_gate(hr)).squeeze(-1)
        return torch.logaddexp(torch.log1p(-g) + lt, torch.log(g) + torch.log(pc.clamp_min(1e-30)))


def evaluate(model, data, offset, lanes, lane_tokens, chunk=256):
    model.eval(); block = np.asarray(data[offset:offset + lanes*lane_tokens]).reshape(lanes, lane_tokens); tot = n = 0.
    with torch.no_grad():
        for i in range(lanes):
            x = torch.from_numpy(block[i:i + 1].astype(np.int64)); L = x.shape[1] - 1; st = model.body.states(x[:, :-1])
            for c0 in range(0, L, chunk):
                rows = list(range(c0, min(c0 + chunk, L))); lp = model.target_log_prob(x[:, :-1], x[:, 1:], rows, st); tot -= float(lp.sum()); n += lp.numel()
    return dict(nll=tot/n, targets=int(n))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--tag', required=True); ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--train-tokens', type=int, default=1048576); ap.add_argument('--lane', type=int, default=256); ap.add_argument('--d', type=int, default=64)
    ap.add_argument('--modes', type=int, default=32); ap.add_argument('--layers', type=int, default=2); ap.add_argument('--heads', type=int, default=4); ap.add_argument('--dk', type=int, default=16)
    ap.add_argument('--dropout', type=float, default=0.1); ap.add_argument('--lr', type=float, default=3e-3); ap.add_argument('--wd', type=float, default=0.01)
    ap.add_argument('--batch', type=int, default=4); ap.add_argument('--epochs', type=int, default=20); ap.add_argument('--patience', type=int, default=2)
    ap.add_argument('--max-wall-s', type=float, default=10800); ap.add_argument('--max-steps', type=int, default=0); ap.add_argument('--score', type=int, default=1); ap.add_argument('--checkpoint-blocks', type=int, default=0)
    a = ap.parse_args(); torch.set_num_threads(1); torch.manual_seed(a.seed); np.random.seed(a.seed)
    path = np.load(ROOT/TREE)['path']
    train_all = load(ROOT/'data/fineweb_gpt2/fineweb_train_000001.bin'); val = load(ROOT/'data/fineweb_gpt2/fineweb_val_000000.bin')
    n_lanes = a.train_tokens//a.lane; train = np.asarray(train_all[:n_lanes*a.lane]).reshape(n_lanes, a.lane)
    model = RaceTreeLM(a.d, a.modes, a.layers, a.heads, a.dk, a.dropout, path); model.checkpoint_blocks = bool(a.checkpoint_blocks); params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd); total = math.ceil(n_lanes/a.batch)*a.epochs
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=total, pct_start=0.05)
    out_dir = ROOT/'experiments/results/token_language'; ckpt = out_dir/f'{a.tag}.pt'; out = out_dir/f'{a.tag}.json'; assert not out.exists()
    rng = np.random.default_rng(a.seed); best, best_epoch, history, start, steps = math.inf, -1, [], time.time(), 0
    for epoch in range(a.epochs):
        model.train(); e0 = time.time(); tl = tn = 0.0; order = rng.permutation(n_lanes)
        for bi, i in enumerate(range(0, n_lanes, a.batch)):
            x = torch.from_numpy(train[np.sort(order[i:i + a.batch])].astype(np.int64))
            lp = model.target_log_prob(x[:, :-1], x[:, 1:], list(range(a.lane - 1))); loss = -lp.mean()
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step(); steps += 1
            tl += loss.item()*lp.numel(); tn += lp.numel()
            if bi % 100 == 0: print(json.dumps(dict(epoch=epoch, step=bi, train_nll=tl/tn, elapsed_s=round(time.time() - start))), flush=True)
            if time.time() - start > a.max_wall_s or (a.max_steps and steps >= a.max_steps): break
        sel = evaluate(model, val, 10485760, 8, 2048); history.append(dict(epoch=epoch, train_nll=tl/tn, select_nll=sel['nll'], epoch_s=round(time.time() - e0))); print(json.dumps(history[-1]), flush=True)
        if sel['nll'] < best: best, best_epoch = sel['nll'], epoch; torch.save(model.state_dict(), ckpt)
        if epoch - best_epoch >= a.patience or time.time() - start > a.max_wall_s or (a.max_steps and steps >= a.max_steps): break
    model.load_state_dict(torch.load(ckpt)); d = a.d
    result = dict(status='completed', battle='R1', stage='decision depth: race-tree output cascade', tag=a.tag, args=vars(a), parameters=params, best_epoch=best_epoch, epochs_run=len(history),
                  steps=steps, wall_s=time.time() - start, history=history, select=evaluate(model, val, 10485760, 8, 2048), checkpoint=str(ckpt.relative_to(ROOT)),
                  output_work_per_token=dict(race_tree_child_scores=DEPTH*BR, race_tree_macs=DEPTH*BR*d + (DEPTH - 1)*d, dense_head_macs=V*d),
                  reference=dict(keyed_L2_1M_score_65528=6.019141, note='dense-head plan B with the same backbone: aws_r1_deep_keyed_L{layers}_1M_s0_20261015_v3'))
    if a.score: result['score_65528'] = evaluate(model, val, 20971520, 8, 8192)
    result['source_sha256'] = {n: hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in ['experiments/r1_race_tree_lm.py', 'experiments/r1_deep_keyed_lm.py', 'experiments/r1_token_keyed_lm.py', 'experiments/tpp/race_tpp_v5.py', TREE]}
    out.write_text(json.dumps(result, indent=1) + '\n'); print('RESULT', json.dumps({k: result.get(k) for k in ('select', 'score_65528', 'parameters', 'best_epoch')}), flush=True)


if __name__ == '__main__':
    main()
