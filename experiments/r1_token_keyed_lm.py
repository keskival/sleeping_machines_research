#!/usr/bin/env python3
"""R1 gate 2: a compact temporal-memory token model with a keyed predecessor-message read, scored against KN references.

Data and protocol (token_ngram_reference.py, AWS reference aws_token_ngram_reference_20261006T143000Z):
GPT-2 BPE FineWeb shards (llm.c format). TRAIN = the first N tokens of fineweb_train_000001, cut into lanes of 256
tokens (255 next-token targets per lane, as in the native stages). SCORING slice = 8 lanes x 8,192 tokens from offset
20,971,520 of fineweb_val_000000 (65,528 targets; each target conditions on its lane's history), scored ONCE with the
selected checkpoint. SELECTION slice = 8 lanes x 2,048 tokens from offset 10,485,760 of the same val shard (disjoint).
Units: nats per token. Gate: below KN trigram at the same N (1M: 6.537; 4M: 6.100).

Model (events at unit time steps):
- token embedding (tied with the output), two TemporalMemoryLayers from race_tpp_v5 (complex-diagonal memory that decays
  and rotates with elapsed time, written by gated messages) -> state h_t;
- keyed predecessor-message read (R1 gate-1 mechanism, experiments/R1_RECALL.md): the event at position j writes, to the
  slot of its own token x_j, a key k_j = W_k [h_j ; e(x_{j-1})] (its state and the message of the event before it);
  per head, keys decay with distance at a learned rate. The query q_t = W_q h_t scores every written key, and each
  token's logit gains the summed scores of its slot: bonus_v(t) = sum_h sum_{j<=t, x_j=v} exp(-r_h (t-j)) q_t,h . k_j,h.
  Writes are sparse (one slot per event); the read touches only slots of tokens present in the lane.
- `--keyed 0` removes the read (ablation); `--keyed 2` learns the read by local race credit only (state and messages
  detached at the read, as in recall_tpp_v2).
"""
import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'experiments/tpp'))
from race_tpp_v5 import TemporalMemoryLayer  # noqa: E402

V = 50257


def load(path):
    header = np.fromfile(path, dtype=np.int32, count=256)
    assert header[0] == 20240520, 'not an llm.c token shard'
    return np.memmap(path, dtype=np.uint16, mode='r', offset=1024)


class KeyedTokenLM(nn.Module):
    def __init__(self, d, modes, layers, heads, dk, keyed, dropout):
        super().__init__()
        self.keyed, self.heads, self.dk = keyed, heads, dk
        self.embed = nn.Embedding(V, d)
        nn.init.normal_(self.embed.weight, std=0.02)
        self.layers = nn.ModuleList(TemporalMemoryLayer(d, modes, dropout) for _ in range(layers))
        self.out_norm = nn.LayerNorm(d)
        self.bias = nn.Parameter(torch.zeros(V))
        if keyed:
            self.key_w = nn.Linear(2 * d, heads * dk)
            self.query_w = nn.Linear(d, heads * dk)
            self.log_rate = nn.Parameter(torch.linspace(math.log(1e-3), math.log(0.3), heads))

    def states(self, x):
        e = self.embed(x)
        dt = torch.ones(x.shape, dtype=e.dtype)
        h = e
        for layer in self.layers:
            h = layer(h, dt)
        return self.out_norm(h), e

    def keyed_scores(self, h, e, rows):
        """Scores S[b, t, j] (t in rows, all j <= t) summed over heads with per-head distance decay."""
        B, L, _ = h.shape
        hk, ek = (h.detach(), e.detach()) if self.keyed == 2 else (h, e)
        prev = torch.cat([torch.zeros_like(ek[:, :1]), ek[:, :-1]], 1)
        k = self.key_w(torch.cat([hk, prev], -1)).view(B, L, self.heads, self.dk)
        q = self.query_w(hk[:, rows]).view(B, len(rows), self.heads, self.dk)
        dist = (torch.as_tensor(rows)[:, None] - torch.arange(L)[None, :]).to(h.dtype)    # [T, L]
        causal = dist >= 0
        S = h.new_zeros(B, len(rows), L)
        for hd in range(self.heads):
            decay = torch.where(causal, torch.exp(-self.log_rate[hd].exp() * dist.clamp_min(0)), torch.zeros_like(dist))
            S = S + torch.einsum('btd,bjd->btj', q[:, :, hd], k[:, :, hd]) * decay / math.sqrt(self.dk)
        return S

    def logits(self, h, e, x, rows):
        out = h[:, rows] @ self.embed.weight.T + self.bias                                  # [B, T, V]
        if self.keyed:
            S = self.keyed_scores(h, e, rows)
            out = out.scatter_add(2, x[:, None, :].expand(-1, len(rows), -1), S)
        return out


def lane_nll(model, lanes, chunk):
    """Total NLL and count over lanes [B, L]; every target conditions on its lane's history."""
    x = torch.from_numpy(lanes.astype(np.int64))
    h, e = model.states(x[:, :-1])
    tot, n = 0.0, 0
    L = x.shape[1] - 1
    for c0 in range(0, L, chunk):
        rows = list(range(c0, min(c0 + chunk, L)))
        lg = model.logits(h, e, x[:, :-1], rows)
        tot += F.cross_entropy(lg.reshape(-1, V), x[:, 1:][:, rows].reshape(-1), reduction='sum').item()
        n += lg.shape[0] * lg.shape[1]
    return tot, n


def evaluate(model, data, offset, lanes, lane_tokens, chunk=256):
    model.eval()
    block = np.asarray(data[offset:offset + lanes * lane_tokens]).reshape(lanes, lane_tokens)
    with torch.no_grad():
        tot = n = 0
        for i in range(lanes):                                                              # one lane at a time
            t_, n_ = lane_nll(model, block[i:i + 1], chunk)
            tot += t_; n += n_
    return dict(nll=tot / n, targets=n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', required=True)
    ap.add_argument('--keyed', type=int, default=1)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--train-tokens', type=int, default=1048576)
    ap.add_argument('--lane', type=int, default=256)
    ap.add_argument('--d', type=int, default=64)
    ap.add_argument('--modes', type=int, default=32)
    ap.add_argument('--layers', type=int, default=2)
    ap.add_argument('--heads', type=int, default=4)
    ap.add_argument('--dk', type=int, default=16)
    ap.add_argument('--dropout', type=float, default=0.1)
    ap.add_argument('--lr', type=float, default=3e-3)
    ap.add_argument('--wd', type=float, default=0.01)
    ap.add_argument('--batch', type=int, default=4)
    ap.add_argument('--epochs', type=int, default=20)
    ap.add_argument('--patience', type=int, default=2)
    ap.add_argument('--max-wall-s', type=float, default=10800)
    ap.add_argument('--score', type=int, default=1, help='score the 65,528-target slice once at the end')
    ap.add_argument('--threads', type=int, default=1)
    a = ap.parse_args()
    torch.set_num_threads(a.threads); torch.manual_seed(a.seed); np.random.seed(a.seed)
    train_all = load(ROOT / 'data/fineweb_gpt2/fineweb_train_000001.bin')
    val = load(ROOT / 'data/fineweb_gpt2/fineweb_val_000000.bin')
    n_lanes = a.train_tokens // a.lane
    train = np.asarray(train_all[:n_lanes * a.lane]).reshape(n_lanes, a.lane)
    model = KeyedTokenLM(a.d, a.modes, a.layers, a.heads, a.dk, a.keyed, a.dropout)
    params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
    steps_per_epoch = math.ceil(n_lanes / a.batch); total = steps_per_epoch * a.epochs
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=total, pct_start=0.05)
    out_dir = ROOT / 'experiments/results/token_language'; out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = out_dir / f'{a.tag}.pt'
    rng = np.random.default_rng(a.seed)
    best, best_epoch, history, start = math.inf, -1, [], time.time()
    for epoch in range(a.epochs):
        model.train(); e0 = time.time(); tl = tn = 0.0
        perm = rng.permutation(n_lanes)
        for bi, i in enumerate(range(0, n_lanes, a.batch)):
            lanes = train[np.sort(perm[i:i + a.batch])]
            x = torch.from_numpy(lanes.astype(np.int64))
            h, e = model.states(x[:, :-1])
            lg = model.logits(h, e, x[:, :-1], list(range(a.lane - 1)))
            loss = F.cross_entropy(lg.reshape(-1, V), x[:, 1:].reshape(-1))
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step()
            tl += loss.item() * x[:, 1:].numel(); tn += x[:, 1:].numel()
            if bi % 100 == 0:
                print(json.dumps(dict(epoch=epoch, step=bi, train_nll=tl / tn, elapsed_s=round(time.time() - start))), flush=True)
            if time.time() - start > a.max_wall_s:
                break
        sel = evaluate(model, val, 10485760, 8, 2048)
        history.append(dict(epoch=epoch, train_nll=tl / tn, select_nll=sel['nll'], epoch_s=round(time.time() - e0)))
        print(json.dumps(history[-1]), flush=True)
        if sel['nll'] < best:
            best, best_epoch = sel['nll'], epoch; torch.save(model.state_dict(), ckpt)
        if epoch - best_epoch >= a.patience or time.time() - start > a.max_wall_s:
            break
    model.load_state_dict(torch.load(ckpt))
    result = dict(status='completed', battle='R1', gate=2,
                  decision='does the keyed predecessor-message read carry over to tokenized language; KN trigram at the '
                           'same training tokens is the gate (1M: 6.537, 4M: 6.100 on the 65,528-target slice)',
                  tag=a.tag, args=vars(a), parameters=params, best_epoch=best_epoch, epochs_run=len(history),
                  wall_s=time.time() - start, history=history, select=evaluate(model, val, 10485760, 8, 2048),
                  checkpoint=str(ckpt.relative_to(ROOT)),
                  references=dict(kn_trigram={'65536': 7.551, '262144': 7.049, '1048576': 6.537, '4194304': 6.100},
                                  kn_bigram={'65536': 7.544, '262144': 7.058, '1048576': 6.584, '4194304': 6.209},
                                  source='experiments/results/token_language/aws_token_ngram_reference_20261006T143000Z.json'))
    if a.score:
        result['score_65528'] = evaluate(model, val, 20971520, 8, 8192)
    srcs = [Path(__file__).resolve(), ROOT / 'experiments/tpp/race_tpp_v5.py']
    result['source_sha256'] = {str(s.relative_to(ROOT)): hashlib.sha256(s.read_bytes()).hexdigest() for s in srcs}
    (out_dir / f'{a.tag}.json').write_text(json.dumps(result, indent=1) + '\n')
    print('RESULT', json.dumps({k: result.get(k) for k in ('select', 'score_65528', 'parameters', 'best_epoch')}), flush=True)


if __name__ == '__main__':
    main()
