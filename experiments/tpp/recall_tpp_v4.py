#!/usr/bin/env python3
"""R1 gate: multi-query associative recall with irregular gaps, on the B1 race-of-clocks event model.

Task (MQAR with irregular time): marks 0..Kk-1 are keys, Kk..Kk+Kv-1 values. A sequence shows n_pairs (key, value) pairs
(distinct keys, random values); the value follows its key after a short gap, pairs are separated by heavy-tailed gaps
spanning about three decades. Then every key is queried once in random order, each query again followed by its value.
The scored recall events are the values after query keys: chance accuracy is 1/Kv; only binding the earlier pair can do
better. The whole stream is scored as a marked temporal point process (time and mark log-likelihood, exact survival).

Models (THEORY: separate keys and values; sparse addressed writes; query-dependent race scores):
- `--keyed 0`: the frozen B1 model (race_tpp_v5.RaceTPP), unchanged. Its addressed mark memory has one slot per mark,
  read by a fixed per-clock vector, so a stored slot cannot be compared with the current event.
- `--keyed 1`: adds a keyed mark memory. When mark k occurs, its key slot absorbs W_k h (h is the event's state, which
  carries the preceding events through the temporal memory); slots decay with learned elapsed-time rates. Each event
  issues a query W_q h, and every mark's score in every clock's mark race gains q · key_k / sqrt(dk). Writes stay sparse
  (only the occurring mark's slot); the read scores all K slots (K small here; candidate selection is future work).
- `--keyed 2` (v2): the same keyed memory learned by **local race credit only**. The state h entering the keyed memory is
  detached, so no gradient from the read reaches the temporal memory, embeddings or earlier layers. The read's credit is
  the race's own error at the realized event: for each clock, its posterior responsibility for the event times
  (1[k = target] − p_k) for every mark k (the winner and every losing mark, weighted by its probability); W_q learns from that error times the stored keys, and W_k from that error times the
  query and each slot's eligibility trace (the elapsed-time-decayed sum of the h written to that slot). Every factor is
  available at the slot and the read: a three-factor rule with time-decaying traces. Autograd computes exactly this
  rule because the detached graph contains nothing else; the rest of the model trains on its own likelihood as in v5.

v4 task correction: v1–v3 queried every key exactly once, so the last queries could be answered by elimination (the value
not yet repeated); the v1 plateau (38.6–38.9%) matched that shortcut (recall rose from ~20% to 100% with query position)
and the preceding key was not decodable from the written state (3.9% vs 3.1% chance). v4 draws the queries with
replacement, so elimination carries no information, and reports the set baseline: a uniform guess among the values shown
in the context. Binding is shown only by recall above the set baseline.

v3 options (addressing the v1/v2 recall plateau; chosen by the recall diagnosis):
- `--prev-msg 1`: the key written at event i is built from [h_i ; e(m_{i-1})], the event's state and the message of the
  event before it (the mark embedding of the predecessor). An event model's analogue of an induction head's
  previous-token step: binding "value after key" needs the predecessor's identity explicitly, not diluted in a decaying
  sum of all past events.
- `--qk-norm 1`: written keys and queries are unit vectors and the score is exp(s) q·k with learned sharpness s, so a
  match can dominate the race regardless of state norms.
"""
import argparse
import hashlib
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from race_tpp_v5 import RaceTPP

ROOT = Path(__file__).resolve().parents[2]


def make_split(n_seq, n_pairs, Kk, Kv, rng):
    """Returns times [N, 4n], marks [N, 4n], recall mask [N, 4n] (True at values that answer a query), and the set
    baseline [N, 4n] (1 / number of distinct context values, at recall positions)."""
    L = 4 * n_pairs
    T = np.zeros((n_seq, L)); M = np.zeros((n_seq, L), np.int64); R = np.zeros((n_seq, L), bool); B = np.zeros((n_seq, L))
    for s in range(n_seq):
        keys = rng.choice(Kk, n_pairs, replace=False); vals = rng.integers(0, Kv, n_pairs)
        order = list(range(n_pairs)) + list(rng.integers(0, n_pairs, n_pairs))          # queries with replacement
        t = 0.0
        for j, p in enumerate(order):
            t += math.exp(rng.uniform(math.log(0.05), math.log(50.0))) if j else 0.0   # irregular inter-pair gap
            T[s, 2 * j], M[s, 2 * j] = t, keys[p]
            t += math.exp(rng.normal(math.log(0.2), 0.5))                               # key -> value gap
            T[s, 2 * j + 1], M[s, 2 * j + 1] = t, Kk + vals[p]
            if j >= n_pairs:
                R[s, 2 * j + 1] = True; B[s, 2 * j + 1] = 1.0 / len(set(vals.tolist()))
    return torch.from_numpy(T), torch.from_numpy(M), torch.from_numpy(R), torch.from_numpy(B)


class KeyedRaceTPP(RaceTPP):
    def __init__(self, *args, dk=16, local=False, prev_msg=False, qk_norm=False, **kw):
        super().__init__(*args, **kw)
        self.local, self.prev_msg, self.qk_norm = local, prev_msg, qk_norm
        d = self.embed.embedding_dim
        self.dk = dk
        self.key_w = nn.Linear(2 * d if prev_msg else d, dk)
        self.log_sharp = nn.Parameter(torch.tensor(math.log(4.0)))
        self.query_w = nn.Linear(d, dk)
        self.key_log_rate = nn.Parameter(torch.logspace(-3, 0, dk).log())

    def encode(self, t, m, mask):
        x, slots = super().encode(t, m, mask)
        xk = x.detach() if self.local else x                                  # local credit: no gradient into h
        dt = torch.diff(t, dim=1, prepend=t[:, :1]).clamp_min(0) / self.scale
        kin = xk
        if self.prev_msg:
            prev = torch.cat([torch.zeros_like(xk[:, :1]), self.embed(m[:, :-1])], 1)
            kin = torch.cat([xk, prev.detach() if self.local else prev], -1)
        k = self.key_w(kin)
        if self.qk_norm:
            k = F.normalize(k, dim=-1)
        decay = torch.exp(-self.key_log_rate.exp() * dt.unsqueeze(-1))
        onehot = F.one_hot(m, self.K).to(x.dtype)
        B, L, _ = x.shape
        s = x.new_zeros(B, self.K, self.dk); keyed = []
        for i in range(L):
            s = s * decay[:, i].unsqueeze(1) + onehot[:, i].unsqueeze(-1) * k[:, i].unsqueeze(1)
            keyed.append(s)
        self._keyed = torch.stack(keyed, 1)                                   # [B, L, K, dk], after event i
        q = self.query_w(xk)
        self._query = F.normalize(q, dim=-1) * self.log_sharp.exp() if self.qk_norm else q   # [B, L, dk]
        return x, slots

    def clocks(self, h, slots):
        log_rate, mu, sigma, log_w, log_pk = super().clocks(h, slots)
        L = h.shape[1]
        bonus = torch.einsum('bld,blkd->blk', self._query[:, :L], self._keyed[:, :L])
        if not self.qk_norm:
            bonus = bonus / math.sqrt(self.dk)
        logits = log_pk + bonus.unsqueeze(2)
        return log_rate, mu, sigma, log_w, F.log_softmax(logits, -1)


def mark_posterior(model, t, m, mask):
    """Total LL, and log p(mark | true gap, history) for every scored event: [B, L-1, K]."""
    tl, ml, n, (h, slots, params, tau, valid) = model.loglik(t, m, mask)
    log_h, _ = model.clock_terms(tau, *params[:4])
    log_lam_k = torch.logsumexp(log_h.unsqueeze(-1) + params[4], -2)
    return tl, ml, n, log_lam_k - torch.logsumexp(log_lam_k, -1, keepdim=True)


def evaluate(model, data, Kk, batch=128):
    T, M, R, SB = data
    model.eval(); tot = dict(time=0.0, mark=0.0, n=0, rec_n=0, rec_correct=0, rec_ll=0.0)
    with torch.no_grad():
        for i in range(0, len(T), batch):
            t, m, r = T[i:i + batch], M[i:i + batch], R[i:i + batch]
            tl, ml, n, logp = mark_posterior(model, t, m, torch.ones_like(m, dtype=torch.bool))
            tot['time'] += tl.item(); tot['mark'] += ml.item(); tot['n'] += int(n)
            rr = r[:, 1:]; tgt = m[:, 1:]
            pred_val = logp[..., Kk:].argmax(-1) + Kk                        # answer restricted to the value vocabulary
            tot['rec_n'] += int(rr.sum()); tot['rec_correct'] += int(((pred_val == tgt) & rr).sum())
            tot['rec_ll'] += logp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)[rr].sum().item()
    n = tot['n']
    return dict(ll=(tot['time'] + tot['mark']) / n, time_ll=tot['time'] / n, mark_ll=tot['mark'] / n, events=n,
                recall_acc=tot['rec_correct'] / tot['rec_n'], recall_mark_ll=tot['rec_ll'] / tot['rec_n'],
                set_baseline_acc=float(SB[R].mean()),
                recall_events=tot['rec_n'])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', required=True)
    ap.add_argument('--keyed', type=int, default=1)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--pairs', type=int, default=8)
    ap.add_argument('--pairs-extra', type=int, default=16, help='length-extrapolation evaluation (pairs)')
    ap.add_argument('--kk', type=int, default=32)
    ap.add_argument('--kv', type=int, default=32)
    ap.add_argument('--train', type=int, default=20000)
    ap.add_argument('--dev', type=int, default=1000)
    ap.add_argument('--d', type=int, default=32)
    ap.add_argument('--modes', type=int, default=16)
    ap.add_argument('--layers', type=int, default=2)
    ap.add_argument('--dk', type=int, default=16)
    ap.add_argument('--prev-msg', type=int, default=0)
    ap.add_argument('--qk-norm', type=int, default=0)
    ap.add_argument('--dv', type=int, default=4)
    ap.add_argument('--lr', type=float, default=3e-3)
    ap.add_argument('--batch', type=int, default=64)
    ap.add_argument('--epochs', type=int, default=60)
    ap.add_argument('--patience', type=int, default=8)
    ap.add_argument('--max-wall-s', type=float, default=2400)
    ap.add_argument('--threads', type=int, default=1)
    a = ap.parse_args()
    torch.set_num_threads(a.threads); torch.set_default_dtype(torch.float64)
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    K = a.kk + a.kv
    train = make_split(a.train, a.pairs, a.kk, a.kv, np.random.default_rng(1000 + a.seed))
    dev = make_split(a.dev, a.pairs, a.kk, a.kv, np.random.default_rng(2000 + a.seed))
    test = make_split(a.dev, a.pairs, a.kk, a.kv, np.random.default_rng(3000 + a.seed))
    extra = make_split(a.dev, a.pairs_extra, a.kk, a.kv, np.random.default_rng(4000 + a.seed))
    gaps = torch.diff(train[0], dim=1).flatten(); pos = gaps[gaps > 0].numpy()
    scale = float(np.median(pos)); qs = np.log(np.quantile(pos, np.linspace(0.1, 0.9, 4))).tolist()
    margs = (K, a.d, a.modes, a.layers, 2, 4, a.dv, 0.0, scale, qs)
    model = KeyedRaceTPP(*margs, dk=a.dk, local=a.keyed == 2, prev_msg=bool(a.prev_msg), qk_norm=bool(a.qk_norm)) if a.keyed else RaceTPP(*margs)
    params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.0)
    out_dir = ROOT / 'experiments/results/tpp/recall'; out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = out_dir / f'{a.tag}.pt'
    best, best_epoch, history, start = -math.inf, -1, [], time.time()
    gen = torch.Generator().manual_seed(a.seed)
    for epoch in range(a.epochs):
        model.train(); e0 = time.time(); tr = tn = 0.0
        perm = torch.randperm(a.train, generator=gen)
        for i in range(0, a.train, a.batch):
            idx = perm[i:i + a.batch]; t, m = train[0][idx], train[1][idx]
            tl, ml, n, _ = model.loglik(t, m, torch.ones_like(m, dtype=torch.bool))
            loss = -(tl + ml) / n
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            tr += (tl + ml).item(); tn += int(n)
        dv = evaluate(model, dev, a.kk)
        history.append(dict(epoch=epoch, train_ll=tr / tn, dev_ll=dv['ll'], dev_recall_acc=dv['recall_acc'],
                            epoch_s=round(time.time() - e0, 1)))
        print(json.dumps(history[-1]), flush=True)
        if dv['ll'] > best:
            best, best_epoch = dv['ll'], epoch; torch.save(model.state_dict(), ckpt)
        if epoch - best_epoch >= a.patience or time.time() - start > a.max_wall_s:
            break
    model.load_state_dict(torch.load(ckpt))
    result = dict(status='completed', battle='R1', decision='whether the B1 event model needs a query-keyed addressed '
                  'read to bind content across irregular gaps (gates transfer of the B1 model to language recall)',
                  tag=a.tag, args=vars(a), K=K, scale=scale, parameters=params, chance_recall_acc=1 / a.kv,
                  best_epoch=best_epoch, epochs_run=len(history), wall_s=time.time() - start,
                  dev=evaluate(model, dev, a.kk), test=evaluate(model, test, a.kk),
                  extrapolation=dict(pairs=a.pairs_extra, **evaluate(model, extra, a.kk)), history=history,
                  checkpoint=str(ckpt.relative_to(ROOT)))
    srcs = [Path(__file__).resolve(), Path(__file__).resolve().parent / 'race_tpp_v5.py']
    result['source_sha256'] = {str(s.relative_to(ROOT)): hashlib.sha256(s.read_bytes()).hexdigest() for s in srcs}
    (out_dir / f'{a.tag}.json').write_text(json.dumps(result, indent=1) + '\n')
    print('RESULT', json.dumps({k: result[k] for k in ('dev', 'test', 'extrapolation', 'parameters')}), flush=True)


if __name__ == '__main__':
    main()
