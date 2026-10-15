#!/usr/bin/env python3
"""R1 depth plan C v2: event-sparse asynchronous concept hierarchy (experiments/theory/DEPTH_PLAN_C_ASYNC_HIERARCHY.md).

Token level: `--base` plan-B blocks (pre-norm residual temporal memory + per-layer keyed race read + MLP).
Levels l = 1..`--levels` form a hierarchy of marked event streams (definition H1-H4):
- emission race (H1, H2): at each event of the level below, the level emits when its boundary logit beats its learned
  silence threshold (started alive at the first batch's 0.7-quantile); the cadence is learned against a small compute
  price with only a floor against dead levels (hard, straight-through);
- learned memory fetches: a level's race read uses sparsemax with a learned temperature per head, so each read decides how
  many of its events to fetch (exact zeros elsewhere); fetch counts are logged; emission is possible only at the level below's events (nested segments) and uses only the past.
- segment integration: an emitted event integrates its whole segment, i.e. every level-below event since this level's
  previous emission (mean of their features) and the histogram of their race outcomes (the lower level's concept
  choices); higher events therefore summarise many lower events and races.
- concept race (decision-tree routes): from that integrated evidence, a hard race over K concept slots routes the event;
  its content is the winning concept's embedding plus gated projections of the segment summary and race histogram. Successive choices across
  levels form a root-to-leaf style path of discrete branch decisions; upper levels see a vocabulary of concepts.
- event-sparse computation (H3): each level gathers only its own events into a compressed sequence carrying their
  token times; its temporal memory decays/rotates by the exact elapsed time between its events and is written only at
  them; its keyed race read attends over its own events. Compute falls with the emission rate at every level.
- top-down: every token position receives, through a zero-initialised gate, the output of each level's most recent
  event at or before it (causal hold), so at initialisation the model equals its base exactly.
- local learning (F2): each level predicts, from its state, the time to the next event of the level below (per-token
  silence-aware hazard) and that event's concept (cross-entropy over the lower level's K concepts, from level 2);
  an emission-rate target keeps every level alive and sparse; deep supervision reads the token stream after each level.
Same data, lanes, selection and 65,528-target scoring as the keyed member and plan B.
"""
import argparse, hashlib, json, math, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT/'experiments')); sys.path.insert(0, str(ROOT/'experiments/tpp'))
from r1_token_keyed_lm import load, evaluate as evaluate_keyed, V, KeyedTokenLM  # noqa: E402
from r1_deep_keyed_lm import Block  # noqa: E402


def event_memory(tm, u, times, valid, slow, rows=128):
    """Exact continuous-time memory over a compressed event sequence: z_i = sum_{j<=i} c^(t_i - t_j) w_j (valid events),
    computed in blocks of query rows."""
    g = torch.sigmoid(tm.gate(u)); w = tm.write(u); n = tm.n; vm = valid.to(u.dtype).unsqueeze(-1)
    wr, wi = w[..., :n]*g*vm, w[..., n:]*g*vm; rate, freq = tm.log_rate.exp()/slow, tm.freq/slow; out = []
    for r0 in range(0, u.shape[1], rows):
        ti = times[:, r0:r0 + rows]; lag = ti.unsqueeze(-1) - times.unsqueeze(-2)
        causal = ((lag >= 0) & valid[:, r0:r0 + rows].unsqueeze(-1) & valid.unsqueeze(-2)).to(u.dtype); lagc = lag.clamp_min(0).unsqueeze(-1)
        mag = torch.exp(-rate*lagc)*causal.unsqueeze(-1); ang = freq*lagc; kr, ki = mag*torch.cos(ang), mag*torch.sin(ang)
        zr = torch.einsum('bijn,bjn->bin', kr, wr) - torch.einsum('bijn,bjn->bin', ki, wi); zi = torch.einsum('bijn,bjn->bin', kr, wi) + torch.einsum('bijn,bjn->bin', ki, wr)
        out.append(torch.cat([zr, zi], -1))
    return torch.cat(out, 1)


def sparsemax(z, mask):
    """Sparsemax over the last dim with masked entries excluded: exact zeros outside a learned-size support."""
    z = z.masked_fill(~mask, -1e9); zs, _ = torch.sort(z, -1, descending=True); k = torch.arange(1, z.shape[-1] + 1, dtype=z.dtype)
    cs = zs.cumsum(-1); support = (1 + k*zs > cs) & (zs > -1e8); kz = support.sum(-1, keepdim=True).clamp_min(1)
    tau = (torch.gather(cs, -1, kz - 1) - 1)/kz.to(z.dtype); return torch.clamp(z - tau, min=0)*mask


def event_read(read, x, times, valid, slow, prev_content, temp, rows=256):
    """Race read over a level's own events; sparsemax with a learned temperature per head decides how many events are fetched."""
    B, N, _ = x.shape; kv = torch.cat([x, prev_content], -1)
    q = read.q(x).view(B, N, read.h, read.dk).transpose(1, 2); k = read.k(kv).view(B, N, read.h, read.dk).transpose(1, 2); v = read.v(kv).view(B, N, read.h, read.dk).transpose(1, 2)
    outs = []; fetched = []
    for r0 in range(0, N, rows):
        lag = (times[:, r0:r0 + rows].unsqueeze(-1) - times.unsqueeze(-2)).unsqueeze(1)
        ok = (lag >= 0) & valid.view(B, 1, 1, N) & valid[:, r0:r0 + rows].view(B, 1, -1, 1)
        logits = ((q[:, :, r0:r0 + rows] @ k.transpose(-1, -2))/math.sqrt(read.dk) - (read.log_rate.exp().view(1, -1, 1, 1)/slow)*lag.clamp_min(0))/F.softplus(temp).view(1, -1, 1, 1)
        a = sparsemax(logits, ok); outs.append(a @ v); fetched.append(((a > 0).sum(-1).to(x.dtype), ok.any(-1)))
    nf = torch.cat([f for f, _ in fetched], 2); okq = torch.cat([o for _, o in fetched], 2)
    return read.o(torch.cat(outs, 2).transpose(1, 2).reshape(B, N, -1)), (nf*okq).sum()/okq.sum().clamp_min(1)


class Level(nn.Module):
    def __init__(self, d, modes, heads, dk, dropout, depth, index, concepts):
        super().__init__(); self.block = Block(d, modes, heads, dk, dropout, depth); self.slow = 4.0**index; self.K = concepts
        self.norm_in = nn.LayerNorm(d); self.boundary = nn.Linear(d, 1); nn.init.normal_(self.boundary.weight, std=d**-.5); nn.init.zeros_(self.boundary.bias)
        self.threshold = nn.Parameter(torch.zeros(())); self.register_buffer('initialised', torch.zeros((), dtype=torch.bool))   # learned emission race vs silence
        self.read_temp = nn.Parameter(torch.zeros(heads))                                          # learned fetch sharpness per head
        self.concept_logits = nn.Linear(d, concepts); self.concepts = nn.Embedding(concepts, d); nn.init.normal_(self.concepts.weight, std=1.0)
        self.summary = nn.Linear(d, d); self.summary_gate = nn.Parameter(torch.tensor(0.0)); self.content_norm = nn.LayerNorm(d)
        self.gate = nn.Parameter(torch.zeros(())); self.top_down = nn.Linear(d, d)                  # ReZero top-down
        self.hazard = nn.Linear(d, 1); self.next_concept = nn.Linear(d, concepts)                   # local predictive heads (lower K equals K)
        self.hist_proj = nn.Linear(concepts, d)

    def forward(self, below, times, valid, below_concepts=None, below_K=None):
        """below: [B, N, d] events of the level below (token level: every position); times: [B, N]; valid: [B, N] bool;
        below_concepts: [B, N] the lower level's concept race outcomes (None at the token level)."""
        u = self.norm_in(below); logit = self.boundary(u).squeeze(-1); p = torch.sigmoid(logit - self.threshold)
        hard = (logit >= self.threshold) & valid; st = hard.to(u.dtype) + (p - p.detach())*valid.to(u.dtype)   # straight-through emission
        if self.training and not bool(self.initialised) and valid.any():                         # start alive: threshold at the 0.7-quantile of the first batch
            with torch.no_grad(): self.threshold.copy_(torch.quantile(logit[valid].detach().float(), 0.7).to(self.threshold.dtype)); self.initialised.fill_(True)
            p = torch.sigmoid(logit - self.threshold); hard = (logit >= self.threshold) & valid; st = hard.to(u.dtype) + (p - p.detach())*valid.to(u.dtype)
        # segments: lower events since this level's previous emission, closed by the emitting event (causal)
        B, N, d = u.shape; hv = hard.to(u.dtype); seg = (torch.cumsum(hv, 1) - hv).long()        # segment index of each lower event
        member = ((seg.unsqueeze(-1) == seg.unsqueeze(-2)) & (torch.arange(N).view(1, 1, N) <= torch.arange(N).view(1, N, 1)) & valid.unsqueeze(1)).to(u.dtype)
        count = member.sum(-1, keepdim=True).clamp_min(1); seg_mean = (member @ u)/count                 # integrate segment up to and including each position
        evidence = seg_mean
        if below_concepts is not None:
            hist = (member @ F.one_hot(below_concepts, below_K).to(u.dtype))/count                         # histogram of lower race outcomes in the segment
            evidence = evidence + self.hist_proj(hist)
        cl = self.concept_logits(evidence); soft = torch.softmax(cl, -1); idx = cl.argmax(-1)
        onehot = F.one_hot(idx, self.K).to(u.dtype); route = onehot + soft - soft.detach()           # straight-through concept race
        content = self.content_norm(route @ self.concepts.weight + torch.sigmoid(self.summary_gate)*self.summary(evidence))*st.unsqueeze(-1)
        # gather this level's events into a compressed sequence (pad to the longest lane)
        B, N, d = content.shape; counts = hard.sum(1); M = max(int(counts.max()), 1)
        order = torch.argsort((~hard).to(torch.int8), dim=1, stable=True)[:, :M]                  # emitted positions first, in time order
        ev_valid = torch.arange(M).unsqueeze(0) < counts.unsqueeze(1)
        ev = torch.gather(content, 1, order.unsqueeze(-1).expand(-1, -1, d)); ev_t = torch.gather(times, 1, order); ev_c = torch.gather(idx, 1, order)
        prev = torch.cat([torch.zeros_like(ev[:, :1]), ev[:, :-1]], 1)
        b = self.block; x = ev
        x = x + b.drop(b.tm_out(event_memory(b.tm, b.n1(x), ev_t, ev_valid, self.slow)))
        rd, fetches = event_read(b.read, b.n2(x), ev_t, ev_valid, self.slow, prev, self.read_temp); x = x + b.drop(rd)
        x = x + b.drop(b.mlp(b.n3(x)))
        return dict(events=x, times=ev_t, valid=ev_valid, concepts=ev_c, order=order, hard=hard, p=p, st=st, concept_soft=soft, fetches=fetches)


def hold(events, ev_times, ev_valid, token_times):
    """Causal hold: each token position takes the output of the level's latest event at or before it (zero before any)."""
    B, M, d = events.shape; le = (ev_times.unsqueeze(1) <= token_times.unsqueeze(-1)) & ev_valid.unsqueeze(1)   # [B, L, M]
    last = torch.where(le, torch.arange(M).view(1, 1, M), torch.full_like(le, -1, dtype=torch.long)).max(-1).values
    out = torch.gather(events, 1, last.clamp_min(0).unsqueeze(-1).expand(-1, -1, d)); return out*(last >= 0).unsqueeze(-1).to(events.dtype)


class AsyncHierarchyLM(nn.Module):
    def __init__(self, d, modes, base, levels, heads, dk, dropout, concepts, rate=0.25):
        super().__init__(); depth = base + levels; self.keyed, self.heads, self.dk = 1, 4, 16
        self.embed = nn.Embedding(V, d); nn.init.normal_(self.embed.weight, std=0.02)
        self.base = nn.ModuleList(Block(d, modes, heads, dk, dropout, depth) for _ in range(base))
        self.levels = nn.ModuleList(Level(d, modes, heads, dk, dropout, depth, i + 1, concepts) for i in range(levels))
        self.price = rate; self.floor = 0.01
        self.out_norm = nn.LayerNorm(d); self.bias = nn.Parameter(torch.zeros(V))
        self.key_w = nn.Linear(2*d, 64); self.query_w = nn.Linear(d, 64); self.log_rate = nn.Parameter(torch.linspace(math.log(1e-3), math.log(0.3), 4))
        self.checkpoint_blocks = False; self._aux = None

    keyed_scores = KeyedTokenLM.keyed_scores; logits = KeyedTokenLM.logits

    def states(self, x):
        e0 = self.embed(x); h = e0
        for b in self.base:
            h = torch.utils.checkpoint.checkpoint(b, h, e0, use_reentrant=False) if (self.checkpoint_blocks and self.training) else b(h, e0)
        B, L, _ = h.shape; tok_t = torch.arange(L, dtype=h.dtype).expand(B, L)
        tokens = h; below, times, valid, bconc = h, tok_t, torch.ones(B, L, dtype=torch.bool), None; streams = [tokens]; levels = []
        for lv in self.levels:
            r = lv(below, times, valid, bconc, lv.K)
            tokens = tokens + lv.gate*lv.top_down(hold(r['events'], r['times'], r['valid'], tok_t))
            levels.append(dict(r, below_times=times, below_valid=valid)); streams.append(tokens)
            below, times, valid, bconc = r['events'], r['times'], r['valid'], r['concepts']
        self._aux = dict(levels=levels, streams=streams, token_times=tok_t)
        return self.out_norm(tokens), e0

    def local_losses(self, x, rate):
        aux = self._aux; haz = torch.zeros(()); conc = torch.zeros(()); rate_loss = torch.zeros(()); ds = []; tok_t = aux['token_times']
        for i, lv in enumerate(aux['levels']):
            st = lv['st']; vb = lv['below_valid'].to(st.dtype); frac = (st*vb).sum()/vb.sum().clamp_min(1)
            rate_loss = rate_loss + self.price*frac + 10.0*F.relu(self.floor - frac)**2                   # learned cadence: compute price, floor against death
            em = lv['hard'] & lv['below_valid']
            if em.any():                                                                          # concept-usage balance (KL to uniform)
                marg = lv['concept_soft'][em].mean(0); rate_loss = rate_loss + (marg*(marg.clamp_min(1e-9).log() + math.log(marg.numel()))).sum()
            if i == 0: continue
            prev = aux['levels'][i - 1]; module = self.levels[i]
            # level i predicts, at each token time, whether level i-1 emits at the next token (silence-aware hazard)
            state_tok = hold(lv['events'], lv['times'], lv['valid'], tok_t)
            below_emit = torch.zeros_like(tok_t)
            below_emit = below_emit.scatter(1, prev['order'], prev['valid'].to(tok_t.dtype))
            haz = haz + F.binary_cross_entropy_with_logits(module.hazard(state_tok).squeeze(-1)[:, :-1], below_emit[:, 1:])
            # and the concept of level i-1's next event, from level i's state at the preceding event of level i-1
            nxt = module.next_concept(hold(lv['events'], lv['times'], lv['valid'], prev['times']))          # [B, M_prev, K]
            tgt = torch.cat([prev['concepts'][:, 1:], prev['concepts'][:, :1]], 1); ok = torch.cat([prev['valid'][:, 1:], torch.zeros_like(prev['valid'][:, :1])], 1)
            if ok.any(): conc = conc + F.cross_entropy(nxt[ok], tgt[ok])
        for s in aux['streams'][:-1]:
            lg = self.out_norm(s) @ self.embed.weight.T + self.bias; ds.append(F.cross_entropy(lg.reshape(-1, V), x[:, 1:].reshape(-1)))
        return haz, conc, rate_loss, (sum(ds)/len(ds) if ds else torch.zeros(()))

    def stats(self):
        out = []
        for lv in self._aux['levels']:
            n = float(lv['below_valid'].sum()); out.append(dict(rate=float(lv['hard'].sum())/max(n, 1), events=float(lv['valid'].sum())/lv['valid'].shape[0], fetches=round(float(lv['fetches']), 2),
                                                                  concepts_used=int(torch.unique(lv['concepts'][lv['valid']]).numel()) if lv['valid'].any() else 0))
        return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--tag', required=True); ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--train-tokens', type=int, default=1048576); ap.add_argument('--lane', type=int, default=256); ap.add_argument('--d', type=int, default=64)
    ap.add_argument('--modes', type=int, default=32); ap.add_argument('--base', type=int, default=2); ap.add_argument('--levels', type=int, default=2)
    ap.add_argument('--heads', type=int, default=4); ap.add_argument('--dk', type=int, default=16); ap.add_argument('--dropout', type=float, default=0.1); ap.add_argument('--concepts', type=int, default=32)
    ap.add_argument('--rate', type=float, default=0.02, help='compute price per unit emission rate (learned cadence)'); ap.add_argument('--w-hazard', type=float, default=0.1); ap.add_argument('--w-concept', type=float, default=0.1)
    ap.add_argument('--w-rate', type=float, default=1.0); ap.add_argument('--w-ds', type=float, default=0.3)
    ap.add_argument('--lr', type=float, default=3e-3); ap.add_argument('--wd', type=float, default=0.01); ap.add_argument('--batch', type=int, default=4); ap.add_argument('--epochs', type=int, default=20)
    ap.add_argument('--patience', type=int, default=2); ap.add_argument('--max-wall-s', type=float, default=10800); ap.add_argument('--max-steps', type=int, default=0)
    ap.add_argument('--score', type=int, default=1); ap.add_argument('--checkpoint-blocks', type=int, default=0)
    a = ap.parse_args(); torch.set_num_threads(1); torch.manual_seed(a.seed); np.random.seed(a.seed)
    train_all = load(ROOT/'data/fineweb_gpt2/fineweb_train_000001.bin'); val = load(ROOT/'data/fineweb_gpt2/fineweb_val_000000.bin')
    n_lanes = a.train_tokens//a.lane; train = np.asarray(train_all[:n_lanes*a.lane]).reshape(n_lanes, a.lane)
    model = AsyncHierarchyLM(a.d, a.modes, a.base, a.levels, a.heads, a.dk, a.dropout, a.concepts, a.rate); model.checkpoint_blocks = bool(a.checkpoint_blocks); params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd); total = math.ceil(n_lanes/a.batch)*a.epochs
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=total, pct_start=0.05)
    out_dir = ROOT/'experiments/results/token_language'; ckpt = out_dir/f'{a.tag}.pt'; out = out_dir/f'{a.tag}.json'; assert not out.exists()
    rng = np.random.default_rng(a.seed); best, best_epoch, history, start, steps = math.inf, -1, [], time.time(), 0
    for epoch in range(a.epochs):
        model.train(); e0 = time.time(); tl = tn = 0.0; st_hist = []; order = rng.permutation(n_lanes)
        for bi, i in enumerate(range(0, n_lanes, a.batch)):
            x = torch.from_numpy(train[np.sort(order[i:i + a.batch])].astype(np.int64))
            h, emb = model.states(x[:, :-1]); lg = model.logits(h, emb, x[:, :-1], list(range(a.lane - 1)))
            main_loss = F.cross_entropy(lg.reshape(-1, V), x[:, 1:].reshape(-1)); haz, conc, rl, ds = model.local_losses(x, a.rate)
            loss = main_loss + a.w_hazard*haz + a.w_concept*conc + a.w_rate*rl + a.w_ds*ds
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step(); steps += 1
            tl += main_loss.item()*x[:, 1:].numel(); tn += x[:, 1:].numel(); st_hist.append(model.stats())
            if bi % 100 == 0: print(json.dumps(dict(epoch=epoch, step=bi, train_nll=tl/tn, levels=st_hist[-1], gates=[round(float(l.gate), 4) for l in model.levels], elapsed_s=round(time.time() - start))), flush=True)
            if time.time() - start > a.max_wall_s or (a.max_steps and steps >= a.max_steps): break
        sel = evaluate_keyed(model, val, 10485760, 8, 2048); history.append(dict(epoch=epoch, train_nll=tl/tn, select_nll=sel['nll'], levels=st_hist[-1], epoch_s=round(time.time() - e0))); print(json.dumps(history[-1]), flush=True)
        if sel['nll'] < best: best, best_epoch = sel['nll'], epoch; torch.save(model.state_dict(), ckpt)
        if epoch - best_epoch >= a.patience or time.time() - start > a.max_wall_s or (a.max_steps and steps >= a.max_steps): break
    model.load_state_dict(torch.load(ckpt))
    result = dict(status='completed', battle='R1', stage='depth plan C v2: event-sparse asynchronous concept hierarchy', tag=a.tag, args=vars(a), parameters=params, best_epoch=best_epoch,
                  epochs_run=len(history), steps=steps, wall_s=time.time() - start, history=history, select=evaluate_keyed(model, val, 10485760, 8, 2048), checkpoint=str(ckpt.relative_to(ROOT)),
                  reference=dict(keyed_L2_1M_score_65528=6.019141, keyed_L2_4M_score_65528=5.521061))
    if a.score: result['score_65528'] = evaluate_keyed(model, val, 20971520, 8, 8192)
    result['source_sha256'] = {n: hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in ['experiments/r1_async_hierarchy_lm.py', 'experiments/r1_deep_keyed_lm.py', 'experiments/r1_token_keyed_lm.py', 'experiments/tpp/race_tpp_v5.py']}
    out.write_text(json.dumps(result, indent=1) + '\n'); print('RESULT', json.dumps({k: result.get(k) for k in ('select', 'score_65528', 'parameters', 'best_epoch')}), flush=True)


if __name__ == '__main__':
    main()
