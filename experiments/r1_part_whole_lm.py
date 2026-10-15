#!/usr/bin/env python3
"""R1: observed part-whole hierarchy on the strongest native language member (theory/PART_WHOLE_COMPOSITION.md).

Level 0: plan-B deep keyed blocks (r1_deep_keyed_lm.Block) over tokens. Levels 1-3: word, clause, sentence/line, with
boundaries observed from GPT-2 token roles (experiments/tables/gpt2_token_roles.json). Each level k:
- ordered integration of the completed parts of its current span (temporal memory reset at the span start, lag in parts);
- provisional whole b^k_t = MLP(LN[integration ; part in progress u^{k-1}_t]);
- level memory over completed wholes (temporal memory with lag in level events) and a keyed race read over past wholes;
  u^k_t = b^k_t + memory + read.
Readout: x = u^0 + A LN[u^1..u^K] (A zero-initialised), x += MLP(LN x), then the member's tied logits and keyed bonus; at
initialisation the model equals its base exactly. Level-matched predictive targets: level k predicts the tokens of the
next N_k positions from u^k (averaged cross-entropy, subsampled positions). Training: one pass over fresh 256-token lanes
(as r1_deep_keyed_stream.py), full backpropagation over the lane, exact resume.
"""
import argparse, hashlib, json, math, os, resource, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT/'experiments')); sys.path.insert(0, str(ROOT/'experiments/tpp'))
from r1_token_keyed_lm import load, evaluate as evaluate_keyed, V  # noqa: E402
from r1_deep_keyed_lm import DeepKeyedLM  # noqa: E402

ROLES = torch.tensor([int(c) for c in json.loads((ROOT/'experiments/tables/gpt2_token_roles.json').read_text())['roles']], dtype=torch.long)


def boundaries(x):
    """Span starts for levels 1..3 from input tokens x [B, L] (causal: uses x_t and x_{t-1} only)."""
    r = ROLES[x]; prev = torch.cat([torch.full_like(r[:, :1], 7), r[:, :-1]], 1); first = torch.zeros_like(r, dtype=torch.bool); first[:, 0] = True
    s3 = first | (prev == 4) | (prev == 5) | (prev == 7)
    s2 = s3 | (prev == 3)
    s1 = s2 | (r != 1) | ~((prev == 0) | (prev == 1))
    return [s1, s2, s3]


def _kernel_rows(wr, wi, lag, mask, rate, freq):
    lagc = lag.clamp_min(0).unsqueeze(-1); m = mask.to(wr.dtype).unsqueeze(-1)
    mag = torch.exp(-rate*lagc)*m; ang = freq*lagc; kr, ki = mag*torch.cos(ang), mag*torch.sin(ang)
    zr = torch.einsum('btjn,bjn->btn', kr, wr) - torch.einsum('btjn,bjn->btn', ki, wi)
    zi = torch.einsum('btjn,bjn->btn', kr, wi) + torch.einsum('btjn,bjn->btn', ki, wr)
    return torch.cat([zr, zi], -1)


def masked_memory(tm, u, lag, mask, rows=64):
    """z_t = sum_j mask[t,j] c^lag[t,j] w_j for a decaying/rotating complex-diagonal memory (exact, parallel).
    Computed in blocks of query rows; under autograd each block is recomputed in backward (bounded memory)."""
    n = tm.n; g = torch.sigmoid(tm.gate(u)); w = tm.write(u); wr, wi = w[..., :n]*g, w[..., n:]*g; rate, freq = tm.log_rate.exp(), tm.freq; out = []
    for r0 in range(0, u.shape[1], rows):
        args = (wr, wi, lag[:, r0:r0 + rows], mask[:, r0:r0 + rows], rate, freq)
        out.append(torch.utils.checkpoint.checkpoint(_kernel_rows, *args, use_reentrant=False) if torch.is_grad_enabled() else _kernel_rows(*args))
    return torch.cat(out, 1)


def _gathered_rows(wr, wi, idx, valid, lag, rate, freq):
    bi = torch.arange(wr.shape[0])[:, None, None]; wrg, wig = wr[bi, idx], wi[bi, idx]
    lagc = lag.clamp_min(0).to(wr.dtype).unsqueeze(-1); m = valid.to(wr.dtype).unsqueeze(-1)
    mag = torch.exp(-rate*lagc)*m; ang = freq*lagc; kr, ki = mag*torch.cos(ang), mag*torch.sin(ang)
    return torch.cat([(kr*wrg - ki*wig).sum(2), (kr*wig + ki*wrg).sum(2)], -1)


def gathered_memory(tm, u, idx, valid, lag, rows=64):
    """Same memory as masked_memory, restricted to the K candidate columns idx[b,t,k] of each row (span band or level
    events): z_t = sum_k valid c^lag w_idx. Exact when every nonzero column of the full kernel is among the candidates."""
    n = tm.n; g = torch.sigmoid(tm.gate(u)); w = tm.write(u); wr, wi = w[..., :n]*g, w[..., n:]*g; rate, freq = tm.log_rate.exp(), tm.freq
    if idx.shape[2] <= 64 or not torch.is_grad_enabled(): return _gathered_rows(wr, wi, idx, valid, lag, rate, freq)
    return torch.cat([torch.utils.checkpoint.checkpoint(_gathered_rows, wr, wi, idx[:, r:r + rows], valid[:, r:r + rows], lag[:, r:r + rows], rate, freq, use_reentrant=False)
                      for r in range(0, u.shape[1], rows)], 1)


def span_band(span, part_ev, n):
    """Candidate columns for in-span integration: the previous D tokens, D = longest span in the batch (exact)."""
    B, L = span.shape; lengths = torch.stack([torch.bincount(s, minlength=L + 2).max() for s in span]); D = int(lengths.max())
    idx = (torch.arange(L)[:, None] - torch.arange(D)[None, :]).expand(B, -1, -1); ok = idx >= 0; idx = idx.clamp_min(0)
    g = lambda a: a.gather(1, idx.reshape(B, -1)).view(B, L, D)
    valid = ok & (g(span) == span[:, :, None]) & g(part_ev); lag = n[:, :, None] - g(n)
    return idx, valid, lag


def event_columns(ev):
    """Candidate columns for level memory/read: that lane's event positions (padded), event e available from its time."""
    B, L = ev.shape; counts = ev.sum(1); E = max(int(counts.max()), 1); t = torch.arange(L)
    key = torch.where(ev, t, t + L); pos = key.sort(1).values[:, :E]; real = pos < L; pos = torch.where(real, pos, torch.zeros_like(pos))
    m = ev.long().cumsum(1); idx = pos[:, None, :].expand(B, L, E)
    valid = real[:, None, :] & (pos[:, None, :] <= t[None, :, None]); lag = m[:, :, None] - (torch.arange(E) + 1)[None, None, :]
    return idx, valid, lag


def chunked_scan(wr, wi, rate, freq, C=256):
    """Unit-step complex-diagonal recurrence Z_k = c Z_{k-1} + w_k over a sequence [B, S, n]: exact chunked kernel with
    the state carried across chunks (as r1_deep_keyed_lm.ParallelTemporalMemory); memory O(C^2 n)."""
    B, S, n = wr.shape; C = max(1, min(C, S)); k = torch.arange(C, dtype=wr.dtype); lag = k[:, None] - k[None, :]
    causal = (lag >= 0).to(wr.dtype); lagc = lag.clamp_min(0).unsqueeze(-1)
    mag = torch.exp(-rate*lagc)*causal.unsqueeze(-1); ang = freq*lagc; kr, ki = mag*torch.cos(ang), mag*torch.sin(ang)
    steps = (k + 1).unsqueeze(-1); cr, ci = torch.exp(-rate*steps)*torch.cos(freq*steps), torch.exp(-rate*steps)*torch.sin(freq*steps)
    sr = wr.new_zeros(B, n); si = wr.new_zeros(B, n); outs = []
    for c0 in range(0, S, C):
        ar, ai = wr[:, c0:c0 + C], wi[:, c0:c0 + C]; m = ar.shape[1]; Kr, Ki = kr[:m, :m], ki[:m, :m]
        zr = torch.einsum('ijn,bjn->bin', Kr, ar) - torch.einsum('ijn,bjn->bin', Ki, ai); zi = torch.einsum('ijn,bjn->bin', Kr, ai) + torch.einsum('ijn,bjn->bin', Ki, ar)
        zr = zr + cr[:m]*sr.unsqueeze(1) - ci[:m]*si.unsqueeze(1); zi = zi + cr[:m]*si.unsqueeze(1) + ci[:m]*sr.unsqueeze(1)
        outs.append(torch.cat([zr, zi], -1)); sr, si = zr[:, -1], zi[:, -1]
    return torch.cat(outs, 1)


def band_rows(tm, u, span, part_ev, n, r0, r1, D):
    """In-span integration for query rows r0:r1 over the previous D tokens (D = longest span: exact)."""
    B, L = span.shape; t = torch.arange(r0, r1); idx = (t[:, None] - torch.arange(D)[None, :]).expand(B, -1, -1); ok = idx >= 0; idx = idx.clamp_min(0)
    g = lambda a: a.gather(1, idx.reshape(B, -1)).view(B, r1 - r0, D)
    valid = ok & (g(span) == span[:, r0:r1, None]) & g(part_ev); lag = n[:, r0:r1, None] - g(n)
    nm = tm.n; gg = torch.sigmoid(tm.gate(u)); w = tm.write(u); wr, wi = w[..., :nm]*gg, w[..., nm:]*gg
    return _gathered_rows(wr, wi, idx, valid, lag, tm.log_rate.exp(), tm.freq)


def span_integrate(tm, u, span, part_ev, rows=64):
    B, L = span.shape; D = int(torch.stack([torch.bincount(s, minlength=L + 2).max() for s in span]).max()); n = part_ev.long().cumsum(1); out = []
    for r0 in range(0, L, rows):
        args = (tm, u, span, part_ev, n, r0, min(r0 + rows, L), D)
        out.append(torch.utils.checkpoint.checkpoint(band_rows, *args, use_reentrant=False) if torch.is_grad_enabled() else band_rows(*args))
    return torch.cat(out, 1)


class Memory(nn.Module):
    def __init__(self, d, modes):
        super().__init__(); self.n = modes
        self.log_rate = nn.Parameter(torch.logspace(-2, 1, modes).log()); self.freq = nn.Parameter(torch.randn(modes)*0.5)
        self.write = nn.Linear(d, 2*modes); self.gate = nn.Linear(d, modes)


class Level(nn.Module):
    def __init__(self, d, modes, heads, dk, horizon):
        super().__init__(); self.h, self.dk, self.horizon = heads, dk, horizon
        self.part = nn.Linear(d, d); self.integ = Memory(d, modes); self.integ_out = nn.Linear(2*modes, d)
        self.n_in = nn.LayerNorm(2*d); self.whole = nn.Sequential(nn.Linear(2*d, 4*d), nn.GELU(), nn.Linear(4*d, d))
        self.mem = Memory(d, modes); self.mem_out = nn.Linear(2*modes, d); self.n_q = nn.LayerNorm(d)
        self.q = nn.Linear(d, heads*dk); self.k = nn.Linear(d, heads*dk); self.v = nn.Linear(d, heads*dk); self.o = nn.Linear(heads*dk, d)
        self.log_rate = nn.Parameter(torch.linspace(math.log(1e-2), math.log(0.5), heads))
        self.future = nn.Linear(d, d); self.reference = False

    def forward(self, below, start, below_start, token_level):
        B, L, d = below.shape; span = start.long().cumsum(1)
        if self.reference: t = torch.arange(L); causal = t[:, None] >= t[None, :]; same = span[:, :, None] == span[:, None, :]
        if token_level:     # parts are the span's tokens, each complete when observed
            part_ev = torch.ones_like(start); part_val = below
        else:               # a part completes when the next level-(k-1) unit starts inside the same span
            part_ev = below_start & ~start; part_ev[:, 0] = False; part_val = torch.cat([torch.zeros_like(below[:, :1]), below[:, :-1]], 1)
        if self.reference:   # full L x L kernels (contract reference)
            n = part_ev.long().cumsum(1).to(below.dtype)
            I = masked_memory(self.integ, self.part(part_val), n[:, :, None] - n[:, None, :], causal & same & part_ev[:, None, :])
        else:
            I = span_integrate(self.integ, self.part(part_val), span, part_ev)
        b = self.whole(self.n_in(torch.cat([self.integ_out(I), below], -1)))
        # Completed wholes: when a span starts at j > 0, the whole that ended at j-1 becomes a level event at j.
        ev = start.clone(); ev[:, 0] = False; wv = torch.cat([torch.zeros_like(b[:, :1]), b[:, :-1]], 1)
        if self.reference:
            m = ev.long().cumsum(1).to(b.dtype); lag = m[:, :, None] - m[:, None, :]; emask = causal & ev[:, None, :]
            R = self.mem_out(masked_memory(self.mem, wv, lag, emask)); kv = wv
        else:   # compressed event sequence: level memory is a unit-step scan over its own events; reads in query-row blocks
            m = ev.long().cumsum(1); E = max(int(m[:, -1].max()), 1); tt = torch.arange(L)
            pos = torch.where(ev, tt, tt + L).sort(1).values[:, :E]; real = pos < L; pos = torch.where(real, pos, torch.zeros_like(pos))
            kv = wv.gather(1, pos[..., None].expand(-1, -1, d)) * real[..., None].to(wv.dtype)
            nm = self.mem.n; gg = torch.sigmoid(self.mem.gate(kv)); w = self.mem.write(kv); rm = real[..., None].to(kv.dtype)
            Z = chunked_scan(w[..., :nm]*gg*rm, w[..., nm:]*gg*rm, self.mem.log_rate.exp(), self.mem.freq)
            R = torch.where((m > 0)[..., None], Z.gather(1, (m - 1).clamp_min(0)[..., None].expand(-1, -1, Z.shape[-1])), torch.zeros_like(Z[:, :1]).expand(-1, L, -1))
            R = self.mem_out(R)
            k = self.k(kv).view(B, E, self.h, self.dk).transpose(1, 2); v = self.v(kv).view(B, E, self.h, self.dk).transpose(1, 2)
            q = self.q(self.n_q(b)).view(B, L, self.h, self.dk).transpose(1, 2); reads = []
            for r0 in range(0, L, 256):
                r1 = min(r0 + 256, L); mt = m[:, r0:r1]; e = torch.arange(E)
                valid = (e[None, None, :] < mt[:, :, None]) & real[:, None, :]; lagb = (mt[:, :, None] - (e + 1)[None, None, :]).to(b.dtype)
                sc = q[:, :, r0:r1] @ k.transpose(-1, -2)/math.sqrt(self.dk) - self.log_rate.exp().view(1, -1, 1, 1)*lagb.unsqueeze(1)
                sc = sc.masked_fill(~valid.unsqueeze(1), float('-inf')); sc = torch.cat([sc, torch.zeros_like(sc[..., :1])], -1)
                reads.append((torch.softmax(sc, -1)[..., :-1] @ v).transpose(1, 2).reshape(B, r1 - r0, -1))
            read = self.o(torch.cat(reads, 1))
            return b + R + read
        E = kv.shape[1]
        q = self.q(self.n_q(b)).view(B, L, self.h, self.dk).transpose(1, 2); k = self.k(kv).view(B, E, self.h, self.dk).transpose(1, 2); v = self.v(kv).view(B, E, self.h, self.dk).transpose(1, 2)
        sc = q @ k.transpose(-1, -2)/math.sqrt(self.dk) - self.log_rate.exp().view(1, -1, 1, 1)*lag.unsqueeze(1)
        sc = sc.masked_fill(~emask.unsqueeze(1), float('-inf')); sc = torch.cat([sc, torch.zeros_like(sc[..., :1])], -1)   # null route
        a = torch.softmax(sc, -1)[..., :-1]; read = self.o((a @ v).transpose(1, 2).reshape(B, L, -1))
        return b + R + read


class PartWholeLM(DeepKeyedLM):
    def __init__(self, d, modes, layers, heads, dk, dropout, horizons=(4, 16, 64), checkpoint_levels=True):
        super().__init__(d, modes, layers, heads, dk, dropout)
        self.levels = nn.ModuleList(Level(d, modes, heads, dk, h) for h in horizons); K = len(horizons)
        self.n_levels = nn.LayerNorm(K*d); self.level_in = nn.Linear(K*d, d); nn.init.zeros_(self.level_in.weight); nn.init.zeros_(self.level_in.bias)
        self.n_mix = nn.LayerNorm(d); self.mix = nn.Sequential(nn.Linear(d, 4*d), nn.GELU(), nn.Linear(4*d, d)); nn.init.zeros_(self.mix[2].weight); nn.init.zeros_(self.mix[2].bias)
        self.checkpoint_levels = checkpoint_levels; self.last_levels = None

    def states(self, x):
        e = self.embed(x); h = e
        for b in self.blocks:
            h = torch.utils.checkpoint.checkpoint(b, h, e, use_reentrant=False) if (self.checkpoint_blocks and self.training) else b(h, e)
        starts = boundaries(x); below, below_start, ups = h, None, []
        for i, level in enumerate(self.levels):
            args = (below, starts[i], below_start if below_start is not None else starts[i], i == 0)
            u = torch.utils.checkpoint.checkpoint(level, *args, use_reentrant=False) if (self.checkpoint_levels and self.training) else level(*args)
            ups.append(u); below, below_start = u, starts[i]
        self.last_levels = ups
        y = h + self.level_in(self.n_levels(torch.cat(ups, -1))); y = y + self.mix(self.n_mix(y))
        return self.out_norm(y), e

    def future_loss(self, x, every=8):
        """Level k predicts the tokens at positions t+1..t+N_k of the lane (x includes the final target)."""
        L = x.shape[1] - 1; total = 0.
        for level, u in zip(self.levels, self.last_levels):
            pos = torch.arange(every - 1, L - 1, every); lg = F.log_softmax(level.future(u[:, pos]) @ self.embed.weight.T + self.bias, -1)   # [B, P, V]
            offs = torch.arange(1, level.horizon + 1); idx = pos[:, None] + offs[None, :]; valid = idx <= L
            tgt = x[:, idx.clamp_max(L)]                                                                     # [B, P, N]
            ll = lg.gather(-1, tgt) * valid.to(lg.dtype)
            total = total - ll.sum() / (valid.sum() * x.shape[0])
        return total / len(self.levels)


FLAGSHIP_START, FLAGSHIP_TARGETS, FLAGSHIP_WINDOW = 25165824, 8192, 1024


def flagship(model, val):
    from r1_token_keyed_lm import lane_nll
    model.eval(); total = n = 0
    with torch.no_grad():
        for begin in range(FLAGSHIP_START, FLAGSHIP_START + FLAGSHIP_TARGETS, FLAGSHIP_WINDOW):
            t_, n_ = lane_nll(model, np.asarray(val[begin:begin + FLAGSHIP_WINDOW + 1])[None], 256); total += t_; n += n_
    assert n == FLAGSHIP_TARGETS
    return dict(nll=total/n, targets=n, target_start=FLAGSHIP_START + 1, window=FLAGSHIP_WINDOW)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--tag', required=True); ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--train-tokens', type=int, default=4194304); ap.add_argument('--lane', type=int, default=256); ap.add_argument('--d', type=int, default=64)
    ap.add_argument('--modes', type=int, default=32); ap.add_argument('--layers', type=int, default=2); ap.add_argument('--heads', type=int, default=4); ap.add_argument('--dk', type=int, default=16)
    ap.add_argument('--dropout', type=float, default=0.1); ap.add_argument('--lr', type=float, default=3e-3); ap.add_argument('--wd', type=float, default=0.01)
    ap.add_argument('--batch', type=int, default=4); ap.add_argument('--checkpoint-every', type=int, default=256)
    ap.add_argument('--horizons', type=int, nargs='+', default=[4, 16, 64]); ap.add_argument('--future-weight', type=float, default=0.1)
    ap.add_argument('--max-steps', type=int, default=0, help='smoke only: stop after this many updates (no scoring)')
    a = ap.parse_args(); torch.set_num_threads(1); torch.manual_seed(a.seed)
    train_all = load(ROOT/'data/fineweb_gpt2/fineweb_train_000001.bin'); val = load(ROOT/'data/fineweb_gpt2/fineweb_val_000000.bin')
    n_lanes = a.train_tokens//a.lane; train = np.asarray(train_all[:n_lanes*a.lane]).reshape(n_lanes, a.lane)
    order = np.random.default_rng(a.seed).permutation(n_lanes); total = math.ceil(n_lanes/a.batch); quarter = math.ceil(total/4)
    model = PartWholeLM(a.d, a.modes, a.layers, a.heads, a.dk, a.dropout, tuple(a.horizons)); params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd); sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=total, pct_start=0.05)
    out_dir = ROOT/'experiments/results/token_language'; out = out_dir/f'{a.tag}.json'; state_path = out_dir/f'{a.tag}.resume.pt'; assert not out.exists()
    step, losses, aux, milestones, wall, resumes = 0, [], [], [], 0.0, 0
    if state_path.exists():
        s = torch.load(state_path, weights_only=False); model.load_state_dict(s['model']); opt.load_state_dict(s['opt']); sched.load_state_dict(s['sched'])
        torch.set_rng_state(s['torch_rng']); step, losses, aux, milestones, wall, resumes = s['step'], s['losses'], s['aux'], s['milestones'], s['wall'], s['resumes'] + 1
        assert s['args'] == {k: v for k, v in vars(a).items() if k != 'max_steps'}, 'resume with different settings'
        print(json.dumps(dict(resumed_at_step=step)), flush=True)
    began = time.time() - wall
    def save():
        tmp = state_path.with_suffix('.tmp'); torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(), torch_rng=torch.get_rng_state(), step=step,
                   losses=losses, aux=aux, milestones=milestones, wall=time.time() - began, resumes=resumes, args={k: v for k, v in vars(a).items() if k != 'max_steps'}), tmp); os.replace(tmp, state_path)
    while step < total:
        model.train(); x = torch.from_numpy(train[np.sort(order[step*a.batch:(step + 1)*a.batch])].astype(np.int64)); h, e = model.states(x[:, :-1])
        lg = model.logits(h, e, x[:, :-1], list(range(a.lane - 1))); loss = F.cross_entropy(lg.reshape(-1, V), x[:, 1:].reshape(-1))
        fut = model.future_loss(x) if a.future_weight > 0 else torch.zeros(())
        opt.zero_grad(); (loss + a.future_weight*fut).backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step(); step += 1
        losses.append(loss.item()); aux.append(fut.item())                 # fresh lanes: next-token loss before this update is held out
        if step % 100 == 0: print(json.dumps(dict(step=step, of=total, prequential_last100=float(np.mean(losses[-100:])), future_last100=float(np.mean(aux[-100:])), elapsed_s=round(time.time() - began))), flush=True)
        if step % quarter == 0 or step == total:
            milestones.append(dict(step=step, tokens=step*a.batch*a.lane, select_nll=evaluate_keyed(model, val, 10485760, 8, 2048)['nll'], prequential_segment=float(np.mean(losses[-quarter:])), elapsed_s=round(time.time() - began)))
            print(json.dumps(milestones[-1]), flush=True)
        if step % a.checkpoint_every == 0 or step == total: save()
        if a.max_steps and step >= a.max_steps:
            rec = dict(status='smoke_completed', battle='R1', tag=a.tag, args=vars(a), parameters=params, smoke_steps=step, s_per_step=(time.time() - began)/step,
                       max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, losses=losses, future=aux)
            out.write_text(json.dumps(rec, indent=1) + '\n'); state_path.unlink(missing_ok=True); print(json.dumps({k: rec[k] for k in ('smoke_steps', 's_per_step', 'max_rss_kb')}), flush=True); return
    seg = max(1, total//8)
    result = dict(status='completed', battle='R1', stage='observed part-whole hierarchy on the deep keyed member, one pass', tag=a.tag, args=vars(a), parameters=params, steps=step,
                  presentations=step*a.batch*(a.lane - 1), train_wall_s=time.time() - began, resumes=resumes, milestones=milestones,
                  prequential_by_eighth=[float(np.mean(losses[i:i + seg])) for i in range(0, len(losses), seg)], prequential_last_quarter=float(np.mean(losses[-quarter:])),
                  future_by_eighth=[float(np.mean(aux[i:i + seg])) for i in range(0, len(aux), seg)],
                  select=evaluate_keyed(model, val, 10485760, 8, 2048), score_65528=evaluate_keyed(model, val, 20971520, 8, 8192), flagship_protocol=flagship(model, val),
                  max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                  reference=dict(baseline='curie_r1_deep_keyed_stream_L2_4M_s0_20261015_v1', keyed_L2_4M_flagship=dict(s0=5.624, s1=5.601)))
    torch.save(model.state_dict(), out_dir/f'{a.tag}.pt'); result['checkpoint'] = f'experiments/results/token_language/{a.tag}.pt'
    result['source_sha256'] = {n: hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in ['experiments/r1_part_whole_lm.py', 'experiments/r1_deep_keyed_lm.py', 'experiments/r1_token_keyed_lm.py', 'experiments/tables/gpt2_token_roles.json']}
    out.write_text(json.dumps(result, indent=1) + '\n'); state_path.unlink()
    print('RESULT', json.dumps({k: result[k] for k in ('select', 'score_65528', 'flagship_protocol', 'prequential_last_quarter', 'parameters')}), flush=True)


if __name__ == '__main__':
    main()
