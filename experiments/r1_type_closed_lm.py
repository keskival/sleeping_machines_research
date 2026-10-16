#!/usr/bin/env python3
"""R1: type-closed additive event stack (theory/RECURSIVE_EVENT_DEPTH.md sections 7-8, P9e).

Level 0 is the plan-B token stack (r1_deep_keyed_lm.DeepKeyedLM blocks). Level k >= 1 receives exactly what level 0
receives, in the same form: a causal stream of identity-marked events (the completed wholes of level k-1), a lexicon
(marks -> input values) and the plan-B blocks' keyed race reads over its own earlier events. It emits the same form upward.
- Whole of a level-(k-1) unit (word, clause, sentence, line): the LINEAR ordered integration of its parts' states
  (decaying/rotating memory reset at the unit start, read at the unit's last part; R2) plus a zero-initialised nonlinear
  correction.
- Mark: nearest key of an exponential-moving-average codebook (race with zero noise; no gradient; dead keys re-seeded).
- Input to level k: lambda[mark] + P (whole - sg(key[mark])), lambda initialised to the keys and P to the identity, so at
  initialisation the input equals the whole exactly (R3: mark + residual payload).
- Level k runs plan-B blocks on its compressed event sequence (unit steps per event; per-layer keyed reads over its own
  events, predecessor message from its previous input).
- Readout (R4): y_t = h_t + sum_k A_k u^k(latest event available at t), A_k zero-initialised: equals the base at init.
- Recursive likelihood: after each event, level k predicts the next event's mark over its input lexicon (tied values)
  and its gap in tokens.
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
from r1_deep_keyed_lm import DeepKeyedLM, Block  # noqa: E402
import r1_part_whole_lm as pw  # noqa: E402
import r1_concept_hierarchy_lm as ch  # noqa: E402


class Codebook(nn.Module):
    """EMA codebook: marks = nearest keys (no gradient); keys move to the mean of their assigned wholes."""
    def __init__(self, d, K, decay=0.99):
        super().__init__(); self.decay = decay
        self.register_buffer('keys', torch.randn(K, d) * 0.1); self.register_buffer('count', torch.ones(K)); self.register_buffer('total', self.keys.clone())

    def assign(self, w):
        with torch.no_grad(): return torch.cdist(w.reshape(-1, w.shape[-1]), self.keys.to(w.dtype)).argmin(-1).reshape(w.shape[:-1])

    @torch.no_grad()
    def update(self, w, c):
        w = w.reshape(-1, w.shape[-1]); c = c.reshape(-1); n = torch.bincount(c, minlength=len(self.keys)).float()
        s = torch.zeros_like(self.total).index_add_(0, c, w.to(self.total.dtype))
        self.count.mul_(self.decay).add_(n, alpha=1 - self.decay); self.total.mul_(self.decay).add_(s, alpha=1 - self.decay)
        self.keys.copy_(self.total / self.count.clamp_min(1e-3).unsqueeze(-1))
        dead = self.count < 1e-2
        if dead.any() and len(w):                                                  # re-seed dead keys from recent wholes
            pick = w[torch.randint(len(w), (int(dead.sum()),))]; self.keys[dead] = pick; self.total[dead] = pick * 0.05; self.count[dead] = 0.05


class ConstGate(nn.Module):
    """Constant write gate: keeps the part-to-whole integration exactly linear (R2)."""
    def __init__(self, modes): super().__init__(); self.modes = modes
    def forward(self, u): return u.new_full(u.shape[:-1] + (self.modes,), 20.)


class LinearMemory(pw.Memory):
    def __init__(self, d, modes):
        super().__init__(d, modes); self.write = nn.Linear(d, 2*modes, bias=False); self.gate = ConstGate(modes)


class EventLevel(nn.Module):
    def __init__(self, d, modes, heads, dk, dropout, layers, K, gaps):
        super().__init__(); self.gaps = gaps
        self.part = LinearMemory(d, modes); self.part_out = nn.Linear(2*modes, d, bias=False)   # linear ordered integration of parts (R2)
        self.n_corr = nn.LayerNorm(d); self.corr = nn.Sequential(nn.Linear(d, 2*d), nn.GELU(), nn.Linear(2*d, d))
        nn.init.zeros_(self.corr[2].weight); nn.init.zeros_(self.corr[2].bias)
        self.codebook = Codebook(d, K); self.lexicon = nn.Parameter(self.codebook.keys.clone()); self.P = nn.Linear(d, d, bias=False); nn.init.eye_(self.P.weight)
        self.blocks = nn.ModuleList(Block(d, modes, heads, dk, dropout, layers) for _ in range(layers))
        self.n_out = nn.LayerNorm(d); self.head_gap = nn.Linear(d, gaps); self.head_bias = nn.Parameter(torch.zeros(K))

    def wholes(self, parts, unit_start):
        """Linear in-unit integration of part states (reset at unit starts), read at every part; correction added."""
        span = unit_start.long().cumsum(1); n = torch.ones_like(unit_start)
        I = pw.span_integrate(self.part, parts, span, n) if parts.shape[1] else parts
        return self.part_out(I) + self.corr(self.n_corr(parts))


def compress(ev):
    """Positions of events per lane, padded: [B, E], real mask."""
    B, L = ev.shape; E = max(int(ev.sum(1).max()), 1); t = torch.arange(L)
    pos = torch.where(ev, t, t + L).sort(1).values[:, :E]; real = pos < L; return torch.where(real, pos, torch.zeros_like(pos)), real


class TypeClosedLM(DeepKeyedLM):
    def __init__(self, d, modes, layers, heads, dk, dropout, level_layers=1, K=(1024, 1024, 512, 256), gaps=64):
        super().__init__(d, modes, layers, heads, dk, dropout)
        self.levels = nn.ModuleList(EventLevel(d, modes, heads, dk, dropout, level_layers, k, gaps) for k in K)
        self.A = nn.ModuleList(nn.Linear(d, d) for _ in K)
        for a in self.A: nn.init.zeros_(a.weight); nn.init.zeros_(a.bias)
        self.cache = None

    def states(self, x):
        e = self.embed(x); h = e
        for b in self.blocks:
            h = torch.utils.checkpoint.checkpoint(b, h, e, use_reentrant=False) if (self.checkpoint_blocks and self.training) else b(h, e)
        B, L, d = h.shape; starts = ch.boundaries5(x)                                   # unit starts: word, clause, sentence, line, document
        y = h; parts, part_pos, real_parts = h, torch.arange(L).expand(B, L), torch.ones(B, L, dtype=torch.bool); cache = []
        for k, lv in enumerate(self.levels):
            unit_start_tok = starts[k]                                                   # level-k units over tokens
            # parts of level k = events of the stream below (tokens for k = 0), with their token positions
            first_tok = torch.cat([torch.zeros_like(part_pos[:, :1]), part_pos[:, :-1] + 1], 1) if k > 0 else part_pos
            us = unit_start_tok.gather(1, first_tok.clamp_max(L - 1))                   # does a part begin a new level-k unit (at its first token)?
            w = lv.wholes(parts, us)                                                     # provisional whole at every part
            # a unit is complete once the NEXT real part begins a new unit; the last real part's unit is still open
            real_next = torch.cat([real_parts[:, 1:], torch.zeros_like(real_parts[:, :1])], 1)
            done = torch.cat([us[:, 1:], torch.zeros_like(us[:, :1])], 1) & real_next & real_parts
            pos, real = compress(done); whole = w.gather(1, pos[..., None].expand(-1, -1, d)) * real[..., None]
            tok_end = part_pos.gather(1, pos); avail = (tok_end + 1).clamp_max(L - 1)  # available at the token after the unit's last part
            c = lv.codebook.assign(whole); r = whole - lv.codebook.keys[c].detach()
            inp = lv.lexicon[c] + lv.P(r)                                               # mark value + residual payload (= whole at init)
            u = inp; prev = torch.cat([torch.zeros_like(inp[:, :1]), inp[:, :-1]], 1)
            for blk in lv.blocks:
                u = torch.utils.checkpoint.checkpoint(blk, u, prev, use_reentrant=False) if self.training else blk(u, prev)
            u = lv.n_out(u) * real[..., None]
            # hold: token t sees the latest level-k event with availability <= t (causal), added through A_k (zero at init)
            cnt = torch.zeros(B, L, dtype=torch.long).scatter_add(1, torch.where(real, avail, torch.full_like(avail, L - 1)), (real & (tok_end + 1 < L)).long()).cumsum(1)
            held = torch.where((cnt > 0)[..., None], u.gather(1, (cnt - 1).clamp_min(0)[..., None].expand(-1, -1, d)), torch.zeros(B, L, d))
            y = y + self.A[k](held)
            cache.append(dict(c=c, real=real, u=u, whole=whole, tok_end=tok_end, inp_lexicon=lv.lexicon, level=lv))
            parts, part_pos, real_parts = u, tok_end, real                               # the next level's parts = this level's events
        self.cache = cache
        return self.out_norm(y), e

    def recursive_loss(self):
        """After each event of level k, predict the next event's mark (tied lexicon values) and its gap in tokens."""
        total, used = 0., 0
        for z in self.cache:
            u, c, real, lv = z['u'], z['c'], z['real'], z['level']
            ok = real[:, 1:] & real[:, :-1]
            if not ok.any(): continue
            logits = lv.n_out(u[:, :-1]) @ lv.lexicon.T + lv.head_bias; gap = (z['tok_end'][:, 1:] - z['tok_end'][:, :-1]).clamp(1, lv.gaps) - 1
            nll = F.cross_entropy(logits[ok], c[:, 1:][ok]) + F.cross_entropy(lv.head_gap(u[:, :-1])[ok], gap[ok])
            total = total + nll; used += 1
        return total / max(used, 1), used

    @torch.no_grad()
    def update_codebooks(self):
        for z in self.cache:
            z['level'].codebook.update(z['whole'][z['real']].detach(), z['c'][z['real']])


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--tag', required=True); ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--train-tokens', type=int, default=4194304); ap.add_argument('--lane', type=int, default=1024); ap.add_argument('--d', type=int, default=64)
    ap.add_argument('--modes', type=int, default=32); ap.add_argument('--layers', type=int, default=2); ap.add_argument('--heads', type=int, default=4); ap.add_argument('--dk', type=int, default=16)
    ap.add_argument('--dropout', type=float, default=0.1); ap.add_argument('--lr', type=float, default=3e-3); ap.add_argument('--wd', type=float, default=0.01)
    ap.add_argument('--batch', type=int, default=1); ap.add_argument('--checkpoint-every', type=int, default=256); ap.add_argument('--checkpoint-blocks', type=int, default=0)
    ap.add_argument('--level-layers', type=int, default=1); ap.add_argument('--codebooks', type=int, nargs='+', default=[1024, 1024, 512, 256]); ap.add_argument('--gaps', type=int, default=64)
    ap.add_argument('--recursive-weight', type=float, default=0.1); ap.add_argument('--max-steps', type=int, default=0, help='smoke only: stop after this many updates (no scoring)')
    a = ap.parse_args(); torch.set_num_threads(1); torch.manual_seed(a.seed)
    train_all = load(ROOT/'data/fineweb_gpt2/fineweb_train_000001.bin'); val = load(ROOT/'data/fineweb_gpt2/fineweb_val_000000.bin')
    n_lanes = a.train_tokens//a.lane; train = np.asarray(train_all[:n_lanes*a.lane]).reshape(n_lanes, a.lane)
    order = np.random.default_rng(a.seed).permutation(n_lanes); total = math.ceil(n_lanes/a.batch); quarter = math.ceil(total/4)
    model = TypeClosedLM(a.d, a.modes, a.layers, a.heads, a.dk, a.dropout, a.level_layers, tuple(a.codebooks), a.gaps); model.checkpoint_blocks = bool(a.checkpoint_blocks)
    params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd); sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=total, pct_start=0.05)
    out_dir = ROOT/'experiments/results/token_language'; out = out_dir/f'{a.tag}.json'; state_path = out_dir/f'{a.tag}.resume.pt'; assert not out.exists()
    step, losses, rec_l, milestones, wall, resumes = 0, [], [], [], 0.0, 0
    if state_path.exists():
        s = torch.load(state_path, weights_only=False); model.load_state_dict(s['model']); opt.load_state_dict(s['opt']); sched.load_state_dict(s['sched'])
        torch.set_rng_state(s['torch_rng']); step, losses, rec_l, milestones, wall, resumes = s['step'], s['losses'], s['rec'], s['milestones'], s['wall'], s['resumes'] + 1
        assert s['args'] == {k: v for k, v in vars(a).items() if k != 'max_steps'}, 'resume with different settings'
        print(json.dumps(dict(resumed_at_step=step)), flush=True)
    began = time.time() - wall
    def save():
        tmp = state_path.with_suffix('.tmp'); torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(), torch_rng=torch.get_rng_state(), step=step, losses=losses, rec=rec_l,
                   milestones=milestones, wall=time.time() - began, resumes=resumes, args={k: v for k, v in vars(a).items() if k != 'max_steps'}), tmp); os.replace(tmp, state_path)
    def usage():
        return [dict(effective_marks=float(torch.exp(-(lambda p: (p * p.clamp_min(1e-12).log()).sum())(lv.codebook.count / lv.codebook.count.sum()))), keys=len(lv.codebook.count)) for lv in model.levels]
    while step < total:
        model.train(); x = torch.from_numpy(train[np.sort(order[step*a.batch:(step + 1)*a.batch])].astype(np.int64)); h, e = model.states(x[:, :-1])
        lg = model.logits(h, e, x[:, :-1], list(range(a.lane - 1))); loss = F.cross_entropy(lg.reshape(-1, V), x[:, 1:].reshape(-1))
        rl, _ = model.recursive_loss(); rl = rl if torch.is_tensor(rl) else torch.zeros(())
        opt.zero_grad(); (loss + a.recursive_weight*rl).backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step(); model.update_codebooks(); step += 1
        losses.append(loss.item()); rec_l.append(rl.item())
        if step % 100 == 0: print(json.dumps(dict(step=step, of=total, prequential_last100=float(np.mean(losses[-100:])), recursive_last100=float(np.mean(rec_l[-100:])), usage=usage(), elapsed_s=round(time.time() - began))), flush=True)
        if step % quarter == 0 or step == total:
            milestones.append(dict(step=step, tokens=step*a.batch*a.lane, select_nll=evaluate_keyed(model, val, 10485760, 8, 2048)['nll'], prequential_segment=float(np.mean(losses[-quarter:])), usage=usage(), elapsed_s=round(time.time() - began)))
            print(json.dumps(milestones[-1]), flush=True)
        if step % a.checkpoint_every == 0 or step == total: save()
        if a.max_steps and step >= a.max_steps:
            rec = dict(status='smoke_completed', battle='R1', tag=a.tag, args=vars(a), parameters=params, smoke_steps=step, s_per_step=(time.time() - began)/step,
                       max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, losses=losses, recursive=rec_l, usage=usage())
            out.write_text(json.dumps(rec, indent=1) + '\n'); state_path.unlink(missing_ok=True); print(json.dumps({k: rec[k] for k in ('smoke_steps', 's_per_step', 'max_rss_kb')}), flush=True); return
    seg = max(1, total//8)
    result = dict(status='completed', battle='R1', stage='type-closed additive event stack, one pass', tag=a.tag, args=vars(a), parameters=params, steps=step,
                  presentations=step*a.batch*(a.lane - 1), train_wall_s=time.time() - began, resumes=resumes, milestones=milestones, usage=usage(),
                  prequential_by_eighth=[float(np.mean(losses[i:i + seg])) for i in range(0, len(losses), seg)], prequential_last_quarter=float(np.mean(losses[-quarter:])),
                  recursive_by_eighth=[float(np.mean(rec_l[i:i + seg])) for i in range(0, len(rec_l), seg)],
                  select=evaluate_keyed(model, val, 10485760, 8, 2048), score_65528=evaluate_keyed(model, val, 20971520, 8, 8192), flagship_protocol=pw.flagship(model, val),
                  max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, reference=dict(base='curie_r1_deep_keyed_stream_L2_4M_l1024_s0_20261015_v1'))
    torch.save(model.state_dict(), out_dir/f'{a.tag}.pt'); result['checkpoint'] = f'experiments/results/token_language/{a.tag}.pt'
    result['source_sha256'] = {n: hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in ['experiments/r1_type_closed_lm.py', 'experiments/r1_deep_keyed_lm.py', 'experiments/r1_part_whole_lm.py', 'experiments/r1_concept_hierarchy_lm.py', 'experiments/tables/gpt2_token_roles.json']}
    out.write_text(json.dumps(result, indent=1) + '\n'); state_path.unlink()
    print('RESULT', json.dumps({k: result[k] for k in ('select', 'score_65528', 'flagship_protocol', 'prequential_last_quarter', 'parameters')}), flush=True)


if __name__ == '__main__':
    main()
