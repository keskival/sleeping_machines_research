#!/usr/bin/env python3
"""R1: concept hierarchy for long receptive fields (theory/PART_WHOLE_COMPOSITION.md, section 5).

Extends the observed part-whole hierarchy (r1_part_whole_lm.py, unchanged) in three ways aimed at higher-level concepts:
- five levels with observed boundaries: word (1.1 tokens), clause (8.9), sentence (14.4), line/paragraph (newline, 44)
  and document (end-of-text, 688). A level ticks only at its own events, so its memory reaches (span x horizon) tokens; the
  document level's provisional whole is a running ordered summary of the current document (its topic state);
- level-matched future targets over 4/16/32/64/256 tokens: the long horizons reward document variables directly, where
  next-token loss spreads their signal thinly over many positions;
- sparse concept memories on the sentence, line and document levels: a race over K learned concept slots (top-k of all
  scored keys, null route; values start at zero) adds the selected concepts to the level state, so documents about the
  same thing pool evidence into shared slots (capacity beyond activity: K slots stored, k read per event).
Trained on long lanes (default one 1024-token lane per update, the same tokens per update as 4 x 256) with full
backpropagation over the lane. At initialisation the model equals its plan-B base exactly.
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
import r1_part_whole_lm as pw  # noqa: E402

ROLES = pw.ROLES


def boundaries5(x):
    """Span starts for word, clause, sentence, line, document (causal; nested)."""
    r = ROLES[x]; prev = torch.cat([torch.full_like(r[:, :1], 7), r[:, :-1]], 1); first = torch.zeros_like(r, dtype=torch.bool); first[:, 0] = True
    s5 = first | (prev == 7)
    s4 = s5 | (prev == 5)
    s3 = s4 | (prev == 4)
    s2 = s3 | (prev == 3)
    s1 = s2 | (r != 1) | ~((prev == 0) | (prev == 1))
    return [s1, s2, s3, s4, s5]


class ConceptMemory(nn.Module):
    """Race over K learned concept slots: top-k of the scored keys plus a null route; selected values added (zero at init)."""
    def __init__(self, d, slots, topk, dk=32):
        super().__init__(); self.topk, self.dk = topk, dk
        self.n = nn.LayerNorm(d); self.q = nn.Linear(d, dk); self.keys = nn.Parameter(torch.randn(slots, dk) * dk ** -.5)
        self.values = nn.Parameter(torch.zeros(slots, d)); self.null = nn.Parameter(torch.zeros(()))

    def forward(self, u):
        sc = self.q(self.n(u)) @ self.keys.T / math.sqrt(self.dk); top, idx = sc.topk(self.topk, -1)
        a = torch.softmax(torch.cat([top, self.null.expand(*top.shape[:-1], 1)], -1), -1)[..., :-1]
        return (a.unsqueeze(-1) * self.values[idx]).sum(-2), idx


NEG = -1e4


def group_lse(S, group, G):
    """Per query row, log-sum-exp of S over entries sharing a group id (sibling sets): [B,R,E] -> [B,R,G]."""
    gmax = torch.full(S.shape[:2] + (G,), NEG, dtype=S.dtype).scatter_reduce(2, group, S, 'amax', include_self=True).detach()
    se = torch.zeros(S.shape[:2] + (G,), dtype=S.dtype).scatter_add(2, group, torch.exp(S - gmax.gather(2, group)))
    return torch.log(se.clamp_min(1e-30)) + gmax


class TreeRead(nn.Module):
    """Race attention over the part-whole tree. Nodes per level: the completed wholes so far and the in-progress whole
    (keyed by its provisional state), which parents the completed parts inside it; leaves are tokens. Weight of a node =
    weight of its parent x softmax among its siblings (top level: softmax over its nodes and a null route), so the
    weights over every level form a distribution and every earlier token is reachable through its wholes. Each level's
    read and the token read are added to the prediction through zero-initialised maps."""
    def __init__(self, d, levels, dk=16):
        super().__init__(); self.dk = dk; n = levels + 1                                    # index levels: 0 = tokens, 1.. = word..
        self.norm = nn.ModuleList(nn.LayerNorm(d) for _ in range(n)); self.q = nn.ModuleList(nn.Linear(d, dk) for _ in range(n))
        self.k = nn.ModuleList(nn.Linear(d, dk) for _ in range(n)); self.v = nn.ModuleList(nn.Linear(d, d) for _ in range(n))
        self.o = nn.ModuleList(nn.Linear(d, d) for _ in range(n)); self.rate = nn.Parameter(torch.full((n,), -3.)); self.null = nn.Parameter(torch.zeros(()))
        for o in self.o: nn.init.zeros_(o.weight); nn.init.zeros_(o.bias)
        self.last_mass = None

    def forward(self, y, h, ups, starts, rows=128):
        B, L, d = h.shape; K = len(ups); f = [h] + list(ups)                              # f[i]: states of tree level i
        span = [None] + [s.long().cumsum(1) for s in starts]; m = [None] + [sp - 1 for sp in span[1:]]
        ent, par = [None], [None]                                                         # completed-entry gathers per level
        for i in range(1, K + 1):
            ev = starts[i - 1].clone(); ev[:, 0] = False; E = max(int(m[i][:, -1].max()), 1); tt = torch.arange(L)
            pos = torch.where(ev, tt, tt + L).sort(1).values[:, :E]; real = pos < L; end = torch.where(real, pos - 1, torch.zeros_like(pos))
            node = f[i].gather(1, end[..., None].expand(-1, -1, d))
            ent.append(dict(E=E, real=real, key=self.k[i](self.norm[i](node)), val=self.v[i](self.norm[i](node)), end=end))
        for i in range(1, K):                                                             # parent entry of each completed level-i whole
            par.append(span[i + 1].gather(1, ent[i]['end']) - 1)
        par.append(None)
        leaf_key, leaf_val = self.k[0](self.norm[0](h)), self.v[0](self.norm[0](h)); leaf_par = span[1] - 1  # token -> its word
        out = [[] for _ in range(K + 1)]; mass = []
        for r0 in range(0, L, rows):
            r1 = min(r0 + rows, L); R = r1 - r0; tq = torch.arange(r0, r1); qin = y[:, r0:r1]; la_parent = None
            for i in range(K, 0, -1):
                e = ent[i]; E = e['E']; mt = m[i][:, r0:r1]                                # completed wholes at each query time
                q = self.q[i](self.norm[i](qin)); cur_key = self.k[i](self.norm[i](f[i][:, r0:r1])); cur_val = self.v[i](self.norm[i](f[i][:, r0:r1]))
                lag = (mt[:, :, None] - (torch.arange(E) + 1)[None, None, :]).to(y.dtype)
                S = torch.cat([q @ e['key'].transpose(1, 2), (q * cur_key).sum(-1, keepdim=True)], -1)/math.sqrt(self.dk)
                S = S - F.softplus(self.rate[i])*torch.cat([lag.clamp_min(0), torch.zeros_like(lag[..., :1])], -1)
                valid = torch.cat([(torch.arange(E)[None, None, :] < mt[:, :, None]) & e['real'][:, None, :], torch.ones(B, R, 1, dtype=torch.bool)], -1)
                S = torch.where(valid, S, torch.full_like(S, NEG))
                if la_parent is None:                                                     # top level: nodes + null route
                    la = torch.log_softmax(torch.cat([S, self.null.expand(B, R, 1)], -1), -1)[..., :-1]
                else:
                    Ep = ent[i + 1]['E']; pe = par[i][:, None, :].expand(-1, R, -1); mp = m[i + 1][:, r0:r1, None]
                    group = torch.cat([torch.where(pe < mp, pe, torch.full_like(pe, Ep)), torch.full((B, R, 1), Ep, dtype=torch.long)], -1)
                    la = la_parent.gather(2, group) + S - group_lse(S, group, Ep + 1).gather(2, group)
                la = torch.where(valid, la, torch.full_like(la, NEG)); a = torch.exp(la) * valid
                out[i].append(a[..., :E] @ e['val'] + a[..., E:] * cur_val); la_parent = la
            # Leaves: tokens j <= t, grouped under their word (completed, or the in-progress word).
            q = self.q[0](self.norm[0](qin)); S = q @ leaf_key.transpose(1, 2)/math.sqrt(self.dk) - F.softplus(self.rate[0])*(tq[:, None] - torch.arange(L)[None, :]).clamp_min(0).to(y.dtype)
            valid = (torch.arange(L)[None, None, :] <= tq[None, :, None]).expand(B, -1, -1); S = torch.where(valid, S, torch.full_like(S, NEG))
            E1 = ent[1]['E']; pe = leaf_par[:, None, :].expand(-1, R, -1); group = torch.where(pe < m[1][:, r0:r1, None], pe, torch.full_like(pe, E1)).clamp(0, E1)
            la = la_parent.gather(2, group) + S - group_lse(S, group, E1 + 1).gather(2, group); a = torch.exp(torch.where(valid, la, torch.full_like(la, NEG))) * valid
            out[0].append(a @ leaf_val); mass.append(a.sum(-1).detach())
        self.last_mass = torch.cat(mass, 1)
        return sum(self.o[i](torch.cat(out[i], 1)) for i in range(K + 1))


class ConceptHierarchyLM(pw.PartWholeLM):
    def __init__(self, d, modes, layers, heads, dk, dropout, horizons=(4, 16, 32, 64, 256), slots=1024, topk=8, concept_levels=(2, 3, 4), tree=False):
        super().__init__(d, modes, layers, heads, dk, dropout, tuple(horizons))
        self.tree = TreeRead(d, len(horizons)) if tree else None
        self.concept_levels = tuple(concept_levels)
        self.concepts = nn.ModuleDict({str(k): ConceptMemory(d, slots, topk) for k in self.concept_levels})
        self.concept_usage = None

    def states(self, x):
        e = self.embed(x); h = e
        for b in self.blocks:
            h = torch.utils.checkpoint.checkpoint(b, h, e, use_reentrant=False) if (self.checkpoint_blocks and self.training) else b(h, e)
        starts = boundaries5(x)[:len(self.levels)]; below, below_start, ups, usage = h, None, [], {}
        for i, level in enumerate(self.levels):
            args = (below, starts[i], below_start if below_start is not None else starts[i], i == 0)
            u = torch.utils.checkpoint.checkpoint(level, *args, use_reentrant=False) if (self.checkpoint_levels and self.training) else level(*args)
            if str(i) in self.concepts:
                c, idx = self.concepts[str(i)](u); u = u + c; usage[i] = idx.detach()
            ups.append(u); below, below_start = u, starts[i]
        self.last_levels, self.concept_usage = ups, usage
        y = h + self.level_in(self.n_levels(torch.cat(ups, -1)))
        if self.tree is not None:
            y = y + (torch.utils.checkpoint.checkpoint(self.tree, y, h, ups, starts, use_reentrant=False) if self.training else self.tree(y, h, ups, starts))
        y = y + self.mix(self.n_mix(y))
        return self.out_norm(y), e


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--tag', required=True); ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--train-tokens', type=int, default=4194304); ap.add_argument('--lane', type=int, default=1024); ap.add_argument('--d', type=int, default=64)
    ap.add_argument('--modes', type=int, default=32); ap.add_argument('--layers', type=int, default=2); ap.add_argument('--heads', type=int, default=4); ap.add_argument('--dk', type=int, default=16)
    ap.add_argument('--dropout', type=float, default=0.1); ap.add_argument('--lr', type=float, default=3e-3); ap.add_argument('--wd', type=float, default=0.01)
    ap.add_argument('--batch', type=int, default=1); ap.add_argument('--checkpoint-every', type=int, default=256)
    ap.add_argument('--horizons', type=int, nargs='+', default=[4, 16, 32, 64, 256]); ap.add_argument('--future-weight', type=float, default=0.1)
    ap.add_argument('--slots', type=int, default=1024); ap.add_argument('--topk', type=int, default=8); ap.add_argument('--concept-levels', type=int, nargs='*', default=[2, 3, 4]); ap.add_argument('--tree-read', type=int, default=0)
    ap.add_argument('--max-steps', type=int, default=0, help='smoke only: stop after this many updates (no scoring)')
    a = ap.parse_args(); torch.set_num_threads(1); torch.manual_seed(a.seed)
    train_all = load(ROOT/'data/fineweb_gpt2/fineweb_train_000001.bin'); val = load(ROOT/'data/fineweb_gpt2/fineweb_val_000000.bin')
    n_lanes = a.train_tokens//a.lane; train = np.asarray(train_all[:n_lanes*a.lane]).reshape(n_lanes, a.lane)
    order = np.random.default_rng(a.seed).permutation(n_lanes); total = math.ceil(n_lanes/a.batch); quarter = math.ceil(total/4)
    model = ConceptHierarchyLM(a.d, a.modes, a.layers, a.heads, a.dk, a.dropout, tuple(a.horizons), a.slots, a.topk, tuple(a.concept_levels), bool(a.tree_read)); params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd); sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=total, pct_start=0.05)
    out_dir = ROOT/'experiments/results/token_language'; out = out_dir/f'{a.tag}.json'; state_path = out_dir/f'{a.tag}.resume.pt'; assert not out.exists()
    step, losses, aux, milestones, wall, resumes = 0, [], [], [], 0.0, 0
    slot_hits = {k: torch.zeros(a.slots, dtype=torch.long) for k in a.concept_levels}
    if state_path.exists():
        s = torch.load(state_path, weights_only=False); model.load_state_dict(s['model']); opt.load_state_dict(s['opt']); sched.load_state_dict(s['sched'])
        torch.set_rng_state(s['torch_rng']); step, losses, aux, milestones, wall, resumes, slot_hits = s['step'], s['losses'], s['aux'], s['milestones'], s['wall'], s['resumes'] + 1, s['slot_hits']
        assert s['args'] == {k: v for k, v in vars(a).items() if k != 'max_steps'}, 'resume with different settings'
        print(json.dumps(dict(resumed_at_step=step)), flush=True)
    began = time.time() - wall
    def save():
        tmp = state_path.with_suffix('.tmp'); torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(), torch_rng=torch.get_rng_state(), step=step, losses=losses, aux=aux,
                   milestones=milestones, wall=time.time() - began, resumes=resumes, slot_hits=slot_hits, args={k: v for k, v in vars(a).items() if k != 'max_steps'}), tmp); os.replace(tmp, state_path)
    while step < total:
        model.train(); x = torch.from_numpy(train[np.sort(order[step*a.batch:(step + 1)*a.batch])].astype(np.int64)); h, e = model.states(x[:, :-1])
        lg = model.logits(h, e, x[:, :-1], list(range(a.lane - 1))); loss = F.cross_entropy(lg.reshape(-1, V), x[:, 1:].reshape(-1))
        fut = model.future_loss(x) if a.future_weight > 0 else torch.zeros(())
        opt.zero_grad(); (loss + a.future_weight*fut).backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step(); step += 1
        losses.append(loss.item()); aux.append(fut.item())                     # fresh lanes: next-token loss before this update is held out
        for k, idx in model.concept_usage.items(): slot_hits[k] += torch.bincount(idx.reshape(-1), minlength=a.slots)
        if step % 100 == 0: print(json.dumps(dict(step=step, of=total, prequential_last100=float(np.mean(losses[-100:])), future_last100=float(np.mean(aux[-100:])), elapsed_s=round(time.time() - began))), flush=True)
        if step % quarter == 0 or step == total:
            milestones.append(dict(step=step, tokens=step*a.batch*a.lane, select_nll=evaluate_keyed(model, val, 10485760, 8, 2048)['nll'], prequential_segment=float(np.mean(losses[-quarter:])), elapsed_s=round(time.time() - began)))
            print(json.dumps(milestones[-1]), flush=True)
        if step % a.checkpoint_every == 0 or step == total: save()
        if a.max_steps and step >= a.max_steps:
            rec = dict(status='smoke_completed', battle='R1', tag=a.tag, args=vars(a), parameters=params, smoke_steps=step, s_per_step=(time.time() - began)/step,
                       max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, losses=losses, future=aux)
            out.write_text(json.dumps(rec, indent=1) + '\n'); state_path.unlink(missing_ok=True); print(json.dumps({k: rec[k] for k in ('smoke_steps', 's_per_step', 'max_rss_kb')}), flush=True); return
    seg = max(1, total//8); used = {str(k): int((v > 0).sum()) for k, v in slot_hits.items()}
    result = dict(status='completed', battle='R1', stage='concept hierarchy (5 observed levels, concept memories, long lanes), one pass', tag=a.tag, args=vars(a), parameters=params, steps=step,
                  presentations=step*a.batch*(a.lane - 1), train_wall_s=time.time() - began, resumes=resumes, milestones=milestones,
                  prequential_by_eighth=[float(np.mean(losses[i:i + seg])) for i in range(0, len(losses), seg)], prequential_last_quarter=float(np.mean(losses[-quarter:])),
                  future_by_eighth=[float(np.mean(aux[i:i + seg])) for i in range(0, len(aux), seg)], concept_slots_used=used,
                  select=evaluate_keyed(model, val, 10485760, 8, 2048), score_65528=evaluate_keyed(model, val, 20971520, 8, 8192), flagship_protocol=pw.flagship(model, val),
                  max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, reference=dict(baseline='curie_r1_deep_keyed_stream_L2_4M_l1024_s0_20261015_v1'))
    torch.save(model.state_dict(), out_dir/f'{a.tag}.pt'); result['checkpoint'] = f'experiments/results/token_language/{a.tag}.pt'
    result['source_sha256'] = {n: hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in ['experiments/r1_concept_hierarchy_lm.py', 'experiments/r1_part_whole_lm.py', 'experiments/r1_deep_keyed_lm.py', 'experiments/r1_token_keyed_lm.py', 'experiments/tables/gpt2_token_roles.json']}
    out.write_text(json.dumps(result, indent=1) + '\n'); state_path.unlink()
    print('RESULT', json.dumps({k: result[k] for k in ('select', 'score_65528', 'flagship_protocol', 'prequential_last_quarter', 'parameters', 'concept_slots_used')}), flush=True)


if __name__ == '__main__':
    main()
