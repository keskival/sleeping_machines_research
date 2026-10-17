#!/usr/bin/env python3
"""R1 residual-memory race model (RMR-1; theory/172): levels write *corrections* into one additive, persistent, addressed
memory; the next level reads memory + correction as its memory; the race readout reads the total.

Memory: one event slot per token of the lane. Level 1 writes A: slot j holds the base meaning of event j (the CERL state at
j). Level k >= 2, at each step t (levels in order within a step):
- reads M^(<k): A plus every correction by levels < k written at steps <= t, plus its own corrections from earlier chunks
  (own reads lag by one chunk so training stays parallel within a chunk; theory 172 section 7);
- its query is the current slot's value under M^(<k) (what event t means so far); a race over slots s <= t (key match,
  learned recency clock) gives read weights a_ts and the read value;
- writes a correction only: delta_new into the current slot t, and delta_old into the top-m slots it read (hard address,
  magnitude a_ts), so later readers see refined old memories. The write map is zero-initialised: at init every level writes
  nothing and the model equals level 1 (CERL) exactly.
Readout: the CERL race (r1_cerl.CERL) with the refined current slot as query state and refined slot keys for its
repeat/continue pointers. Corrections of old slots enter the pointer scores through the race's S_rep_add/S_cont_add hook,
which is linear in the corrections and therefore computed with matrix products, no [T, T, d] tensors. One objective: the
exact token log-likelihood; credit reaches every level's corrections directly through the total memory (172 P4).
Nothing dense passes between levels except through memory; level reads are races over addressed slots.
"""
import argparse, hashlib, json, math, os, resource, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
ROOT = Path(__file__).resolve().parents[1]; sys.path[:0] = [str(ROOT/'experiments'), str(ROOT/'experiments/tpp')]
from r1_cerl import CERL, word_classes, load_fineweb, lanes_nll, fineweb_protocols, work_per_target, param_groups  # noqa: E402

EXECUTION_ONLY = ('max_steps', 'checkpoint_every', 'log_every', 'row_chunk')
STAGE = 'native residual-memory race model (RMR-1, theory 172), one pass'


class CorrectionLevel(nn.Module):
    def beta(self):
        return F.softplus(self.log_beta) if self.old_decay > 0 else None

    """One residual-memory level: race read over slots, zero-initialised correction write (new slot + read slots)."""
    def __init__(self, d, dk, top_m, read_topk=0, rezero=True, old_decay=0.0, per_target=False):
        super().__init__(); self.dk, self.top_m, self.read_topk, self.rezero = dk, top_m, read_topk, rezero
        self.old_decay, self.per_target = old_decay, per_target and top_m > 0
        if self.per_target:                                                                        # per-target old-slot corrections: delta_old(tau, i) also reads the value of the slot it refines
            self.nt, self.pt = nn.LayerNorm(d), nn.Linear(d, d)
            if not rezero: nn.init.zeros_(self.pt.weight); nn.init.zeros_(self.pt.bias)
        if old_decay > 0: self.log_beta = nn.Parameter(torch.tensor(math.log(math.expm1(old_decay))))   # learned decay rate of old-slot corrections
        self.nq, self.nr = nn.LayerNorm(d), nn.LayerNorm(d); self.q = nn.Linear(d, dk); self.k = nn.Linear(d, dk, bias=False)
        self.log_rate = nn.Parameter(torch.tensor(0.0)); self.g = nn.Sequential(nn.Linear(2*d, 2*d), nn.GELU(), nn.Linear(2*d, 2*d))
        if rezero:                                                                                 # ReZero (Bachlechner et al. 2020): scalar gates at 0, branch at normal init, so gradients flow at step 0
            self.alpha_new, self.alpha_old = nn.Parameter(torch.tensor(0.0)), nn.Parameter(torch.tensor(0.0))
        else:
            nn.init.zeros_(self.g[2].weight); nn.init.zeros_(self.g[2].bias)


class RMR(nn.Module):
    cause_msgs = False

    def __init__(self, V, cls, words, cerl, levels=2, dk=64, top_m=4, chunk=64, read_topk=0, rezero=True, old_decay=0.0, old_per_target=False):
        super().__init__(); self.base = CERL(V, cls, words, **cerl); d = cerl['d']; self.d = d; self.chunk = chunk
        self.levels = nn.ModuleList(CorrectionLevel(d, dk, top_m, read_topk, rezero, old_decay, old_per_target) for _ in range(levels - 1)); self.aux = None; self._px = None

    # ---- memory algebra: a slot value at time t = base' (A + new-slot corrections) + sum over old-slot corrections written at tau <= t
    @staticmethod
    def age_weight(T, t_rows, beta, dtype):
        """Causal mask tau <= t, times exp(-beta (t - tau)) when the writing level decays its old-slot corrections: the total a
        slot can accumulate stays bounded (about 1/beta writes) at any lane length."""
        lag = (t_rows[:, None] - torch.arange(T, device=t_rows.device)[None, :]).to(dtype)
        mk = (lag >= 0).to(dtype)
        return mk if beta is None else mk*torch.exp(-beta*lag.clamp_min(0))

    @staticmethod
    def corr_scores(qv, writes, t_rows, kmap):
        """sum_j sum_{tau<=t} (qv_t . kmap(dold^j_tau)) W^j[tau, s]  ->  [B, R, T]. Writes are sparse: (dold [B,T,d], idx [B,T,m]
        slot addresses, w [B,T,m] magnitudes); memory O(R T m), never [T, T]."""
        out = 0.0
        for wr_ in writes:
            dold, idx, w = wr_[:3]; beta = wr_[3] if len(wr_) > 3 else None
            B, T, m = idx.shape; R = qv.shape[1]
            if dold.dim() == 4:                                                                    # per-target corrections dold [B, T, m, d]: one projection per written (tau, i)
                proj = (qv @ kmap(dold.reshape(B, T*m, -1)).transpose(1, 2)).reshape(B, R, T, m)
                src = (proj*RMR.age_weight(T, t_rows, beta, proj.dtype)[None, :, :, None]*w[:, None]).reshape(B, R, T*m)
            else:
                proj = qv @ kmap(dold).transpose(1, 2)                                             # [B, R, T(tau)]
                proj = proj*RMR.age_weight(T, t_rows, beta, proj.dtype)
                src = (proj[..., None]*w[:, None]).reshape(B, R, T*m)
            out = out + torch.zeros(B, R, T, device=qv.device, dtype=proj.dtype).scatter_add(2, idx.reshape(B, 1, T*m).expand(-1, R, -1), src)
        return out

    @staticmethod
    def corr_values(a, writes, t_rows):
        """sum_s a_ts sum_{tau<=t} W[tau, s] dold_tau  ->  [B, R, d] with sparse writes (gather of the read weights at the written slots)."""
        out = 0.0
        for wr_ in writes:
            dold, idx, w = wr_[:3]; beta = wr_[3] if len(wr_) > 3 else None
            B, T, m = idx.shape; R = a.shape[1]
            c = a.gather(2, idx.reshape(B, 1, T*m).expand(-1, R, -1)).reshape(B, R, T, m)*w[:, None]   # [B, R, T(tau), m]
            if dold.dim() == 4:
                out = out + (c*RMR.age_weight(T, t_rows, beta, c.dtype)[None, :, :, None]).reshape(B, R, T*m) @ dold.reshape(B, T*m, -1)
            else:
                out = out + (c.sum(-1)*RMR.age_weight(T, t_rows, beta, c.dtype)) @ dold
        return out

    def build_memory(self, x):
        """Run every level over the lane: returns the refined current-slot values base' [B, T, d] and the old-slot writes."""
        A = self.base.states(x); B, T, d = A.shape; dv = A.device; new_total = A; writes = []; stats = []
        for lev in self.levels:
            m_ = max(1, lev.top_m); own_new = torch.zeros_like(A); own_dold = A.new_zeros(B, T, m_, d) if lev.per_target else torch.zeros_like(A)
            own_idx = torch.zeros(B, T, m_, dtype=torch.long, device=dv); own_w = torch.zeros(B, T, m_, device=dv)
            for c0 in range(0, T, self.chunk):
                c1 = min(c0 + self.chunk, T); rows = torch.arange(c0, c1, device=dv)
                prev = (torch.arange(T, device=dv) < c0).to(A.dtype)                               # own corrections visible from earlier chunks only
                base_prime = new_total + own_new*prev[None, :, None]                               # slot values from new-slot corrections
                cur = new_total[:, c0:c1]                                                          # current slot under M^(<k) (own write at t not yet made)
                pv = prev.view(1, T, *([1]*(own_dold.dim() - 2)))
                wr = writes + ([(own_dold*pv, own_idx, own_w, lev.beta())] if lev.top_m else [])
                u = lev.nq(cur); q = lev.q(u)                                                      # [B, R, dk]
                S = (q @ lev.k(base_prime).transpose(1, 2) + self.corr_scores(q, wr, rows, lev.k))/math.sqrt(lev.dk)
                dist = (rows[:, None] - torch.arange(T, device=dv)[None, :]).to(A.dtype)
                S = (S - F.softplus(lev.log_rate)*torch.log1p(dist.clamp_min(0))).masked_fill(dist < 0, float('-inf'))
                if lev.read_topk and lev.read_topk < T:                                            # sparse read: race over the top-k slots only (0 = dense softmax over all past slots)
                    kth = S.topk(min(lev.read_topk, T), -1).values[..., -1:]; S = S.masked_fill(S < kth, float('-inf'))
                a = torch.softmax(S, -1)                                                           # read race over slots s <= t
                r = a @ base_prime + self.corr_values(a, wr, rows)
                dnew, dold = lev.g(torch.cat([u, lev.nr(r)], -1)).chunk(2, -1)
                if lev.top_m:                                                                      # write corrections into the top-m older slots it read (0 = dense-layer-style stack: current slot only)
                    ao = a.masked_fill(dist <= 0, 0.0); top = ao.topk(min(lev.top_m, T), -1)
                    own_idx = own_idx.index_copy(1, rows, top.indices); own_w = own_w.index_copy(1, rows, top.values)
                    if lev.per_target:                                                             # each refined slot gets its own correction: shared message + map of that slot's current value
                        tv = base_prime.gather(1, top.indices.reshape(B, -1, 1).expand(-1, -1, d)).reshape(B, c1 - c0, -1, d)
                        dold = dold[:, :, None] + lev.pt(lev.nt(tv))
                        if top.indices.shape[-1] < m_: dold = F.pad(dold, (0, 0, 0, m_ - top.indices.shape[-1]))
                if lev.rezero: dnew, dold = lev.alpha_new*dnew, lev.alpha_old*torch.tanh(dold)       # old-slot corrections bounded: they refine keys other readers rely on
                elif lev.per_target: dold = torch.tanh(dold)
                own_new = own_new.index_copy(1, rows, dnew); own_dold = own_dold.index_copy(1, rows, dold)
            new_total = new_total + own_new; writes = writes + ([(own_dold, own_idx, own_w, lev.beta())] if lev.top_m else [])
            stats.append(dict(new_norm=float(own_new.detach().norm(dim=-1).mean()), old_norm=float(own_dold.detach().norm(dim=-1).mean())))
        return new_total, writes, stats

    def lane_nll(self, x, y, rows=None, stats=False, msg=None):
        if self._px is not x or self._ptrain is not self.training:
            self._mem = self.build_memory(x); self._px = x; self._ptrain = self.training
        new_total, writes, lst = self._mem; B, L = x.shape; r0, r1 = (0, L) if rows is None else rows; m = self.base
        t_rows = torch.arange(r0, r1, device=x.device); hq = new_total[:, r0:r1]
        wr = writes
        td = dict(S_rep_add=self.corr_scores(m.q_rep(hq), wr, t_rows, lambda z: z @ m.k_rep.weight.T)[:, :, :r1]/math.sqrt(m.dk),
                  S_cont_add=self.corr_scores(m.q_cont(hq), wr, t_rows, lambda z: z @ m.k_cont.weight.T)[:, :, :r1]/math.sqrt(m.dk)) if writes else None
        nll, parts, s_sel, sel, lp, s_bg, S_rep, S_cont = m.race(x, y, new_total, r0, r1, self.training and m.explore > 0, td=td)
        self.aux = m.aux
        if not stats: return nll
        with torch.no_grad():
            post = torch.softmax(parts, -1); K = s_sel.shape[-1]
            st = dict(bg_only_nll=(-lp[..., 0]).mean().item(), resp_bg=post[..., 0].mean().item(), resp_concept=post[..., 1:1 + K].sum(-1).mean().item(),
                      resp_repeat=post[..., 1 + K:1 + K + r1].sum(-1).mean().item(), resp_continue=post[..., 1 + K + r1:].sum(-1).mean().item(),
                      **{f'level{i + 2}_{k}': v for i, s_ in enumerate(lst) for k, v in s_.items()})
        return nll, st

    _ptrain = None


def contracts():
    """Zero corrections equal CERL; exact normalisation; causality through old-slot corrections; row chunking; chunk-size invariance."""
    torch.manual_seed(0); V = 61; counts = np.random.default_rng(0).integers(0, 50, V).astype(float); cls, words = word_classes(counts, 6, s_max=16)
    cerl = dict(d=32, modes=8, layers=1, dk=16, units=32, topk=3, explore=0); m = RMR(V, cls, words, cerl, levels=3, dk=16, top_m=3, chunk=8); m.eval(); res = {}
    with torch.no_grad():
        x = torch.randint(0, 8, (1, 40)); y = torch.cat([x[:, 1:], torch.randint(0, 8, (1, 1))], 1)
        res['zero_corrections_equal_cerl'] = float((m.lane_nll(x.clone(), y) - m.base.lane_nll(x, y)).abs().max())
        for p_ in m.parameters(): p_.add_(torch.randn_like(p_)*0.1)
        full = m.lane_nll(x.clone(), y); t = 25; tot = 0.0
        for v in range(V):
            yy = y.clone(); yy[0, t] = v; tot += math.exp(-m.lane_nll(x.clone(), yy)[0, t].item())
        res['normalisation_sum'] = tot
        x2, y2 = x.clone(), y.clone(); p = 30; x2[0, p] = (x2[0, p] + 1) % V; y2[0, p - 1] = x2[0, p]
        res['causal_prefix_max_diff'] = float((m.lane_nll(x2, y2)[0, :p - 1] - full[0, :p - 1]).abs().max())
        xc = x.clone(); res['row_chunk_max_diff'] = float((torch.cat([m.lane_nll(xc, y, (r, min(r + 7, 40))) for r in range(0, 40, 7)], 1) - full).abs().max())
        res['corrections_change_law'] = float((full - m.base.lane_nll(x, y)).abs().max())
    # memory algebra: the sparse-write form equals explicit slot values V_s(t) = sum_{tau<=t} W[tau, s] dold_tau
    g = torch.Generator().manual_seed(1); B_, T_, d_, k_, m_ = 2, 12, 8, 4, 3
    dold = torch.randn(B_, T_, d_, generator=g); idx = torch.stack([torch.stack([torch.randperm(max(1, t), generator=g)[:m_] if t >= m_ else torch.arange(m_) % max(1, t) for t in range(T_)]) for _ in range(B_)])
    w_ = torch.rand(B_, T_, m_, generator=g)*(torch.arange(T_)[None, :, None] >= m_).float()
    W = torch.zeros(B_, T_, T_).scatter_add(2, idx, w_)
    q_ = torch.randn(B_, T_, k_, generator=g); Km = torch.randn(k_, d_, generator=g); a_ = torch.softmax(torch.randn(B_, T_, T_, generator=g), -1); rows_ = torch.arange(T_)
    Vst = torch.stack([torch.einsum('bts,btd->bsd', W[:, :t + 1], dold[:, :t + 1]) for t in range(T_)], 1)
    res['algebra_scores_max_diff'] = float((RMR.corr_scores(q_, [(dold, idx, w_)], rows_, lambda z: z @ Km.T) - torch.einsum('btk,kd,btsd->bts', q_, Km, Vst)).abs().max())
    res['algebra_values_max_diff'] = float((RMR.corr_values(a_, [(dold, idx, w_)], rows_) - torch.einsum('bts,btsd->btd', a_, Vst)).abs().max())
    beta = torch.tensor(0.3); Vd = torch.stack([torch.einsum('bts,btd->bsd', W[:, :t + 1]*torch.exp(-beta*(t - torch.arange(t + 1)))[None, :, None], dold[:, :t + 1]) for t in range(T_)], 1)
    res['algebra_decay_max_diff'] = max(float((RMR.corr_scores(q_, [(dold, idx, w_, beta)], rows_, lambda z: z @ Km.T) - torch.einsum('btk,kd,btsd->bts', q_, Km, Vd)).abs().max()),
                                        float((RMR.corr_values(a_, [(dold, idx, w_, beta)], rows_) - torch.einsum('bts,btsd->btd', a_, Vd)).abs().max()))
    # per-target corrections dold [B, T, m, d]: V_s(t) = sum_{tau<=t} sum_i [idx_{tau,i} = s] w_{tau,i} dold_{tau,i}
    dpt = torch.randn(B_, T_, m_, d_, generator=g)
    Vp = torch.stack([torch.zeros(B_, T_, d_).scatter_add(1, idx[:, :t + 1].reshape(B_, -1, 1).expand(-1, -1, d_), (w_[:, :t + 1, :, None]*dpt[:, :t + 1]*torch.exp(-beta*(t - torch.arange(t + 1)))[None, :, None, None]).reshape(B_, -1, d_)) for t in range(T_)], 1)
    res['algebra_per_target_max_diff'] = max(float((RMR.corr_scores(q_, [(dpt, idx, w_, beta)], rows_, lambda z: z @ Km.T) - torch.einsum('btk,kd,btsd->bts', q_, Km, Vp)).abs().max()),
                                             float((RMR.corr_values(a_, [(dpt, idx, w_, beta)], rows_) - torch.einsum('bts,btsd->btd', a_, Vp)).abs().max()))
    torch.manual_seed(0); mp = RMR(V, cls, words, cerl, levels=3, dk=16, top_m=3, chunk=8, old_decay=0.05, old_per_target=True); mp.eval()
    with torch.no_grad():
        res['per_target_zero_corrections_equal_cerl'] = float((mp.lane_nll(x.clone(), y) - mp.base.lane_nll(x, y)).abs().max())
        for p_ in mp.parameters(): p_.add_(torch.randn_like(p_)*0.1)
        fp = mp.lane_nll(x.clone(), y); tot = 0.0
        for v in range(V):
            yy = y.clone(); yy[0, 25] = v; tot += math.exp(-mp.lane_nll(x.clone(), yy)[0, 25].item())
        res['per_target_normalisation_sum'] = tot
        res['per_target_causal_prefix_max_diff'] = float((mp.lane_nll(x2, y2)[0, :p - 1] - fp[0, :p - 1]).abs().max())
        xc = x.clone(); res['per_target_row_chunk_max_diff'] = float((torch.cat([mp.lane_nll(xc, y, (r, min(r + 7, 40))) for r in range(0, 40, 7)], 1) - fp).abs().max())
        res['per_target_corrections_change_law'] = float((fp - mp.base.lane_nll(x, y)).abs().max())
    ok = res['algebra_per_target_max_diff'] < 1e-4 and res['per_target_zero_corrections_equal_cerl'] < 1e-5 and abs(res['per_target_normalisation_sum'] - 1) < 1e-4 \
        and res['per_target_causal_prefix_max_diff'] < 1e-5 and res['per_target_row_chunk_max_diff'] < 1e-4 and res['per_target_corrections_change_law'] > 0
    ok = ok and res['algebra_decay_max_diff'] < 1e-4 and res['algebra_scores_max_diff'] < 1e-4 and res['algebra_values_max_diff'] < 1e-4 and res['zero_corrections_equal_cerl'] < 1e-5 and abs(res['normalisation_sum'] - 1) < 1e-4 and res['causal_prefix_max_diff'] < 1e-5 and res['row_chunk_max_diff'] < 1e-4 and res['corrections_change_law'] > 0
    res['status'] = 'passed' if ok else 'failed'; return res


def main():
    ap = argparse.ArgumentParser(); A = ap.add_argument
    A('--mode', choices=('train', 'contract'), default='train'); A('--tag', default=''); A('--seed', type=int, default=0)
    A('--d', type=int, default=128); A('--modes', type=int, default=64); A('--layers', type=int, default=2); A('--dk', type=int, default=64)
    A('--units', type=int, default=1024); A('--topk', type=int, default=4); A('--explore', type=int, default=2); A('--classes', type=int, default=256)
    A('--levels', type=int, default=2); A('--read-dk', type=int, default=64); A('--top-m', type=int, default=4); A('--chunk', type=int, default=64); A('--read-topk', type=int, default=0); A('--rezero', type=int, default=1); A('--old-decay', type=float, default=0.0); A('--old-per-target', type=int, default=0); A('--wd-groups', type=int, default=0); A('--unit-init', type=float, default=0.0); A('--balance', type=float, default=0.0)
    A('--train-tokens', type=int, default=4194304); A('--lane', type=int, default=256); A('--batch', type=int, default=4); A('--lr', type=float, default=3e-3); A('--wd', type=float, default=0.01)
    A('--row-chunk', type=int, default=512); A('--checkpoint-every', type=int, default=256); A('--max-steps', type=int, default=0); A('--log-every', type=int, default=100)
    a = ap.parse_args(); torch.set_num_threads(1); torch.manual_seed(a.seed)
    src = {n: hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in ('experiments/r1_rmr.py', 'experiments/r1_cerl.py', 'experiments/r1_deep_keyed_lm.py')}
    if a.mode == 'contract':
        res = dict(battle='R1', tag=a.tag, **contracts(), source_sha256=src); print(json.dumps(res))
        if a.tag:
            with (ROOT/f'experiments/results/diagnostics/{a.tag}.json').open('x') as f: json.dump(res, f, indent=1); f.write('\n')
        return
    train_all = load_fineweb(ROOT/'data/fineweb_gpt2/fineweb_train_000001.bin'); val = load_fineweb(ROOT/'data/fineweb_gpt2/fineweb_val_000000.bin')
    n_lanes = a.train_tokens//a.lane; train = np.asarray(train_all[:n_lanes*a.lane]).reshape(n_lanes, a.lane).astype(np.int64)
    cls, words = word_classes(np.bincount(train.reshape(-1), minlength=50257).astype(float), a.classes)
    cerl = dict(d=a.d, modes=a.modes, layers=a.layers, dk=a.dk, units=a.units, topk=a.topk, explore=a.explore, unit_init=a.unit_init, balance=a.balance)
    model = RMR(50257, cls, words, cerl, levels=a.levels, dk=a.read_dk, top_m=a.top_m, chunk=a.chunk, read_topk=a.read_topk, rezero=bool(a.rezero), old_decay=a.old_decay, old_per_target=bool(a.old_per_target)); params = sum(p.numel() for p in model.parameters())
    order = np.random.default_rng(a.seed).permutation(n_lanes); total = math.ceil(n_lanes/a.batch); quarter = math.ceil(total/4)
    opt = torch.optim.AdamW(param_groups(model, a.wd, bool(a.wd_groups)), lr=a.lr, weight_decay=a.wd); sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=total, pct_start=0.05)
    out_dir = ROOT/'experiments/results/token_language'; out = out_dir/f'{a.tag}.json'; state_path = out_dir/f'{a.tag}.resume.pt'; assert a.tag and not out.exists()
    ident = {k: v for k, v in vars(a).items() if k not in EXECUTION_ONLY}; step, losses, milestones, wall, resumes = 0, [], [], 0.0, 0
    if state_path.exists():
        s = torch.load(state_path, weights_only=False); assert s['args'] == ident
        model.load_state_dict(s['model']); opt.load_state_dict(s['opt']); sched.load_state_dict(s['sched']); torch.set_rng_state(s['torch_rng'])
        step, losses, milestones, wall, resumes = s['step'], s['losses'], s['milestones'], s['wall'], s['resumes'] + 1
    began = time.time() - wall
    def save():
        tmp = state_path.with_suffix('.tmp'); torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(), torch_rng=torch.get_rng_state(), step=step, losses=losses,
                   milestones=milestones, wall=time.time() - began, resumes=resumes, args=ident), tmp); os.replace(tmp, state_path)
    sel_l = np.asarray(val[10485760:10485760 + 8*2048]).reshape(8, 2048)
    while step < total:
        model.train(); w = torch.from_numpy(train[np.sort(order[step*a.batch:(step + 1)*a.batch])]); x, y = w[:, :-1], w[:, 1:]
        nll = model.lane_nll(x, y); loss = nll.mean(); obj = loss + (model.aux.mean() if model.aux is not None else 0.0)
        opt.zero_grad(); obj.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step(); step += 1; losses.append(loss.item())
        if step % a.log_every == 0:
            print(json.dumps(dict(step=step, of=total, prequential_last=float(np.mean(losses[-a.log_every:])), levels=model._mem[2], elapsed_s=round(time.time() - began))), flush=True)
        if step % quarter == 0 or step == total:
            milestones.append(dict(step=step, tokens=step*a.batch*a.lane, select=lanes_nll(model, sel_l, a.row_chunk, stats=True), prequential_segment=float(np.mean(losses[-quarter:])), elapsed_s=round(time.time() - began)))
            print(json.dumps(milestones[-1]), flush=True)
        if step % a.checkpoint_every == 0 or step == total: save()
        if a.max_steps and step >= a.max_steps:
            print(json.dumps(dict(smoke_steps=step, s_per_step=(time.time() - began)/step, max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, parameters=params)), flush=True)
            print(json.dumps(dict(smoke_select_2lanes=lanes_nll(model, sel_l[:2], a.row_chunk, stats=True))), flush=True); return
    from torch.utils.flop_counter import FlopCounterMode
    with FlopCounterMode(display=False) as fc:
        model.train(); w = torch.from_numpy(train[:1]); model.lane_nll(w[:, :-1], w[:, 1:]).mean().backward()
    model.zero_grad(); fpt = fc.get_total_flops()/(a.lane - 1); seg = max(1, total//8)
    result = dict(status='completed', battle='R1', stage=STAGE, tag=a.tag, args=vars(a), parameters=params, steps=step,
                  presentations=step*a.batch*(a.lane - 1), train_wall_s=time.time() - began, resumes=resumes, milestones=milestones, train_flops_per_target=fpt, train_flops=fpt*step*a.batch*(a.lane - 1),
                  prequential_by_eighth=[float(np.mean(losses[i:i + seg])) for i in range(0, len(losses), seg)], prequential_last_quarter=float(np.mean(losses[-quarter:])),
                  **fineweb_protocols(model, val, a.row_chunk), work_level1_per_target=work_per_target(model.base, 255),
                  reference=dict(cerl_4M='curie_r1_cerl_4M_s0_20261016_v1 (flagship 5.4665)', diagnostic_planB_L8_4M_flagship=5.8414), max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, source_sha256=src)
    torch.save(model.state_dict(), out_dir/f'{a.tag}.pt'); result['checkpoint'] = f'experiments/results/token_language/{a.tag}.pt'
    out.write_text(json.dumps(result, indent=1) + '\n'); state_path.unlink()
    print('RESULT', json.dumps({k: result[k] for k in ('select', 'score_65528', 'flagship', 'train_flops_per_target', 'parameters')}), flush=True)


if __name__ == '__main__':
    main()
