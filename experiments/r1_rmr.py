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
from r1_cerl import CERL, word_classes, load_fineweb, lanes_nll, fineweb_protocols, work_per_target, param_groups, same_run  # noqa: E402

EXECUTION_ONLY = ('max_steps', 'checkpoint_every', 'log_every', 'row_chunk')
STAGE = 'native residual-memory race model (RMR-1, theory 172), one pass'


class CorrectionLevel(nn.Module):
    def beta(self):
        return F.softplus(self.log_beta) if self.old_decay > 0 else None

    """One residual-memory level: race read over slots, zero-initialised correction write (new slot + read slots)."""
    def __init__(self, d, dk, top_m, read_topk=0, rezero=True, old_decay=0.0, per_target=False, write_race=False, write_explore=0):
        super().__init__(); self.dk, self.top_m, self.read_topk, self.rezero = dk, top_m, read_topk, rezero
        self.old_decay, self.per_target = old_decay, per_target and top_m > 0
        self.write_race, self.write_explore = write_race and top_m > 0, write_explore if top_m > 0 else 0
        assert not (self.write_explore and not self.write_race), 'write probes credit an address score: use --write-race 1 (probes on read weights would bend reads toward write targets)'
        if self.write_race:                                                                        # separate write-address race (own keys): where to refine is not where to read
            self.qw, self.kw = nn.Linear(d, dk), nn.Linear(d, dk, bias=False)
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

    def __init__(self, V, cls, words, cerl, levels=2, dk=64, top_m=4, chunk=64, read_topk=0, rezero=True, old_decay=0.0, old_per_target=False, write_race=False, write_explore=0):
        super().__init__(); self.base = CERL(V, cls, words, **cerl); d = cerl['d']; self.d = d; self.chunk = chunk
        self.levels = nn.ModuleList(CorrectionLevel(d, dk, top_m, read_topk, rezero, old_decay, old_per_target, write_race, write_explore) for _ in range(levels - 1)); self.aux = None; self._px = None

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
        slot addresses, w [B,T,m] magnitudes); memory O(R T m), never [T, T]. Accumulated in fp32 under autocast: a popular slot
        sums thousands of writes, beyond bf16's 8-bit mantissa."""
        with torch.autocast(qv.device.type, enabled=False):
            return RMR._corr_scores(qv.float(), [tuple(t_.float() if torch.is_tensor(t_) and t_.is_floating_point() else t_ for t_ in wr_) for wr_ in writes], t_rows, kmap)

    @staticmethod
    def _corr_scores(qv, writes, t_rows, kmap):
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
        """sum_s a_ts sum_{tau<=t} W[tau, s] dold_tau  ->  [B, R, d] with sparse writes (gather of the read weights at the written slots); fp32."""
        with torch.autocast(a.device.type, enabled=False):
            return RMR._corr_values(a.float(), [tuple(t_.float() if torch.is_tensor(t_) and t_.is_floating_point() else t_ for t_ in wr_) for wr_ in writes], t_rows)

    @staticmethod
    def _corr_values(a, writes, t_rows):
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

    def build_memory(self, x, prefix=None):
        """Run every level over the lane: returns the refined current-slot values base' [B, T, d] and the old-slot writes.
        prefix (carry_out of the previous lane, detached): persistent memory. Level-1 temporal-memory state continues, and the
        previous lane's slots and every level's corrections stay in memory as readable, refinable (old-slot writes), pointer-
        addressable slots. Positions [0, P) are the prefix, [P, T) this lane; only this lane's rows are computed."""
        A = self.base.states(x, tm_state=None if prefix is None else prefix['tm']); P = 0
        if prefix is not None: P = prefix['A'].shape[1]; A = torch.cat([prefix['A'], A], 1)
        B, T, d = A.shape; dv = A.device; new_total = A; writes = []; stats = []; self._lev_own = []
        for li, lev in enumerate(self.levels):
            E = lev.write_explore if self.training else 0; m_ = max(1, lev.top_m) + E; own_new = torch.zeros_like(A); own_dold = A.new_zeros(B, T, m_, d) if lev.per_target else torch.zeros_like(A)
            own_idx = torch.zeros(B, T, m_, dtype=torch.long, device=dv); own_w = torch.zeros(B, T, m_, device=dv)
            if P:
                pn, pd, pi, pw = prefix['levels'][li]; k_ = pi.shape[-1]
                own_new[:, :P] = pn; own_idx[:, :P, :k_] = pi; own_w[:, :P, :k_] = pw
                if lev.per_target: own_dold[:, :P, :k_] = pd
                else: own_dold[:, :P] = pd
            for c0 in range(P, T, self.chunk):
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
                if lev.top_m:                                                                      # write corrections into top-m older slots (0 = dense-layer-style stack: current slot only)
                    older = dist > 0
                    if lev.write_race:                                                             # address race: own scores; magnitude sigmoid(score) carries credit to the chosen addresses
                        qw_ = lev.qw(u)                                                            # address keys of the refined slots (old-slot corrections included, as in the read)
                        Uw = ((qw_ @ lev.kw(base_prime).transpose(1, 2) + self.corr_scores(qw_, wr, rows, lev.kw))/math.sqrt(lev.dk)).masked_fill(~older, float('-inf'))
                        top = Uw.topk(min(lev.top_m, T), -1); w_sel = torch.sigmoid(top.values)
                    else:                                                                          # address = the slots it read most, magnitude = read weight
                        top = a.masked_fill(~older, 0.0).topk(min(lev.top_m, T), -1); w_sel = top.values
                    idx_sel = top.indices
                    if E:                                                                          # counterfactual write credit (theory 173 C7): E random older slots receive a probe write whose
                        R_ = c1 - c0                                                               # magnitude is p - sg(p): zero in the forward pass (train = eval), gradient
                        ex = (torch.rand(B, R_, E, device=dv)*rows.clamp_min(1)[None, :, None].to(A.dtype)).long()   # dL/dp = <dL/d(slot), probe correction>: "would a write here help?"
                        sc = Uw.gather(-1, ex) if lev.write_race else a.gather(-1, ex)
                        pr = torch.sigmoid(sc) if lev.write_race else sc
                        valid = (rows[None, :, None] > 0) & ~(ex[..., :, None] == idx_sel[..., None, :]).any(-1)
                        idx_sel = torch.cat([idx_sel, ex], -1); w_sel = torch.cat([w_sel, (pr - pr.detach())*valid.to(pr.dtype)], -1)
                    own_idx = own_idx.index_copy(1, rows, idx_sel); own_w = own_w.index_copy(1, rows, w_sel)
                    if lev.per_target:                                                             # each refined slot gets its own correction: shared message + map of that slot's current value
                        tv = self.slot_values(base_prime, wr, idx_sel, rows)                      # the refined value of each target slot (old corrections included)
                        dold = dold[:, :, None] + lev.pt(lev.nt(tv))
                        if idx_sel.shape[-1] < m_: dold = F.pad(dold, (0, 0, 0, m_ - idx_sel.shape[-1]))
                if lev.rezero: dnew, dold = lev.alpha_new*dnew, lev.alpha_old*torch.tanh(dold)       # old-slot corrections bounded: they refine keys other readers rely on
                elif lev.per_target: dold = torch.tanh(dold)
                own_new = own_new.index_copy(1, rows, dnew); own_dold = own_dold.index_copy(1, rows, dold)
            new_total = new_total + own_new; writes = writes + ([(own_dold, own_idx, own_w, lev.beta())] if lev.top_m else [])
            stats.append(dict(new_norm=float(own_new[:, P:].detach().norm(dim=-1).mean()), old_norm=float(own_dold[:, P:].detach().norm(dim=-1).mean())))
            self._lev_own.append((own_new, own_dold, own_idx, own_w))
        self._A, self._P = A, P
        return new_total, writes, stats

    def slot_values(self, base_prime, wr, idx, rows):
        """Refined values V_s(t) of slots idx [B, R, m] at rows t: base' plus all old-slot corrections visible at t (one-hot reads
        through corr_values, so the algebra is the contracted one)."""
        B, R, m = idx.shape; T = base_prime.shape[1]
        oh = torch.zeros(B, R*m, T, device=idx.device, dtype=base_prime.dtype).scatter_(2, idx.reshape(B, R*m, 1), 1.0)
        with torch.no_grad():                                                                      # the old-correction part informs the writer without a second-order credit path
            corr = self.corr_values(oh, wr, rows.repeat_interleave(m)).to(base_prime.dtype)          # (saves O(R m T m) autograd storage per chunk); base' keeps full credit
        v = base_prime.gather(1, idx.reshape(B, -1, 1).expand(-1, -1, base_prime.shape[-1])) + corr
        return v.reshape(B, R, m, -1)

    def carry_out(self, keep):
        """Persistent memory for the next lane: the last `keep` slots (all levels' corrections, detached; credit stays within a
        lane) and the level-1 temporal-memory state. Old-slot writes that addressed dropped slots are dropped; probe entries
        (zero magnitude) are not carried."""
        T = self._A.shape[1]; K = min(keep, T); s0 = T - K                     # any keep is causal; chunk-aligned lanes reproduce one long lane exactly (contract)
        levels = []
        for lev, (on, od, oi, ow) in zip(self.levels, self._lev_own):
            k_ = max(1, lev.top_m); idx = oi[:, s0:, :k_] - s0; ok = idx >= 0
            levels.append((on[:, s0:].detach(), (od[:, s0:, :k_] if od.dim() == 4 else od[:, s0:]).detach(), idx.clamp_min(0), (ow[:, s0:, :k_]*ok).detach()))
        return dict(A=self._A[:, s0:].detach(), levels=levels, tm=[(a_.detach(), b_.detach()) for a_, b_ in self.base.tm_state], x=self._xcat[:, s0:], y=self._ycat[:, s0:])

    def lane_nll(self, x, y, rows=None, stats=False, msg=None, prefix=None):
        if self._px is not x or self._ptrain is not self.training or self._ppf is not prefix:
            self._mem = self.build_memory(x, prefix); self._px = x; self._ptrain = self.training; self._ppf = prefix
            self._xcat = x if prefix is None else torch.cat([prefix['x'], x], 1); self._ycat = y if prefix is None else torch.cat([prefix['y'], y], 1)
        P = self._P; L0 = x.shape[1]; r0, r1 = (0, L0) if rows is None else rows; r0, r1 = r0 + P, r1 + P; x, y = self._xcat, self._ycat
        new_total, writes, lst = self._mem; B, L = x.shape; m = self.base
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

    _ptrain = None; _ppf = None


def lanes_nll_carry(model, block, row_chunk, keep, stats=False):
    """Mean NLL over consecutive lanes [N, L+1] of one stream with persistent memory: each lane starts from the previous lane's
    carried memory (last `keep` slots and the level-1 temporal state). Same targets as lanes_nll; more history."""
    model.eval(); tot = n = 0.0; acc = {}; pf = None
    with torch.no_grad():
        for i in range(len(block)):
            w = torch.from_numpy(block[i:i + 1].astype(np.int64)); x, y = w[:, :-1], w[:, 1:]; L = x.shape[1]
            for r0 in range(0, L, row_chunk):
                out = model.lane_nll(x, y, (r0, min(r0 + row_chunk, L)), stats=stats, prefix=pf); nll, st = (out if stats else (out, {}))
                tot += nll.sum().item(); n += nll.numel()
                for k, v in st.items(): acc[k] = acc.get(k, 0.0) + v*nll.numel()
            pf = model.carry_out(keep)
    r = dict(nll=tot/n, targets=int(n), carry_keep=keep)
    if stats: r['native'] = {k: v/n for k, v in acc.items()}
    return r


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
    # write-address race + counterfactual write probes: eval law exact (zero corrections = CERL, normalisation, causality);
    # probes change no forward value in training (train law with probes == without) but add credit to the address race
    torch.manual_seed(0); mw = RMR(V, cls, words, cerl, levels=3, dk=16, top_m=3, chunk=8, old_decay=0.05, old_per_target=True, write_race=True, write_explore=2); mw.eval()
    with torch.no_grad():
        res['write_race_zero_corrections_equal_cerl'] = float((mw.lane_nll(x.clone(), y) - mw.base.lane_nll(x, y)).abs().max())
        for p_ in mw.parameters(): p_.add_(torch.randn_like(p_)*0.1)
        fw = mw.lane_nll(x.clone(), y)
        res['write_race_normalisation_sum'] = sum(math.exp(-mw.lane_nll(x.clone(), torch.cat([y[:, :25], torch.tensor([[v]]), y[:, 26:]], 1))[0, 25].item()) for v in range(V))
        res['write_race_causal_prefix_max_diff'] = float((mw.lane_nll(x2, y2)[0, :p - 1] - fw[0, :p - 1]).abs().max())
    m0 = RMR(V, cls, words, cerl, levels=3, dk=16, top_m=3, chunk=8, old_decay=0.05, old_per_target=True, write_race=True, write_explore=0); m0.load_state_dict(mw.state_dict())
    mw.train(); m0.train(); l_e = mw.lane_nll(x.clone(), y); l_0 = m0.lane_nll(x.clone(), y)
    res['probes_forward_unchanged'] = float((l_e - l_0).abs().max())
    l_e.mean().backward(); l_0.mean().backward()
    res['probe_credit_to_write_keys'] = float((mw.levels[0].kw.weight.grad - m0.levels[0].kw.weight.grad).abs().max())
    res['probe_no_credit_to_correction_values'] = float((mw.levels[0].g[2].weight.grad - m0.levels[0].g[2].weight.grad).abs().max())
    ok_w = res['write_race_zero_corrections_equal_cerl'] < 1e-5 and abs(res['write_race_normalisation_sum'] - 1) < 1e-4 and res['write_race_causal_prefix_max_diff'] < 1e-5 \
        and res['probes_forward_unchanged'] < 1e-6 and res['probe_credit_to_write_keys'] > 0
    # persistent memory: lane 2 scored with lane 1 carried == rows L..2L of one 2L lane (all slots, corrections, temporal state)
    mw.eval()
    with torch.no_grad():
        xx = torch.randint(0, 8, (2, 81)); xa, ya = xx[:, :-1], xx[:, 1:]; full2 = mw.lane_nll(xa, ya)
        l1 = mw.lane_nll(xa[:, :40], ya[:, :40]); pf = mw.carry_out(40); l2 = mw.lane_nll(xa[:, 40:], ya[:, 40:], prefix=pf)
        res['carry_equals_long_lane'] = max(float((l1 - full2[:, :40]).abs().max()), float((l2 - full2[:, 40:]).abs().max()))
        l2c = torch.cat([mw.lane_nll(xa[:, 40:], ya[:, 40:], (r, min(r + 7, 40)), prefix=pf) for r in range(0, 40, 7)], 1)
        res['carry_row_chunk_max_diff'] = float((l2c - l2).abs().max())
        pf2 = mw.carry_out(16); xb = torch.randint(0, 8, (2, 41)); l3 = mw.lane_nll(xb[:, :-1], xb[:, 1:], prefix=pf2)
        res['carry_window_finite'] = bool(torch.isfinite(l3).all())
    ok = ok_w and res['carry_equals_long_lane'] < 1e-4 and res['carry_row_chunk_max_diff'] < 1e-4 and res['carry_window_finite'] and res['algebra_per_target_max_diff'] < 1e-4 and res['per_target_zero_corrections_equal_cerl'] < 1e-5 and abs(res['per_target_normalisation_sum'] - 1) < 1e-4 \
        and res['per_target_causal_prefix_max_diff'] < 1e-5 and res['per_target_row_chunk_max_diff'] < 1e-4 and res['per_target_corrections_change_law'] > 0
    ok = ok and res['algebra_decay_max_diff'] < 1e-4 and res['algebra_scores_max_diff'] < 1e-4 and res['algebra_values_max_diff'] < 1e-4 and res['zero_corrections_equal_cerl'] < 1e-5 and abs(res['normalisation_sum'] - 1) < 1e-4 and res['causal_prefix_max_diff'] < 1e-5 and res['row_chunk_max_diff'] < 1e-4 and res['corrections_change_law'] > 0
    res['status'] = 'passed' if ok else 'failed'; return res


def carried_protocols(model, val, row_chunk, keep):
    """The three FineWeb protocols scored with persistent memory across their consecutive lanes (same targets)."""
    sel = np.asarray(val[10485760:10485760 + 8*2048]).reshape(8, 2048); sc = np.asarray(val[20971520:20971520 + 8*8192]).reshape(8, 8192)
    fl = np.stack([np.asarray(val[p:p + 1025]) for p in range(25165824, 25165824 + 8192, 1024)])
    return dict(carried=dict(select=lanes_nll_carry(model, sel, row_chunk, keep, True), score_65528=lanes_nll_carry(model, sc, row_chunk, keep, True), flagship=lanes_nll_carry(model, fl, row_chunk, keep, True)))


def main():
    ap = argparse.ArgumentParser(); A = ap.add_argument
    A('--mode', choices=('train', 'contract'), default='train'); A('--tag', default=''); A('--seed', type=int, default=0)
    A('--d', type=int, default=128); A('--modes', type=int, default=64); A('--layers', type=int, default=2); A('--dk', type=int, default=64)
    A('--units', type=int, default=1024); A('--topk', type=int, default=4); A('--explore', type=int, default=2); A('--classes', type=int, default=256)
    A('--levels', type=int, default=2); A('--read-dk', type=int, default=64); A('--top-m', type=int, default=4); A('--chunk', type=int, default=64); A('--read-topk', type=int, default=0); A('--rezero', type=int, default=1); A('--old-decay', type=float, default=0.0); A('--old-per-target', type=int, default=0); A('--write-race', type=int, default=0); A('--carry', type=int, default=0); A('--write-explore', type=int, default=0); A('--wd-groups', type=int, default=0); A('--unit-init', type=float, default=0.0); A('--balance', type=float, default=0.0)
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
    model = RMR(50257, cls, words, cerl, levels=a.levels, dk=a.read_dk, top_m=a.top_m, chunk=a.chunk, read_topk=a.read_topk, rezero=bool(a.rezero), old_decay=a.old_decay, old_per_target=bool(a.old_per_target), write_race=bool(a.write_race), write_explore=a.write_explore); params = sum(p.numel() for p in model.parameters())
    order = np.random.default_rng(a.seed).permutation(n_lanes); total = math.ceil(n_lanes/a.batch); quarter = math.ceil(total/4)
    if a.carry:                                                                                    # persistent memory: batch row b walks its own contiguous stream of lanes (one pass, every lane once)
        per = n_lanes//a.batch; total = per; quarter = math.ceil(total/4)
        order = (np.arange(a.batch)[None, :]*per + np.arange(per)[:, None]).reshape(-1)        # step s uses lanes b*per + s
    opt = torch.optim.AdamW(param_groups(model, a.wd, bool(a.wd_groups)), lr=a.lr, weight_decay=a.wd); sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=total, pct_start=0.05)
    out_dir = ROOT/'experiments/results/token_language'; out = out_dir/f'{a.tag}.json'; state_path = out_dir/f'{a.tag}.resume.pt'; assert a.tag and not out.exists()
    ident = {k: v for k, v in vars(a).items() if k not in EXECUTION_ONLY}; step, losses, milestones, wall, resumes = 0, [], [], 0.0, 0; carry = None
    if state_path.exists():
        s = torch.load(state_path, weights_only=False); assert same_run(s['args'], ident)
        model.load_state_dict(s['model']); opt.load_state_dict(s['opt']); sched.load_state_dict(s['sched']); torch.set_rng_state(s['torch_rng'])
        step, losses, milestones, wall, resumes = s['step'], s['losses'], s['milestones'], s['wall'], s['resumes'] + 1; carry = s.get('carry')
    began = time.time() - wall
    def save():
        tmp = state_path.with_suffix('.tmp'); torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(), torch_rng=torch.get_rng_state(), step=step, losses=losses,
                   milestones=milestones, wall=time.time() - began, resumes=resumes, args=ident, carry=carry), tmp); os.replace(tmp, state_path)
    sel_l = np.asarray(val[10485760:10485760 + 8*2048]).reshape(8, 2048)
    while step < total:
        ids = order[step*a.batch:(step + 1)*a.batch]; model.train(); w = torch.from_numpy(train[ids if a.carry else np.sort(ids)]); x, y = w[:, :-1], w[:, 1:]
        nll = model.lane_nll(x, y, prefix=carry); loss = nll.mean(); obj = loss + (model.aux.mean() if model.aux is not None else 0.0)
        opt.zero_grad(); obj.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step(); step += 1; losses.append(loss.item())
        if a.carry: carry = model.carry_out(a.carry)
        if step % a.log_every == 0:
            print(json.dumps(dict(step=step, of=total, prequential_last=float(np.mean(losses[-a.log_every:])), levels=model._mem[2], elapsed_s=round(time.time() - began))), flush=True)
        if step % quarter == 0 or step == total:
            milestones.append(dict(step=step, tokens=step*a.batch*a.lane, select=lanes_nll(model, sel_l, a.row_chunk, stats=True), **({'select_carried': lanes_nll_carry(model, sel_l, a.row_chunk, a.carry)} if a.carry else {}), prequential_segment=float(np.mean(losses[-quarter:])), elapsed_s=round(time.time() - began)))
            print(json.dumps(milestones[-1]), flush=True)
        if step % a.checkpoint_every == 0 or step == total: save()
        if a.max_steps and step >= a.max_steps:
            print(json.dumps(dict(smoke_steps=step, s_per_step=(time.time() - began)/step, max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, parameters=params)), flush=True)
            print(json.dumps(dict(smoke_select_2lanes=lanes_nll(model, sel_l[:2], a.row_chunk, stats=True))), flush=True); return
    from torch.utils.flop_counter import FlopCounterMode
    pf_ = None
    if a.carry:                                                                                    # count a carried lane: the prefix slots are read, refined and pointed at
        model.train(); w0 = torch.from_numpy(train[:1]); model.lane_nll(w0[:, :-1], w0[:, 1:]); pf_ = model.carry_out(a.carry)
    with FlopCounterMode(display=False) as fc:
        model.train(); w = torch.from_numpy(train[1:2]); model.lane_nll(w[:, :-1], w[:, 1:], prefix=pf_).mean().backward()
    model.zero_grad(); fpt = fc.get_total_flops()/(a.lane - 1); seg = max(1, total//8)
    result = dict(status='completed', battle='R1', stage=STAGE, tag=a.tag, args=vars(a), parameters=params, steps=step,
                  presentations=step*a.batch*(a.lane - 1), train_wall_s=time.time() - began, resumes=resumes, milestones=milestones, train_flops_per_target=fpt, train_flops=fpt*step*a.batch*(a.lane - 1),
                  prequential_by_eighth=[float(np.mean(losses[i:i + seg])) for i in range(0, len(losses), seg)], prequential_last_quarter=float(np.mean(losses[-quarter:])),
                  **fineweb_protocols(model, val, a.row_chunk), **(carried_protocols(model, val, a.row_chunk, a.carry) if a.carry else {}), work_level1_per_target=work_per_target(model.base, 255),
                  reference=dict(cerl_4M='curie_r1_cerl_4M_s0_20261016_v1 (flagship 5.4665)', diagnostic_planB_L8_4M_flagship=5.8414), max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, source_sha256=src)
    torch.save(model.state_dict(), out_dir/f'{a.tag}.pt'); result['checkpoint'] = f'experiments/results/token_language/{a.tag}.pt'
    out.write_text(json.dumps(result, indent=1) + '\n'); state_path.unlink()
    print('RESULT', json.dumps({k: result[k] for k in ('select', 'score_65528', 'flagship', 'train_flops_per_target', 'parameters')}), flush=True)


if __name__ == '__main__':
    main()
