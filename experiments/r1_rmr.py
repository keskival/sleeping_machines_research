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
from native_slot_memory import pk_topk  # noqa: E402
from r1_cerl import CERL, word_classes, load_fineweb, lanes_nll, lanes_nll_usage, record_unit_usage, LocalClip, fineweb_protocols, work_per_target, param_groups, same_run  # noqa: E402

EXECUTION_ONLY = ('max_steps', 'checkpoint_every', 'log_every', 'row_chunk', 'micro')
STAGE = 'native residual-memory race model (RMR-1, theory 172), one pass'


class RegionIndex(nn.Module):
    """Region index over memory slots (theory 173 section 12). Filing: a slot goes to the region its key wins (hard, no gradient;
    region keys move VQ-style). Race: a query's log pi over regions and its top-b regions. product_keys: C = n*n regions scored
    from 2n half-keys; pi is separable, so log pi(a, b) = log softmax(s1)[a] + log softmax(s2)[b] exactly and the top-b regions come
    from the exact product-key top-k. balance > 0: auxiliary-loss-free filing balance (per-region filing biases nudged toward
    uniform load after each training forward, DeepSeek-V3 style); biases are buffers (checkpointed), fixed in evaluation."""
    def __init__(self, C, dk, product_keys=False, balance=0.0):
        super().__init__(); self.C, self.dk, self.pk, self.gamma = C, dk, product_keys, balance
        if product_keys:
            n = int(round(math.sqrt(C))); assert n*n == C, 'product-key regions need a square count'; self.n = n
            self.r1, self.r2 = nn.Parameter(torch.randn(n, dk//2)/math.sqrt(dk)), nn.Parameter(torch.randn(n, dk//2)/math.sqrt(dk))
            self.register_buffer('fb1', torch.zeros(n)); self.register_buffer('fb2', torch.zeros(n))
        else:
            self.reg = nn.Parameter(torch.randn(C, dk)/math.sqrt(dk)); self.register_buffer('fb', torch.zeros(C))

    @torch.no_grad()
    def file(self, keys):
        h = self.dk//2
        if self.pk:
            a = (keys[..., :h] @ self.r1.T + self.fb1).argmax(-1); b = (keys[..., h:] @ self.r2.T + self.fb2).argmax(-1); rid = a*self.n + b
            if self.training and self.gamma:
                for fb, z in ((self.fb1, a), (self.fb2, b)): fb.add_(self.gamma*torch.sign(1.0/self.n - torch.bincount(z.reshape(-1), minlength=self.n).float()/z.numel()))
        else:
            rid = (keys @ self.reg.T + self.fb).argmax(-1)
            if self.training and self.gamma: self.fb.add_(self.gamma*torch.sign(1.0/self.C - torch.bincount(rid.reshape(-1), minlength=self.C).float()/rid.numel()))
        return rid

    def race(self, q, rid, b):
        """q [B, R, dk], rid [B, T] -> (log pi of each slot's region [B, R, T], slot in a top-b region [B, R, T])."""
        sc = 1/math.sqrt(self.dk); R = q.shape[1]
        if self.pk:
            h = self.dk//2; lp1 = torch.log_softmax(q[..., :h] @ self.r1.T*sc, -1); lp2 = torch.log_softmax(q[..., h:] @ self.r2.T*sc, -1)
            at = lp1.gather(-1, (rid//self.n)[:, None, :].expand(-1, R, -1)) + lp2.gather(-1, (rid % self.n)[:, None, :].expand(-1, R, -1))
            _, top = pk_topk(q*sc, self.r1, self.r2, min(b, self.C))
            cand = torch.zeros(*q.shape[:-1], self.C, device=q.device, dtype=torch.bool).scatter(-1, top, True).gather(-1, rid[:, None, :].expand(-1, R, -1))
        else:
            lp = torch.log_softmax(q @ self.reg.T*sc, -1); ro = rid[:, None, :].expand(-1, R, -1); at = lp.gather(-1, ro)
            cand = torch.zeros_like(lp, dtype=torch.bool).scatter(-1, lp.topk(min(b, self.C), -1).indices, True).gather(-1, ro)
        return at, cand

    def scores(self):
        return 2*self.n if self.pk else self.C


class CorrectionLevel(nn.Module):
    def beta(self):
        return F.softplus(self.log_beta) if self.old_decay > 0 else None

    def writes_of(self, dold, idx, w, clk, norm):
        """This level's old-slot write lists: one with the shared rate, or with clocks one list per clock (its writes weighted by the
        clock choice, its own rate), so the contracted memory algebra applies unchanged."""
        if not self.clocks: return [(dold, idx, w, self.beta(), norm)]
        b = self.log_clock.exp(); return [(dold, idx, w*clk[..., k:k + 1], b[k], norm) for k in range(self.clocks)]

    """One residual-memory level: race read over slots, zero-initialised correction write (new slot + read slots)."""
    def __init__(self, d, dk, top_m, read_topk=0, rezero=True, old_decay=0.0, per_target=False, write_race=False, write_explore=0, fire=False, read_explore=0, index_regions=0, index_read=0, index_pk=False, index_balance=0.0, events=False, clocks=0):
        super().__init__(); self.dk, self.top_m, self.read_topk, self.rezero = dk, top_m, read_topk, rezero; self.events = events
        self.clocks = clocks if top_m > 0 else 0
        if self.clocks:                                                                            # per-correction clocks: each write step races over K learned decay rates (log-spaced 1/16 .. 1/8192 at init)
            self.log_clock = nn.Parameter(torch.linspace(math.log(1/16), math.log(1/8192), clocks)); self.clock_sel = nn.Linear(2*d, clocks)
            nn.init.zeros_(self.clock_sel.weight); nn.init.zeros_(self.clock_sel.bias); self.clock_sel.bias.data[min(1, clocks - 1)] = 1.0   # start at the second clock (~128 tokens)
        if events:                                                                                 # event channel (theory 173 section 17): the lower level's pointer and the composed pointer as race candidates,
            self.ev_b = nn.Parameter(torch.zeros(2)); self.ev_t = nn.Linear(3, d); nn.init.zeros_(self.ev_t.weight); nn.init.zeros_(self.ev_t.bias)   # their timing into the query; all zero at init (C3)
        self.index_C, self.index_b = index_regions, index_read                                     # indexed history discovery (theory 173 section 11): slots filed into C regions, reads race over regions first
        self.read_explore = read_explore if (read_topk or index_regions) else 0
        if index_regions: self.index = RegionIndex(index_regions, dk, index_pk, index_balance)
        self.old_decay, self.per_target = old_decay, per_target and top_m > 0
        self.write_race, self.write_explore = write_race and top_m > 0, write_explore if top_m > 0 else 0
        assert not (self.write_explore and not self.write_race), 'write probes credit an address score: use --write-race 1 (probes on read weights would bend reads toward write targets)'
        self.fire = fire
        if fire:                                                                                   # firing clock (RMR-2, now with carry): the level acts at t only when its clock fires
            self.fire_lin = nn.Linear(d + 2 + int(events), 1); nn.init.zeros_(self.fire_lin.weight); nn.init.constant_(self.fire_lin.bias, 2.0)
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

    def __init__(self, V, cls, words, cerl, levels=2, dk=64, top_m=4, chunk=64, read_topk=0, rezero=True, old_decay=0.0, old_per_target=False, write_race=False, write_explore=0, fire=False, act_cost=0.0, old_norm=False, read_explore=0, index_regions=0, index_read=0, readout_index=0, readout_read=0, readout_tok=32, index_pk=False, index_balance=0.0, events=False, old_clocks=0):
        super().__init__(); self.base = CERL(V, cls, words, **cerl)
        assert not (old_clocks and old_norm), 'per-correction clocks normalise per clock list; use one of --old-clocks / --old-norm'; d = cerl['d']; self.d = d; self.chunk = chunk; self.act_cost = act_cost; self.old_norm = old_norm
        self.ro_C, self.ro_b, self.ro_tok = readout_index, readout_read, readout_tok
        if readout_index: self.ro_index = RegionIndex(readout_index, cerl['dk'], index_pk, index_balance)
        self.levels = nn.ModuleList(CorrectionLevel(d, dk, top_m, read_topk, rezero, old_decay, old_per_target, write_race, write_explore, fire, read_explore, index_regions, index_read, index_pk, index_balance, events, old_clocks) for _ in range(levels - 1)); self.aux = None; self._px = None
        self.events = events and levels > 1
        if self.events: self.ro_ev = nn.Parameter(torch.zeros(2))                                   # the top level's pointer and composed pointer enter the readout's repeat/continue race (zero at init)
        self.n_active = levels - 1                                                                 # depth growth (theory 173 section 14): correction levels in use; a level enters at its ReZero gate 0

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
    def slot_mass(idx, w, aw):
        """Accumulated write mass M[t, s] = sum_{tau, i: idx = s} aw[t, tau] w[tau, i]  ->  [B, R, T] (for evidence-weighted old corrections)."""
        B, T, m = idx.shape; R = aw.shape[0]
        return torch.zeros(B, R, T, device=w.device, dtype=w.dtype).scatter_add(2, idx.reshape(B, 1, T*m).expand(-1, R, -1), (aw[None, :, :, None]*w[:, None]).reshape(B, R, T*m))

    @staticmethod
    def _corr_scores(qv, writes, t_rows, kmap):
        out = 0.0
        for wr_ in writes:
            dold, idx, w = wr_[:3]; beta = wr_[3] if len(wr_) > 3 else None; nrm = len(wr_) > 4 and wr_[4]
            B, T, m = idx.shape; R = qv.shape[1]
            if dold.dim() == 4:                                                                    # per-target corrections dold [B, T, m, d]: one projection per written (tau, i)
                proj = (qv @ kmap(dold.reshape(B, T*m, -1)).transpose(1, 2)).reshape(B, R, T, m)
                src = (proj*RMR.age_weight(T, t_rows, beta, proj.dtype)[None, :, :, None]*w[:, None]).reshape(B, R, T*m)
            else:
                proj = qv @ kmap(dold).transpose(1, 2)                                             # [B, R, T(tau)]
                proj = proj*RMR.age_weight(T, t_rows, beta, proj.dtype)
                src = (proj[..., None]*w[:, None]).reshape(B, R, T*m)
            e = torch.zeros(B, R, T, device=qv.device, dtype=proj.dtype).scatter_add(2, idx.reshape(B, 1, T*m).expand(-1, R, -1), src)
            out = out + (e/(1 + RMR.slot_mass(idx, w, RMR.age_weight(T, t_rows, beta, proj.dtype))) if nrm else e)   # --old-norm: a slot's old corrections read as their weighted mean
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
            dold, idx, w = wr_[:3]; beta = wr_[3] if len(wr_) > 3 else None; nrm = len(wr_) > 4 and wr_[4]
            B, T, m = idx.shape; R = a.shape[1]
            ae = a/(1 + RMR.slot_mass(idx, w, RMR.age_weight(T, t_rows, beta, a.dtype))) if nrm else a
            c = ae.gather(2, idx.reshape(B, 1, T*m).expand(-1, R, -1)).reshape(B, R, T, m)*w[:, None]   # [B, R, T(tau), m]
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
        B, T, d = A.shape; dv = A.device; new_total = A; writes = []; stats = []; self._lev_own = []; acts = []
        prev_corr = torch.zeros(B, T, device=dv, dtype=A.dtype); prev_fire = torch.ones(B, T, device=dv, dtype=A.dtype)   # level 1 fires on every event
        prev_ev = None; self._ev = []; self._clk = []; self._phis = []                                                              # (pointer [B, T], its race weight [B, T]) of the level below; level 1 emits no pointer events
        for li, lev in enumerate(self.levels[:self.n_active]):                                    # levels beyond n_active have not entered yet: no compute, no writes, no credit
            E = lev.write_explore if self.training else 0; m_ = max(1, lev.top_m) + E; own_new = torch.zeros_like(A); own_dold = A.new_zeros(B, T, m_, d) if lev.per_target else torch.zeros_like(A)
            own_idx = torch.zeros(B, T, m_, dtype=torch.long, device=dv); own_w = torch.zeros(B, T, m_, device=dv); own_phi = torch.ones(B, T, device=dv, dtype=A.dtype)
            own_ptr = torch.arange(T, device=dv).expand(B, T).clone(); own_pw = torch.ones(B, T, device=dv, dtype=A.dtype)
            own_clk = A.new_zeros(B, T, max(1, lev.clocks))                                       # clock choice per write step (one-hot forward, straight-through credit)   # this level's read-race winner per step (its event pointer) and winning weight
            use_ev = lev.events and prev_ev is not None
            if use_ev:                                                                             # lower events: pointer p1 = ptr(t), composed p2 = ptr(ptr(t)); straight-through weights (forward 1) credit the lower race
                lp, lw = prev_ev; st = lw - lw.detach() + 1.0; p2_all = lp.gather(1, lp); st2_all = st*st.gather(1, lp)
                chg = torch.ones(B, T, device=dv, dtype=A.dtype); chg[:, 1:] = (~((lp[:, 1:] == lp[:, :-1]) | (lp[:, 1:] == lp[:, :-1] + 1))).to(A.dtype)   # event arrival: the lower attribution moved
            if lev.index_C:                                                                        # filing: slot s goes to the region its key wins, from the level's input memory (causal: data <= s)
                region_of = lev.index.file(lev.k(new_total))                                       # [B, T]; hard filing, region keys move VQ-style; stale under later refinement (labelled)
                cand_frac = []
            if P and li < len(prefix['levels']):                                               # a level that entered after the prefix was carried has written nothing there
                pn, pd, pi, pw = prefix['levels'][li][:4]; k_ = pi.shape[-1]
                if len(prefix['levels'][li]) > 4: own_ptr[:, :P], own_pw[:, :P] = prefix['levels'][li][4], prefix['levels'][li][5]
                if len(prefix['levels'][li]) > 6: own_clk[:, :P] = prefix['levels'][li][6]
                own_new[:, :P] = pn; own_idx[:, :P, :k_] = pi; own_w[:, :P, :k_] = pw
                if lev.per_target: own_dold[:, :P, :k_] = pd
                else: own_dold[:, :P] = pd
            for c0 in range(P, T, self.chunk):
                c1 = min(c0 + self.chunk, T); rows = torch.arange(c0, c1, device=dv)
                prev = (torch.arange(T, device=dv) < c0).to(A.dtype)                               # own corrections visible from earlier chunks only
                base_prime = new_total + own_new*prev[None, :, None]                               # slot values from new-slot corrections
                cur = new_total[:, c0:c1]                                                          # current slot under M^(<k) (own write at t not yet made)
                pv = prev.view(1, T, *([1]*(own_dold.dim() - 2)))
                wr = writes + (lev.writes_of(own_dold*pv, own_idx, own_w, own_clk, self.old_norm) if lev.top_m else [])
                u = lev.nq(cur)
                if use_ev:
                    p1, p2 = lp[:, c0:c1], p2_all[:, c0:c1]
                    u = u + lev.ev_t(torch.stack([torch.log1p((rows[None] - p1).to(A.dtype)), torch.log1p((rows[None] - p2).to(A.dtype)), chg[:, c0:c1]], -1))   # event timing (age of the pointed events)
                q = lev.q(u)                                                                       # [B, R, dk]
                S = (q @ lev.k(base_prime).transpose(1, 2) + self.corr_scores(q, wr, rows, lev.k))/math.sqrt(lev.dk)
                dist = (rows[:, None] - torch.arange(T, device=dv)[None, :]).to(A.dtype)
                S = (S - F.softplus(lev.log_rate)*torch.log1p(dist.clamp_min(0))).masked_fill(dist < 0, float('-inf'))
                if use_ev:                                                                         # event-addressed candidates: the lower pointer and the composed pointer get learned bonuses (causal: both <= t)
                    evb = torch.zeros_like(S).scatter_add(-1, p1[..., None], lev.ev_b[0]*st[:, c0:c1, None]).scatter_add(-1, p2[..., None], lev.ev_b[1]*st2_all[:, c0:c1, None])
                    evm = torch.zeros_like(S, dtype=torch.bool).scatter(-1, p1[..., None], True).scatter(-1, p2[..., None], True)
                    S = S + evb
                S_full = None; probe = None
                if lev.index_C:                                                                    # two-stage race: regions (C scores), then slots in the top-b regions; score S + log pi(region)
                    at, cand = lev.index.race(q, region_of, lev.index_b)
                    S_full = S + at
                    cand = cand | (dist == 0)                                                      # the current slot is always readable
                    S = S_full.masked_fill(~cand, float('-inf')); cand_frac.append(float(((cand & (dist >= 0)).sum(-1).float()/(rows[None, :] + 1).float()).mean()))
                S_full = S if S_full is None else S_full
                if lev.read_topk and lev.read_topk < T:                                            # sparse read: race over the top-k slots only (0 = dense softmax over all past slots)
                    kth = S.topk(min(lev.read_topk, T), -1).values[..., -1:]; S = S.masked_fill(S < kth, float('-inf'))
                if use_ev and (lev.read_topk or lev.index_C): S = torch.where(evm, S_full + evb if lev.index_C else S_full, S)   # event pointers are always candidates
                a = torch.softmax(S, -1)                                                           # read race over slots s <= t
                win = a.detach().argmax(-1); own_ptr = own_ptr.index_copy(1, rows, win); own_pw = own_pw.index_copy(1, rows, a.gather(-1, win[..., None]).squeeze(-1))   # this level's event
                if lev.read_explore and self.training:                                             # read probes (theory 173 C7): E unread slots s <= t with zero-forward weight p - sg(p),
                    lz = torch.logsumexp(S, -1, keepdim=True)                                      # p = exp(score - logZ_topk); d/dp = value_s - read: would reading s help?
                    ex = (torch.rand(B, c1 - c0, lev.read_explore, device=dv)*(rows + 1)[None, :, None].to(A.dtype)).long()
                    valid = torch.isinf(S.gather(-1, ex)) & torch.isfinite(S_full.gather(-1, ex))
                    pr = torch.exp(S_full.gather(-1, ex) - lz).masked_fill(~valid, 0.0); qp = pr - pr.detach()
                    a = a.scatter_add(-1, ex, qp); probe = qp.sum(-1)
                r = a @ base_prime + self.corr_values(a, wr, rows)
                if probe is not None: r = r - probe[..., None]*r.detach()
                inp_ = torch.cat([u, lev.nr(r)], -1); dnew, dold = lev.g(inp_).chunk(2, -1)
                if lev.clocks:                                                                     # clock race: hard choice (train = eval), straight-through credit to the selector and rates
                    pc = torch.softmax(lev.clock_sel(inp_), -1); hc = F.one_hot(pc.detach().argmax(-1), lev.clocks).to(pc.dtype)
                    own_clk = own_clk.index_copy(1, rows, hc + pc - pc.detach() if self.training else hc)
                if lev.top_m:                                                                      # write corrections into top-m older slots (0 = dense-layer-style stack: current slot only)
                    older = dist > 0
                    if lev.write_race:                                                             # address race: own scores; magnitude sigmoid(score) carries credit to the chosen addresses
                        qw_ = lev.qw(u)                                                            # address keys of the refined slots (old-slot corrections included, as in the read)
                        Uw = ((qw_ @ lev.kw(base_prime).transpose(1, 2) + self.corr_scores(qw_, wr, rows, lev.kw))/math.sqrt(lev.dk)).masked_fill(~older, float('-inf'))
                        if lev.index_C:                                                            # indexed write discovery: the address race also goes through the region race
                            atw, candw = lev.index.race(qw_, region_of, lev.index_b)
                            Uw = (Uw + atw).masked_fill(~candw, float('-inf'))
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
                if lev.fire:                                                                       # events in (current slot, size of the correction below, its firing) -> fire or stay silent
                    ph = torch.sigmoid(lev.fire_lin(torch.cat([u, prev_corr[:, c0:c1, None], prev_fire[:, c0:c1, None]] + ([chg[:, c0:c1, None]] if lev.events and use_ev else [torch.zeros_like(prev_fire[:, c0:c1, None])] if lev.events else []), -1))).squeeze(-1)
                    hard = (ph > 0.5).to(ph.dtype); phi = hard + ph - ph.detach() if self.training else hard   # hard in training and evaluation; straight-through credit = value of firing
                    dnew = phi[..., None]*dnew; dold = phi.view(*phi.shape, *([1]*(dold.dim() - 2)))*dold; own_phi = own_phi.index_copy(1, rows, phi); acts.append(ph)
                own_new = own_new.index_copy(1, rows, dnew); own_dold = own_dold.index_copy(1, rows, dold)
            new_total = new_total + own_new; writes = writes + (lev.writes_of(own_dold, own_idx, own_w, own_clk, self.old_norm) if lev.top_m else []); self._clk.append(own_clk)
            prev_corr = own_new.norm(dim=-1); prev_fire = own_phi; self._phis.append(own_phi.detach())
            if lev.fire:                                                                           # a silent level emits no event: its pointer holds from its last firing step (event-driven state)
                tt = torch.arange(T, device=dv).expand(B, T); fired = own_phi.detach() > 0.5; fired[:, :max(P, 1)] = True
                last = torch.where(fired, tt, torch.zeros_like(tt)).cummax(1).values; own_ptr = own_ptr.gather(1, last); own_pw = own_pw.gather(1, last)
            prev_ev = (own_ptr, own_pw); self._ev.append(prev_ev)
            if lev.index_C:
                occ = torch.bincount(region_of.reshape(-1), minlength=lev.index_C).float(); occ = occ/occ.sum()
                idx_stats = dict(index_candidate_fraction=float(np.mean(cand_frac)), index_max_region_share=float(occ.max()), index_region_entropy_bits=float(-(occ[occ > 0]*occ[occ > 0].log2()).sum()))
            else: idx_stats = {}
            evs = {}
            if lev.events or self.events:                                                          # event statistics: arrival rate (attribution moved), pointer reach, composed reach
                lpr = own_ptr[:, P:]; ch_ = (~((lpr[:, 1:] == lpr[:, :-1]) | (lpr[:, 1:] == lpr[:, :-1] + 1))).float().mean() if lpr.shape[1] > 1 else torch.tensor(1.0)
                tt_ = torch.arange(P, T, device=dv)[None]; evs = dict(event_rate=float(ch_), pointer_reach=float((tt_ - lpr).float().mean()), composed_reach=float((tt_ - own_ptr.gather(1, own_ptr)[:, P:]).float().mean()))
            if lev.clocks:
                fr_ = own_clk[:, P:].detach().mean((0, 1)); evs = dict(evs, clock_use=[round(float(f_), 4) for f_ in fr_], clock_horizons=[round(float(1/b_), 1) for b_ in lev.log_clock.detach().exp()])
            stats.append(dict(**idx_stats, **evs, new_norm=float(own_new[:, P:].detach().norm(dim=-1).mean()), old_norm=float(own_dold[:, P:].detach().norm(dim=-1).mean()),
                              **({'fire_rate': float(own_phi[:, P:].detach().mean())} if lev.fire else {})))
            self._lev_own.append((own_new, own_dold, own_idx, own_w))
        self._A, self._P = A, P; self._act = torch.cat([a_.reshape(-1) for a_ in acts]).mean() if acts else None
        return new_total, writes, stats

    def slot_values(self, base_prime, wr, idx, rows):
        """Refined values V_s(t) of slots idx [B, R, m] at rows t (one chunk, rows contiguous): base' plus every old-slot correction
        visible at t, with full credit (AWS review 17 Oct: the earlier no_grad form dropped a first-order writer-input path).
        Exact and cheap: writes before the chunk are accumulated per slot at the chunk start (decay factorises:
        e^{-b(t - tau)} = e^{-b(t - c0)} e^{-b(c0 - tau)}; O(T d) per write list), writes inside the chunk by an explicit
        [R, m] x [R, m'] address match (tau <= t)."""
        B, R, m = idx.shape; T, d = base_prime.shape[1:]; c0 = int(rows[0]); dv = idx.device; dt = base_prime.dtype
        v = base_prime.gather(1, idx.reshape(B, -1, 1).expand(-1, -1, d)).reshape(B, R, m, d)
        tt = (rows - c0).to(dt)
        for wr_ in wr:
            dold, widx, w = wr_[:3]; beta = wr_[3] if len(wr_) > 3 else None; nrm = len(wr_) > 4 and wr_[4]; m2 = widx.shape[-1]
            D = dold if dold.dim() == 4 else dold[:, :, None, :].expand(-1, -1, m2, -1)              # [B, T, m', d]
            corr = 0.0; mass = 0.0; dec = 1.0 if beta is None else torch.exp(-beta*tt)[None, :, None]
            if c0 > 0:                                                                             # writes at tau < c0, accumulated per slot at c0
                f = w[:, :c0] if beta is None else w[:, :c0]*torch.exp(-beta*(c0 - torch.arange(c0, device=dv).to(dt)))[None, :, None]
                acc = torch.zeros(B, T, d, device=dv, dtype=dt).scatter_add(1, widx[:, :c0].reshape(B, -1, 1).expand(-1, -1, d), (f[..., None]*D[:, :c0]).reshape(B, -1, d))
                corr = acc.gather(1, idx.reshape(B, -1, 1).expand(-1, -1, d)).reshape(B, R, m, d)*(dec[..., None] if beta is not None else 1.0)
                if nrm: mass = torch.zeros(B, T, device=dv, dtype=dt).scatter_add(1, widx[:, :c0].reshape(B, -1), f.reshape(B, -1)).gather(1, idx.reshape(B, -1)).reshape(B, R, m)*dec
            hit = (idx[:, :, :, None, None] == widx[:, None, None, c0:c0 + R, :]).to(dt)          # [B, R(t), m, R(tau), m']
            lag = tt[:, None] - tt[None, :]; k = (lag >= 0).to(dt) if beta is None else (lag >= 0).to(dt)*torch.exp(-beta*lag.clamp_min(0))
            coef = hit*k[None, :, None, :, None]*w[:, None, None, c0:c0 + R, :]
            corr = corr + torch.einsum('btmun,bund->btmd', coef, D[:, c0:c0 + R])
            v = v + (corr/(1 + mass + coef.sum((-1, -2)))[..., None] if nrm else corr)
        return v

    def slot_values_onehot(self, base_prime, wr, idx, rows):
        """Reference: the same values through the contracted corr_values algebra (one-hot reads; O(R m T m') memory)."""
        B, R, m = idx.shape; T = base_prime.shape[1]
        oh = torch.zeros(B, R*m, T, device=idx.device, dtype=base_prime.dtype).scatter_(2, idx.reshape(B, R*m, 1), 1.0)
        corr = self.corr_values(oh, wr, rows.repeat_interleave(m)).to(base_prime.dtype)
        return (base_prime.gather(1, idx.reshape(B, -1, 1).expand(-1, -1, base_prime.shape[-1])) + corr).reshape(B, R, m, -1)

    def carry_out(self, keep, live=False):
        """Persistent memory for the next lane: the last `keep` slots (all levels' corrections, detached; credit stays within a
        lane) and the level-1 temporal-memory state. Old-slot writes that addressed dropped slots are dropped; probe entries
        (zero magnitude) are not carried. live=True keeps the graph (cross-lane credit, --carry-credit)."""
        D = (lambda t_: t_) if live else (lambda t_: t_.detach())
        T = self._A.shape[1]; K = min(keep, T); s0 = T - K                     # any keep is causal; chunk-aligned lanes reproduce one long lane exactly (contract)
        levels = []
        for li_, (lev, (on, od, oi, ow)) in enumerate(zip(self.levels, self._lev_own)):
            k_ = max(1, lev.top_m); idx = oi[:, s0:, :k_] - s0; ok = idx >= 0
            ev_ = ()
            if self.events and li_ < len(self._ev):
                pt_, pw_ = self._ev[li_]; pt_ = pt_[:, s0:] - s0; self_ = torch.arange(K, device=pt_.device).expand_as(pt_); okp = pt_ >= 0
                ev_ = (torch.where(okp, pt_, self_), torch.where(okp, pw_[:, s0:], torch.ones_like(pw_[:, s0:])))
            if lev.clocks and li_ < len(self._clk):                                                # clock choices travel with their writes (positions 4, 5 hold self-pointers when events are off)
                if not ev_: ev_ = (torch.arange(K, device=on.device).expand(on.shape[0], K).clone(), torch.ones(on.shape[0], K, device=on.device, dtype=on.dtype))
                ev_ = ev_ + (self._clk[li_][:, s0:],)
            levels.append((D(on[:, s0:]), D(od[:, s0:, :k_] if od.dim() == 4 else od[:, s0:]), idx.clamp_min(0), D(ow[:, s0:, :k_]*ok)) + tuple(D(e_) if e_.is_floating_point() else e_ for e_ in ev_))
        return dict(A=D(self._A[:, s0:]), levels=levels, tm=[tuple(D(t_) for t_ in st_) for st_ in self.base.tm_state], x=self._xcat[:, s0:], y=self._ycat[:, s0:])

    def lane_nll(self, x, y, rows=None, stats=False, msg=None, prefix=None):
        if self._px is not x or self._ptrain is not self.training or self._ppf is not prefix or self._pna != self.n_active:
            self._mem = self.build_memory(x, prefix); self._px = x; self._ptrain = self.training; self._ppf = prefix; self._pna = self.n_active
            self._xcat = x if prefix is None else torch.cat([prefix['x'], x], 1); self._ycat = y if prefix is None else torch.cat([prefix['y'], y], 1)
        P = self._P; L0 = x.shape[1]; r0, r1 = (0, L0) if rows is None else rows; r0, r1 = r0 + P, r1 + P; x, y = self._xcat, self._ycat
        new_total, writes, lst = self._mem; B, L = x.shape; m = self.base
        t_rows = torch.arange(r0, r1, device=x.device); hq = new_total[:, r0:r1]
        wr = writes
        td = dict(S_rep_add=self.corr_scores(m.q_rep(hq), wr, t_rows, lambda z: z @ m.k_rep.weight.T)[:, :, :r1]/math.sqrt(m.dk),
                  S_cont_add=self.corr_scores(m.q_cont(hq), wr, t_rows, lambda z: z @ m.k_cont.weight.T)[:, :, :r1]/math.sqrt(m.dk)) if writes else None
        if self.events and self._ev:                                                               # the top level's pointer and composed pointer enter the readout race (repeat and continue causes at those events)
            tp, tw = self._ev[-1]; st_ = tw - tw.detach() + 1.0; q1 = tp[:, r0:r1]; q2 = tp.gather(1, tp)[:, r0:r1]; s1 = st_[:, r0:r1]; s2 = (st_*st_.gather(1, tp))[:, r0:r1]
            eb = x.new_zeros(B, r1 - r0, r1, dtype=new_total.dtype).scatter_add(-1, q1[..., None], self.ro_ev[0]*s1[..., None]).scatter_add(-1, q2[..., None], self.ro_ev[1]*s2[..., None])
            td = dict(S_rep_add=eb, S_cont_add=eb) if td is None else dict(S_rep_add=td['S_rep_add'] + eb, S_cont_add=td['S_cont_add'] + eb)
        ro_cand = None
        if self.ro_C:                                                                              # indexed readout: pointer candidates j from the top-b regions of the query's region race
            R_ = r1 - r0; jj = torch.arange(r1, device=x.device)
            ro_of = self.ro_index.file(m.k_rep(new_total[:, :r1]))                                 # filing of event j by its pointer key (data <= j)
            lg, ro_cand = self.ro_index.race(m.q_rep(hq), ro_of, self.ro_b); ro_cand = ro_cand | (jj[None, None, :] == t_rows[None, :, None])
            add = lg.masked_fill(~ro_cand, float('-inf'))                 # two-stage race: + log pi(region(j)); outside the candidates no pointer
            same = (x[:, None, :r1] == x[:, r0:r1, None]) & (jj[None, None, :] < t_rows[None, :, None])   # token-identity index: earlier occurrences of the current token,
            rec = same.flip(-1).cumsum(-1).flip(-1)                                                # the most recent ro_tok of them (an exact causal hash lookup, no y):
            ro_cont = ro_cand | (same & (rec <= self.ro_tok))                                      # continuation pointers from them are always candidates (induction copy reachable)
            addc = lg.masked_fill(~ro_cont, float('-inf'))
            td = dict(S_rep_add=add, S_cont_add=addc) if td is None else dict(S_rep_add=td['S_rep_add'] + add, S_cont_add=td['S_cont_add'] + addc)
        nll, parts, s_sel, sel, lp, s_bg, S_rep, S_cont = m.race(x, y, new_total, r0, r1, self.training and m.explore > 0, td=td)
        self.aux = m.aux
        if self.training and self.act_cost and self._act is not None:                               # activity cost: firing must pay for itself (hard-firing work is what is counted)
            self.aux = (self.aux if self.aux is not None else 0.0) + self.act_cost*self._act
        if not stats: return nll
        with torch.no_grad():
            post = torch.softmax(parts, -1); K = s_sel.shape[-1]
            st = dict(bg_only_nll=(-lp[..., 0]).mean().item(), resp_bg=post[..., 0].mean().item(), resp_concept=post[..., 1:1 + K].sum(-1).mean().item(),
                      resp_repeat=post[..., 1 + K:1 + K + r1].sum(-1).mean().item(), resp_continue=post[..., 1 + K + r1:].sum(-1).mean().item(),
                      **{f'level{i + 2}_{k}': v for i, s_ in enumerate(lst) for k, v in s_.items()})
            record_unit_usage(m, post, sel, K)
            if ro_cand is not None:                                                                # copy coverage: targets with an earlier copy source that keep one among the candidates
                yq = y[:, r0:r1]; before = jj[None, None, :] <= t_rows[None, :, None]
                src = (x[:, None, :r1] == yq[:, :, None]) & before                                 # repeat sources: x_j = y_t
                nx = torch.cat([x[:, 1:r1], y[:, r1 - 1:r1]], 1); srcc = (nx[:, None, :] == yq[:, :, None]) & (jj[None, None, :] < t_rows[None, :, None])   # continuation sources: x_{j+1} = y_t
                has = src.any(-1) | srcc.any(-1); cov = (src & ro_cand).any(-1) | (srcc & ro_cont).any(-1)
                st['readout_candidate_fraction'] = float(((ro_cand | ro_cont) & before).sum(-1).float().div(t_rows[None, :] + 1).mean())
                st['readout_copy_coverage'] = float((cov & has).sum()/has.sum().clamp_min(1))
        return nll, st

    _ptrain = None; _ppf = None; _pna = None


def carry_rows(c, rs):
    """Rows rs of a carried-memory structure (every tensor is batch-first)."""
    if torch.is_tensor(c): return c[rs]
    if isinstance(c, dict): return {k: carry_rows(v, rs) for k, v in c.items()}
    return type(c)(carry_rows(v, rs) for v in c)


def carry_cat(parts):
    """Concatenate microbatch carries along the batch."""
    c = parts[0]
    if torch.is_tensor(c): return torch.cat(parts, 0)
    if isinstance(c, dict): return {k: carry_cat([p_[k] for p_ in parts]) for k in c}
    return type(c)(carry_cat(list(z)) for z in zip(*parts))


def train_step(model, x, y, carry, keep, micro):
    """One update's forward/backward in `micro` row groups with gradient accumulation (same update; activation memory / micro).
    Exact when no batch statistic enters the objective (load-balance and region-filing statistics are per microbatch)."""
    B = x.shape[0]; loss = 0.0; parts = []
    for rs in np.array_split(np.arange(B), max(1, min(micro, B))):
        rs = torch.from_numpy(rs); nll = model.lane_nll(x[rs], y[rs], prefix=None if carry is None else carry_rows(carry, rs))
        l_ = nll.mean(); ((l_ + (model.aux.mean() if model.aux is not None else 0.0))*len(rs)/B).backward(); loss += l_.item()*len(rs)/B
        if keep: parts.append(model.carry_out(keep))
    return loss, (carry_cat(parts) if keep else None)


def carry_floats(c):
    """The differentiable (floating) tensors of a carried-memory structure, in a fixed order."""
    if torch.is_tensor(c): return [c] if c.is_floating_point() else []
    if isinstance(c, dict): return [t_ for k in sorted(c) for t_ in carry_floats(c[k])]
    return [t_ for v in c for t_ in carry_floats(v)]


def carry_leaves(c):
    """A copy of a carried structure whose floating tensors are fresh leaves that require grad (their gradient is the cotangent on the carry)."""
    if torch.is_tensor(c): return c.detach().requires_grad_(True) if c.is_floating_point() else c
    if isinstance(c, dict): return {k: carry_leaves(v) for k, v in c.items()}
    return type(c)(carry_leaves(v) for v in c)


class CarryCredit:
    """Credit across lane boundaries for persistent memory (theory 160: the adjoint of the memory is the memory run backward; founder: future-
    oriented credit, persistent state separate from the credit window). Each lane runs on a snapshot of the parameters, so optimizer steps do
    not invalidate its graph. A lane's graph is kept for H later lanes: each later lane's cotangent on its carried prefix (lambda) is pushed back
    through the kept lanes' carried outputs, adding to their parameter credit and continuing to the next older lane. A lane's update (its own
    loss plus its effect on the next H lanes) is applied when it leaves the window: updates lag by H lanes. With fixed parameters the summed
    credit equals backprop through H + 1 concatenated lanes exactly (contract)."""
    def __init__(self, model, H):
        self.model, self.H, self.pending = model, H, []                                           # pending: [names, snapshot params, accumulated grads, carry_live, prefix leaves]

    def step(self, x, y, carry, keep):
        from torch.nn.utils.stateless import _reparametrize_module
        m = self.model; names = [n for n, _ in m.named_parameters()]; snap = {n: p_.detach().clone().requires_grad_(True) for n, p_ in m.named_parameters()}
        pf = None if carry is None else carry_leaves(carry); leaves = [] if pf is None else carry_floats(pf)
        with _reparametrize_module(m, snap):
            nll = m.lane_nll(x, y, prefix=pf); l_ = nll.mean(); obj = l_ + (m.aux.mean() if m.aux is not None else 0.0); live = m.carry_out(keep, live=True)
        ps = [snap[n] for n in names]; g = torch.autograd.grad(obj, ps + leaves, allow_unused=True, retain_graph=True)
        acc = [gi if gi is not None else torch.zeros_like(pi) for gi, pi in zip(g[:len(ps)], ps)]; lam = list(g[len(ps):])
        for old in reversed(self.pending):                                                        # newest kept lane first: push lambda back through its carried outputs
            if all(l_i is None for l_i in lam): break
            outs = [(o_, l_i) for o_, l_i in zip(carry_floats(old[3]), lam) if l_i is not None and o_.requires_grad]
            if not outs: break
            gg = torch.autograd.grad([o_ for o_, _ in outs], old[1] + old[4], grad_outputs=[l_i for _, l_i in outs], allow_unused=True, retain_graph=True)
            for k, gk in enumerate(gg[:len(old[1])]):
                if gk is not None: old[2][k] += gk
            lam = list(gg[len(old[1]):])
        self.pending.append([names, ps, acc, live, leaves])
        done = self.pending.pop(0) if len(self.pending) > self.H else None
        return float(l_), (carry_leaves(live) if keep else None), done

    def flush(self):
        out, self.pending = self.pending, []; return out

    @staticmethod
    def apply(model, done):
        """Write a finished lane's accumulated credit into .grad (its snapshot parameters are in model.named_parameters() order)."""
        for p_, g_ in zip(model.parameters(), done[2]): p_.grad = g_.clone()


def member_work(model, history, fire_rates=None, carry=0):
    """Active work per target of the whole member in event-driven execution (AGENTS.md: candidates scored, units woken, state rows
    moved, beside the loss). Modelled from the architecture at `history` visible slots (plus `carry` carried slots); firing rates
    measured on data when given (a silent level skips everything but its clock). Units: multiply-adds (MACs) and counts."""
    b = model.base; d = b.d; H = history + 1 + carry; out = dict(scope='modelled event-driven work per target; fire rates measured' if fire_rates else 'modelled event-driven work per target', history=history, carried=carry)
    lin = lambda mod: sum(p.numel() for n, p in mod.named_parameters() if n.endswith('weight') and p.dim() == 2)   # dense local maps: MACs = weights
    l1 = []
    for t_, to, mlp in zip(b.tm, b.tm_out, b.mlp):
        if getattr(b, 'level1', 'tm') == 'slots':
            keys = 4*t_.n if t_.pk else 2*t_.S; rd = t_.read_topk or t_.S
            l1.append(dict(kind='slots', slots=t_.S, key_scores=keys, rows_written=t_.m, rows_read=rd, macs=keys*t_.dk//(2 if t_.pk else 1) + (t_.m + rd)*d + lin(t_) + lin(to) + lin(mlp)))
        else:
            l1.append(dict(kind='temporal-memory recurrence (dense)', state_size=2*t_.n, macs=lin(t_) + lin(to) + lin(mlp) + 4*t_.n))
    out['level1'] = l1; levels = []
    for k, lev in enumerate(model.levels):
        f = 1.0 if not (lev.fire and fire_rates) else fire_rates[k]
        n_old = sum(max(1, l_.top_m) for l_ in model.levels[:k + 1] if l_.top_m)                 # old-write streams visible to this level (lower levels + own)
        read_keys = (lev.index.scores() + H*lev.index_b/lev.index_C) if lev.index_C else H; corr = read_keys*n_old*lev.dk; vals = lev.read_topk or read_keys   # indexed: C region scores + candidates (balanced filing)
        w_keys = (read_keys if lev.index_C else H) if lev.write_race else 0; local = lin(lev) - (lin(lev.fire_lin) if lev.fire else 0)
        macs = read_keys*lev.dk + corr + vals*d + w_keys*lev.dk + local + (lev.top_m*d*d if lev.per_target else 0)
        levels.append(dict(fire_rate=f, read_keys_scored=read_keys, values_delivered=vals, old_slot_writes=lev.top_m, write_race_keys=w_keys,
                           macs_if_fired=macs, macs=f*macs + (lin(lev.fire_lin) if lev.fire else 0)))
    out['levels'] = levels; out['readout'] = work_per_target(b, H - 1)
    if model.ro_C:                                                                                 # indexed readout: C region scores + pointer candidates in the top-b regions (balanced filing)
        cand = model.ro_index.scores() + 2*H*model.ro_b/model.ro_C + model.ro_tok; out['readout'].update(episodic_candidates=cand, episodic_macs=4*d*b.dk + cand*b.dk)
    ro = out['readout']; ro_macs = ro['episodic_macs'] + ro['unit_macs'] + ro['proposal_macs']
    out['total_macs'] = sum(x['macs'] for x in l1) + sum(x['macs'] for x in levels) + ro_macs
    out['state_rows_moved'] = sum(x.get('rows_written', 0) + x.get('rows_read', 0) for x in l1) + sum(x['fire_rate']*(x['old_slot_writes'] + 1 + x['values_delivered']) for x in levels)
    return out


def lanes_nll_carry(model, block, row_chunk, keep, stats=False, adjacent=False):
    """Mean NLL over consecutive lanes [N, L+1] of one stream with persistent memory: each lane starts from the previous lane's
    carried memory (last `keep` slots and the level-1 temporal state). Same targets as lanes_nll; more history. adjacent=True:
    lanes are back-to-back rows of the stream (lane i's last token is only a target), so that token is fed as one extra
    unscored input to keep the carried input stream contiguous; overlapping windows (flagship) need nothing."""
    model.eval(); tot = n = 0.0; acc = {}; pf = None; dev = next(model.parameters()).device
    with torch.no_grad(), torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == 'cuda'):
        for i in range(len(block)):
            row = block[i] if not (adjacent and i) else np.concatenate([block[i - 1][-1:], block[i]]); s0 = 1 if (adjacent and i) else 0
            w = torch.from_numpy(row[None].astype(np.int64)).to(dev); x, y = w[:, :-1], w[:, 1:]; L = x.shape[1]
            for r0 in range(s0, L, row_chunk):
                out = model.lane_nll(x, y, (r0, min(r0 + row_chunk, L)), stats=stats, prefix=pf); nll, st = (out if stats else (out, {}))
                tot += nll.float().sum().item(); n += nll.numel()
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
    Mp = torch.stack([torch.zeros(B_, T_).scatter_add(1, idx[:, :t + 1].reshape(B_, -1), (w_[:, :t + 1]*torch.exp(-beta*(t - torch.arange(t + 1)))[None, :, None]).reshape(B_, -1)) for t in range(T_)], 1)   # [B, t, s]
    Vn = Vp/(1 + Mp[..., None])
    res['algebra_norm_max_diff'] = max(float((RMR.corr_scores(q_, [(dpt, idx, w_, beta, True)], rows_, lambda z: z @ Km.T) - torch.einsum('btk,kd,btsd->bts', q_, Km, Vn)).abs().max()),
                                       float((RMR.corr_values(a_, [(dpt, idx, w_, beta, True)], rows_) - torch.einsum('bts,btsd->btd', a_, Vn)).abs().max()))
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
        z = torch.randint(0, 8, (1, 82)); full_z = mw.lane_nll(z[:, :-1], z[:, 1:])[0]               # adjacent-lane protocol: carried scoring == one stream on the same targets
        res['carry_adjacent_protocol'] = abs(lanes_nll_carry(mw, z[0].numpy().reshape(2, 41), 16, 64, adjacent=True)['nll'] - float(torch.cat([full_z[:40], full_z[41:]]).mean()))
    # writer slot values: chunk-start accumulation + in-chunk match == one-hot corr_values algebra, values and full gradients
    g2 = torch.Generator().manual_seed(7); Bv, Tv, dv_, mv = 2, 24, 6, 3
    bp = torch.randn(Bv, Tv, dv_, generator=g2, requires_grad=True); dpt2 = torch.randn(Bv, Tv, mv, dv_, generator=g2, requires_grad=True); dsh = torch.randn(Bv, Tv, dv_, generator=g2, requires_grad=True)
    wi = torch.randint(0, Tv, (Bv, Tv, mv), generator=g2); ww = torch.rand(Bv, Tv, mv, generator=g2, requires_grad=True); bt = torch.tensor(0.2, requires_grad=True)
    rr = torch.arange(16, 24); tidx = torch.randint(0, Tv, (Bv, 8, 2), generator=g2); res['writer_values_fast_equals_onehot'] = res['writer_grads_fast_equals_onehot'] = 0.0
    for nrm in (False, True):
        wrs = [(dpt2, wi, ww, bt, nrm), (dsh, wi.flip(1), ww, None, nrm)]
        fa, fb = mw.slot_values(bp, wrs, tidx, rr), mw.slot_values_onehot(bp, wrs, tidx, rr)
        ga = torch.autograd.grad(fa.sum(), (bp, dpt2, dsh, ww, bt)); gb = torch.autograd.grad(fb.sum(), (bp, dpt2, dsh, ww, bt))
        res['writer_values_fast_equals_onehot'] = max(res['writer_values_fast_equals_onehot'], float((fa - fb).abs().max()))
        res['writer_grads_fast_equals_onehot'] = max(res['writer_grads_fast_equals_onehot'], max(float((x_ - y_).abs().max()) for x_, y_ in zip(ga, gb)))
    ok_v = res['algebra_norm_max_diff'] < 1e-4 and res['writer_values_fast_equals_onehot'] < 1e-5 and res['writer_grads_fast_equals_onehot'] < 1e-4
    # firing clocks: zero corrections = CERL; normalised; causal; training forward (straight-through) == hard evaluation; carry exact
    torch.manual_seed(0); mf = RMR(V, cls, words, cerl, levels=3, dk=16, top_m=3, chunk=8, old_decay=0.05, old_per_target=True, write_race=True, write_explore=0, fire=True); mf.eval()
    with torch.no_grad():
        res['fire_zero_corrections_equal_cerl'] = float((mf.lane_nll(x.clone(), y) - mf.base.lane_nll(x, y)).abs().max())
        for p_ in mf.parameters(): p_.add_(torch.randn_like(p_)*0.3)
        ff = mf.lane_nll(x.clone(), y); res['fire_normalisation_sum'] = sum(math.exp(-mf.lane_nll(x.clone(), torch.cat([y[:, :25], torch.tensor([[v]]), y[:, 26:]], 1))[0, 25].item()) for v in range(V))
        res['fire_causal_prefix_max_diff'] = float((mf.lane_nll(x2, y2)[0, :p - 1] - ff[0, :p - 1]).abs().max())
        mf.train(); fft = mf.lane_nll(x.clone(), y); mf.eval(); res['fire_train_equals_eval'] = float((fft - ff).abs().max())
        res['fire_rates'] = [s_['fire_rate'] for s_ in mf._mem[2]]
        full2 = mf.lane_nll(xa, ya); l1 = mf.lane_nll(xa[:, :40], ya[:, :40]); pf = mf.carry_out(40); l2 = mf.lane_nll(xa[:, 40:], ya[:, 40:], prefix=pf)
        res['fire_carry_equals_long_lane'] = max(float((l1 - full2[:, :40]).abs().max()), float((l2 - full2[:, 40:]).abs().max()))
    ok_f = res['fire_zero_corrections_equal_cerl'] < 1e-5 and abs(res['fire_normalisation_sum'] - 1) < 1e-4 and res['fire_causal_prefix_max_diff'] < 1e-5 and res['fire_train_equals_eval'] < 1e-5 \
        and res['fire_carry_equals_long_lane'] < 1e-4 and 0 < min(res['fire_rates']) <= max(res['fire_rates']) <= 1
    # sparse reads with read probes (levels and slot level 1): eval law exact and causal; probes change no training forward value, add credit to read keys
    torch.manual_seed(0); mr = RMR(V, cls, words, dict(cerl, level1='slots', slots=16, slot_m=3, slot_explore=2, slot_read_topk=4), levels=3, dk=16, top_m=3, chunk=8, read_topk=5, old_decay=0.05, old_per_target=True, write_race=True, read_explore=2); mr.eval()
    for t_ in mr.base.tm: t_.CHUNK = 8
    with torch.no_grad():
        for p_ in mr.parameters(): p_.add_(torch.randn_like(p_)*0.1)
        fr = mr.lane_nll(x.clone(), y); res['sparse_read_normalisation_sum'] = sum(math.exp(-mr.lane_nll(x.clone(), torch.cat([y[:, :25], torch.tensor([[v]]), y[:, 26:]], 1))[0, 25].item()) for v in range(V))
        res['sparse_read_causal_prefix_max_diff'] = float((mr.lane_nll(x2, y2)[0, :p - 1] - fr[0, :p - 1]).abs().max())
    m0r = RMR(V, cls, words, dict(cerl, level1='slots', slots=16, slot_m=3, slot_explore=0, slot_read_topk=4), levels=3, dk=16, top_m=3, chunk=8, read_topk=5, old_decay=0.05, old_per_target=True, write_race=True, read_explore=0)
    m0r.load_state_dict(mr.state_dict())
    for t_ in m0r.base.tm: t_.CHUNK = 8
    mr.train(); m0r.train(); le = mr.lane_nll(x.clone(), y); l0 = m0r.lane_nll(x.clone(), y); res['read_probes_forward_unchanged'] = float((le - l0).abs().max())
    le.mean().backward(); l0.mean().backward(); res['read_probe_credit_to_level_keys'] = float((mr.levels[0].k.weight.grad - m0r.levels[0].k.weight.grad).abs().max())
    torch.manual_seed(0); mq = RMR(V, cls, words, dict(cerl, level1='slots', slots=16, slot_m=3, slot_explore=2, slot_read_topk=4, slot_pk=True), levels=3, dk=16, top_m=3, chunk=8, read_topk=5, write_race=True, read_explore=2); mq.eval()
    for t_ in mq.base.tm: t_.CHUNK = 8
    with torch.no_grad():
        for p_ in mq.parameters(): p_.add_(torch.randn_like(p_)*0.1)
        fq = mq.lane_nll(x.clone(), y); res['slot_pk_normalisation_sum'] = sum(math.exp(-mq.lane_nll(x.clone(), torch.cat([y[:, :25], torch.tensor([[v]]), y[:, 26:]], 1))[0, 25].item()) for v in range(V))
        res['slot_pk_causal_prefix_max_diff'] = float((mq.lane_nll(x2, y2)[0, :p - 1] - fq[0, :p - 1]).abs().max())
    mq.eval()
    with torch.no_grad(): mq.lane_nll(x.clone(), y)
    wk = member_work(mq, 39, [s_.get('fire_rate') for s_ in mq._mem[2]]); res['member_work_total_macs'] = wk['total_macs']; res['member_work_state_rows'] = wk['state_rows_moved']
    ok_r = wk['total_macs'] > 0 and wk['levels'][0]['values_delivered'] == 5 and wk['level1'][0]['key_scores'] == 16 and abs(res['slot_pk_normalisation_sum'] - 1) < 1e-4 and res['slot_pk_causal_prefix_max_diff'] < 1e-5 and abs(res['sparse_read_normalisation_sum'] - 1) < 1e-4 and res['sparse_read_causal_prefix_max_diff'] < 1e-5 and res['read_probes_forward_unchanged'] < 1e-5 and res['read_probe_credit_to_level_keys'] > 0
    # indexed history discovery: normalised, causal, carry exact; candidates are a fraction of history; b = C reads every slot
    torch.manual_seed(0); mi = RMR(V, cls, words, cerl, levels=3, dk=16, top_m=3, chunk=8, read_topk=4, old_decay=0.05, old_per_target=True, write_race=True, read_explore=2, index_regions=4, index_read=1); mi.eval()
    with torch.no_grad():
        for p_ in mi.parameters(): p_.add_(torch.randn_like(p_)*0.1)
        fi = mi.lane_nll(x.clone(), y); res['index_normalisation_sum'] = sum(math.exp(-mi.lane_nll(x.clone(), torch.cat([y[:, :25], torch.tensor([[v]]), y[:, 26:]], 1))[0, 25].item()) for v in range(V))
        res['index_causal_prefix_max_diff'] = float((mi.lane_nll(x2, y2)[0, :p - 1] - fi[0, :p - 1]).abs().max())
        mi.lane_nll(x.clone(), y); res['index_candidate_fraction'] = mi._mem[2][0]['index_candidate_fraction']
        full2 = mi.lane_nll(xa, ya); l1 = mi.lane_nll(xa[:, :40], ya[:, :40]); pf = mi.carry_out(40); l2 = mi.lane_nll(xa[:, 40:], ya[:, 40:], prefix=pf)
        res['index_carry_equals_long_lane'] = max(float((l1 - full2[:, :40]).abs().max()), float((l2 - full2[:, 40:]).abs().max()))
        for l_ in mi.levels: l_.index_b = 4
        mi.lane_nll(x.clone(), y); res['index_all_regions_fraction'] = mi._mem[2][0]['index_candidate_fraction']
    mi.train(); [setattr(l_, 'index_b', 1) for l_ in mi.levels]; li = mi.lane_nll(x.clone(), y); li.mean().backward(); res['index_region_keys_credited'] = float(mi.levels[0].index.reg.grad.abs().max())
    # product-key regions (separable exact pi, exact top-b) and balanced filing: law exact, causal, carry; biases move only in training
    torch.manual_seed(0); mk_ = RMR(V, cls, words, cerl, levels=3, dk=16, top_m=3, chunk=8, read_topk=4, write_race=True, read_explore=2, index_regions=16, index_read=2, readout_index=16, readout_read=4, index_pk=True, index_balance=0.01); mk_.eval()
    with torch.no_grad():
        for p_ in mk_.parameters(): p_.add_(torch.randn_like(p_)*0.1)
        fk = mk_.lane_nll(x.clone(), y); res['index_pk_normalisation_sum'] = sum(math.exp(-mk_.lane_nll(x.clone(), torch.cat([y[:, :25], torch.tensor([[v]]), y[:, 26:]], 1))[0, 25].item()) for v in range(V))
        res['index_pk_causal_prefix_max_diff'] = float((mk_.lane_nll(x2, y2)[0, :p - 1] - fk[0, :p - 1]).abs().max())
        full2 = mk_.lane_nll(xa, ya); l1 = mk_.lane_nll(xa[:, :40], ya[:, :40]); pf = mk_.carry_out(40); l2 = mk_.lane_nll(xa[:, 40:], ya[:, 40:], prefix=pf)
        res['index_pk_carry_equals_long_lane'] = max(float((l1 - full2[:, :40]).abs().max()), float((l2 - full2[:, 40:]).abs().max()))
        qq_ = torch.randn(2, 3, 16); ix = mk_.ro_index; lp1 = torch.log_softmax(qq_[..., :8] @ ix.r1.T/4, -1); lp2 = torch.log_softmax(qq_[..., 8:] @ ix.r2.T/4, -1)
        res['index_pk_pi_sums_to_one'] = float((lp1[..., :, None] + lp2[..., None, :]).exp().sum((-1, -2)).sub(1).abs().max())
        fb0 = ix.fb1.clone()
    mk_.train(); mk_.lane_nll(x.clone(), y); res['index_balance_biases_moved'] = float((mk_.ro_index.fb1 - fb0).abs().max())
    ok_k = abs(res['index_pk_normalisation_sum'] - 1) < 1e-4 and res['index_pk_causal_prefix_max_diff'] < 1e-5 and res['index_pk_carry_equals_long_lane'] < 1e-4 and res['index_pk_pi_sums_to_one'] < 1e-5 and res['index_balance_biases_moved'] > 0
    torch.manual_seed(0); mo = RMR(V, cls, words, cerl, levels=3, dk=16, top_m=3, chunk=8, old_decay=0.05, write_race=True, index_regions=4, index_read=1, readout_index=4, readout_read=2); mo.eval()
    with torch.no_grad():
        for p_ in mo.parameters(): p_.add_(torch.randn_like(p_)*0.1)
        fo = mo.lane_nll(x.clone(), y); res['readout_index_normalisation_sum'] = sum(math.exp(-mo.lane_nll(x.clone(), torch.cat([y[:, :25], torch.tensor([[v]]), y[:, 26:]], 1))[0, 25].item()) for v in range(V))
        res['readout_index_causal_prefix_max_diff'] = float((mo.lane_nll(x2, y2)[0, :p - 1] - fo[0, :p - 1]).abs().max())
        _, sto = mo.lane_nll(x.clone(), y, stats=True); res['readout_candidate_fraction'] = sto['readout_candidate_fraction']; res['readout_copy_coverage'] = sto['readout_copy_coverage']
        full2 = mo.lane_nll(xa, ya); l1 = mo.lane_nll(xa[:, :40], ya[:, :40]); pf = mo.carry_out(40); l2 = mo.lane_nll(xa[:, 40:], ya[:, 40:], prefix=pf)
        res['readout_index_carry_equals_long_lane'] = max(float((l1 - full2[:, :40]).abs().max()), float((l2 - full2[:, 40:]).abs().max()))
        mo.ro_b = 4; _, sto4 = mo.lane_nll(x.clone(), y, stats=True); res['readout_all_regions_coverage'] = sto4['readout_copy_coverage']
    ok_i = abs(res['readout_index_normalisation_sum'] - 1) < 1e-4 and res['readout_index_causal_prefix_max_diff'] < 1e-5 and res['readout_index_carry_equals_long_lane'] < 1e-4 \
        and res['readout_candidate_fraction'] < 1 and 0 <= res['readout_copy_coverage'] <= 1 and abs(res['readout_all_regions_coverage'] - 1) < 1e-6 and abs(res['index_normalisation_sum'] - 1) < 1e-4 and res['index_causal_prefix_max_diff'] < 1e-5 and res['index_carry_equals_long_lane'] < 1e-4 and res['index_candidate_fraction'] < 1 \
        and abs(res['index_all_regions_fraction'] - 1) < 1e-6 and res['index_region_keys_credited'] > 0
    # evidence-weighted old corrections through the whole member: exact law, causal, carry
    torch.manual_seed(0); mn = RMR(V, cls, words, cerl, levels=3, dk=16, top_m=3, chunk=8, old_decay=0.05, old_per_target=True, write_race=True, old_norm=True); mn.eval()
    with torch.no_grad():
        for p_ in mn.parameters(): p_.add_(torch.randn_like(p_)*0.1)
        fn = mn.lane_nll(x.clone(), y); res['norm_normalisation_sum'] = sum(math.exp(-mn.lane_nll(x.clone(), torch.cat([y[:, :25], torch.tensor([[v]]), y[:, 26:]], 1))[0, 25].item()) for v in range(V))
        res['norm_causal_prefix_max_diff'] = float((mn.lane_nll(x2, y2)[0, :p - 1] - fn[0, :p - 1]).abs().max())
        full2 = mn.lane_nll(xa, ya); l1 = mn.lane_nll(xa[:, :40], ya[:, :40]); pf = mn.carry_out(40); l2 = mn.lane_nll(xa[:, 40:], ya[:, 40:], prefix=pf)
        res['norm_carry_equals_long_lane'] = max(float((l1 - full2[:, :40]).abs().max()), float((l2 - full2[:, 40:]).abs().max()))
    ok_n = abs(res['norm_normalisation_sum'] - 1) < 1e-4 and res['norm_causal_prefix_max_diff'] < 1e-5 and res['norm_carry_equals_long_lane'] < 1e-4
    # native level 1 (sparse addressed slots): exact law and persistent memory through the whole member
    torch.manual_seed(0); ms = RMR(V, cls, words, dict(cerl, level1='slots', slots=16, slot_m=3), levels=3, dk=16, top_m=3, chunk=8, old_decay=0.05, old_per_target=True, write_race=True, write_explore=2); ms.eval()
    for t_ in ms.base.tm: t_.CHUNK = 8
    with torch.no_grad():
        res['slots_zero_corrections_equal_cerl'] = float((ms.lane_nll(x.clone(), y) - ms.base.lane_nll(x, y)).abs().max())
        for p_ in ms.parameters(): p_.add_(torch.randn_like(p_)*0.1)
        res['slots_normalisation_sum'] = sum(math.exp(-ms.lane_nll(x.clone(), torch.cat([y[:, :25], torch.tensor([[v]]), y[:, 26:]], 1))[0, 25].item()) for v in range(V))
        fs = ms.lane_nll(x.clone(), y); res['slots_causal_prefix_max_diff'] = float((ms.lane_nll(x2, y2)[0, :p - 1] - fs[0, :p - 1]).abs().max())
        full2 = ms.lane_nll(xa, ya); l1 = ms.lane_nll(xa[:, :40], ya[:, :40]); pf = ms.carry_out(40); l2 = ms.lane_nll(xa[:, 40:], ya[:, 40:], prefix=pf)
        res['slots_carry_equals_long_lane'] = max(float((l1 - full2[:, :40]).abs().max()), float((l2 - full2[:, 40:]).abs().max()))
    # depth growth: a level that enters at its ReZero gate 0 leaves the law unchanged (insertion is exact, C3); levels not yet in use
    # receive no credit; n_active = 0 is CERL; growth composes with persistent memory (a level entering after the prefix was carried)
    torch.manual_seed(0); mg = RMR(V, cls, words, cerl, levels=3, dk=16, top_m=3, chunk=8, old_decay=0.05, old_per_target=True, write_race=True); mg.eval()
    with torch.no_grad():
        for p_ in mg.parameters(): p_.add_(torch.randn_like(p_)*0.1)
        mg.levels[1].alpha_new.zero_(); mg.levels[1].alpha_old.zero_()
        mg.n_active = 1; g1 = mg.lane_nll(x.clone(), y); mg.n_active = 2; g2 = mg.lane_nll(x.clone(), y); res['grow_insertion_exact'] = float((g1 - g2).abs().max())
        mg.n_active = 0; res['grow_zero_levels_equal_cerl'] = float((mg.lane_nll(x.clone(), y) - mg.base.lane_nll(x, y)).abs().max())
        mg.n_active = 1; l1 = mg.lane_nll(xa[:, :40], ya[:, :40]); pf = mg.carry_out(40); mg.n_active = 2; l2g = mg.lane_nll(xa[:, 40:], ya[:, 40:], prefix=pf)
        mg.n_active = 1; l2a = mg.lane_nll(xa[:, 40:], ya[:, 40:], prefix=pf); res['grow_enter_after_carry_exact'] = float((l2g - l2a).abs().max())
    mg.train(); mg.n_active = 1; mg.lane_nll(x.clone(), y).mean().backward()
    res['grow_inactive_level_no_credit'] = all(p_.grad is None for p_ in mg.levels[1].parameters()); res['grow_active_level_credit'] = any(p_.grad is not None and float(p_.grad.abs().max()) > 0 for p_ in mg.levels[0].parameters())
    # conditioning credit (as in a residual Transformer, where layer l gets gradient through the identity path and through every later
    # layer that reads it): an upper level's corrections depend on the lower level's new-slot and old-slot corrections, and gating the
    # upper level off changes the gradient on the lower level's corrections (CERL-H and the detached local learner lacked this path)
    torch.manual_seed(0); mc = RMR(V, cls, words, cerl, levels=3, dk=16, top_m=3, chunk=8, old_decay=0.05, old_per_target=True, write_race=True); mc.train()
    with torch.no_grad():
        for p_ in mc.parameters(): p_.add_(torch.randn_like(p_)*0.1)
    lc = mc.lane_nll(x.clone(), y).mean(); (on1, od1), (on2, od2) = mc._lev_own[0][:2], mc._lev_own[1][:2]
    gc = torch.autograd.grad(on2.sum() + od2.sum(), [on1, od1], retain_graph=True); gt_ = torch.autograd.grad(lc, [on1])[0]
    mc.levels[1].alpha_new.data.zero_(); mc.levels[1].alpha_old.data.zero_(); mc._px = None; gd_ = torch.autograd.grad(mc.lane_nll(x.clone(), y).mean(), [mc._lev_own[0][0]])[0]
    res['conditioning_credit_new_slot'] = float(gc[0].abs().max()); res['conditioning_credit_old_slot'] = float(gc[1].abs().max()); res['conditioning_credit_share'] = float((gt_ - gd_).norm()/gt_.norm())
    # event channel (theory 173 section 17): zero at init equals the member without events; exact law, causal, carry exact with fire;
    # event bonuses credit the lower level's read race; composed pointers reach further than single pointers
    NAT = dict(levels=4, dk=16, top_m=3, chunk=8, read_topk=6, old_decay=0.05, old_per_target=True, write_race=True, write_explore=2, read_explore=2, fire=True)
    NAT0 = dict(NAT, read_topk=0, read_explore=0)                                                  # identity at init on dense reads (with sparse reads, event pointers are always candidates by design)
    torch.manual_seed(0); mz = RMR(V, cls, words, cerl, events=True, **NAT0); mz0 = RMR(V, cls, words, cerl, events=False, **NAT0)
    with torch.no_grad():
        for n_, p_ in mz.named_parameters():
            if 'ev_' not in n_ and 'ro_ev' not in n_: p_.add_(torch.randn_like(p_)*0.1)
            if 'fire_lin.weight' in n_: p_[:, -1] = 0.0
    mz0.load_state_dict({k: (v[:, :-1] if k.endswith('fire_lin.weight') else v) for k, v in mz.state_dict().items() if k in mz0.state_dict()}); mz.eval(); mz0.eval()
    with torch.no_grad(): res['events_zero_equal_no_events'] = float((mz.lane_nll(x.clone(), y) - mz0.lane_nll(x.clone(), y)).abs().max())
    torch.manual_seed(0); me = RMR(V, cls, words, cerl, events=True, **NAT)
    with torch.no_grad():
        for p_ in me.parameters(): p_.add_(torch.randn_like(p_)*0.1)
        for n_, p_ in me.named_parameters():
            if 'ev_' in n_ or 'ro_ev' in n_: p_.zero_()
            if 'fire_lin.weight' in n_: p_[:, -1] = 0.0
    me.eval()
    with torch.no_grad():
        for n_, p_ in me.named_parameters():
            if 'ev_' in n_ or 'ro_ev' in n_ or 'fire_lin' in n_: p_.add_(torch.randn_like(p_)*0.5)
        fe = me.lane_nll(x.clone(), y); res['events_normalisation_sum'] = sum(math.exp(-me.lane_nll(x.clone(), torch.cat([y[:, :25], torch.tensor([[v]]), y[:, 26:]], 1))[0, 25].item()) for v in range(V))
        res['events_causal_prefix_max_diff'] = float((me.lane_nll(x2, y2)[0, :p - 1] - fe[0, :p - 1]).abs().max())
        me.train(); res['events_train_equals_eval'] = float((me.lane_nll(x.clone(), y) - fe).abs().max()); me.eval()
        full2 = me.lane_nll(xa, ya); l1 = me.lane_nll(xa[:, :40], ya[:, :40]); pf = me.carry_out(40); l2 = me.lane_nll(xa[:, 40:], ya[:, 40:], prefix=pf)
        res['events_carry_equals_long_lane'] = max(float((l1 - full2[:, :40]).abs().max()), float((l2 - full2[:, 40:]).abs().max()))
        me.lane_nll(xa, ya); st_e = me._mem[2]; res['events_rates'] = [s_['event_rate'] for s_ in st_e]; res['events_reach'] = [(s_['pointer_reach'], s_['composed_reach']) for s_ in st_e]
    me.train(); me.lane_nll(x.clone(), y).mean().backward(); ge = me.levels[0].k.weight.grad.clone(); me.zero_grad()
    with torch.no_grad():
        for n_, p_ in me.named_parameters():
            if 'ev_b' in n_ or 'ro_ev' in n_: p_.zero_()
    me.lane_nll(x.clone(), y).mean().backward(); res['events_credit_to_lower_read_keys'] = float((ge - me.levels[0].k.weight.grad).abs().max())
    ok_e = res['events_zero_equal_no_events'] < 1e-6 and abs(res['events_normalisation_sum'] - 1) < 1e-4 and res['events_causal_prefix_max_diff'] < 1e-5 and res['events_train_equals_eval'] < 1e-5 \
        and res['events_carry_equals_long_lane'] < 1e-4 and res['events_credit_to_lower_read_keys'] > 0 and all(c_ >= r_ - 1e-9 for r_, c_ in res['events_reach'])
    # calibrated bank-local clipping: banks partition the parameters; after calibration each bank is scaled on its own (direction preserved
    # inside every bank, no cross-bank dependence: scaling one bank's gradient leaves the others' clipped gradients unchanged)
    torch.manual_seed(0); mlc = RMR(V, cls, words, cerl, levels=3, dk=16, top_m=3, chunk=8, write_race=True); lc_ = LocalClip(mlc, calib=2)
    res['local_clip_banks_partition'] = sum(len(v) for v in lc_.banks.values()) == len(list(mlc.parameters())); mlc.train()
    for _ in range(2): mlc.zero_grad(); mlc.lane_nll(x.clone(), y).mean().backward(); lc_(mlc)
    lc_.caps = {b: 1e-3 for b in lc_.caps}; mlc.zero_grad(); mlc.lane_nll(x.clone(), y).mean().backward()
    g0 = {n: p.grad.clone() for n, p in mlc.named_parameters() if p.grad is not None}; lc_(mlc); g1 = {n: p.grad.clone() for n, p in mlc.named_parameters() if p.grad is not None}
    cosd = 0.0
    for b, ps in lc_.banks.items():
        names = [n for n, p in mlc.named_parameters() if any(p is q for q in ps) and n in g0]
        if names:
            a_ = torch.cat([g0[n].reshape(-1) for n in names]); b_ = torch.cat([g1[n].reshape(-1) for n in names])
            if a_.norm() > 0: cosd = max(cosd, float(1 - (a_ @ b_)/(a_.norm()*b_.norm())))
    res['local_clip_direction_preserved'] = cosd; res['local_clip_banks'] = len(lc_.banks)
    torch.manual_seed(0); mm = RMR(V, cls, words, dict(cerl, level1='slots', slots=16, slot_m=3, slot_read_topk=4), events=True, **dict(NAT, write_explore=0, read_explore=0)); mm.train()
    for t_ in mm.base.tm: t_.CHUNK = 8
    xb4 = torch.randint(0, 8, (4, 41)); pfm = None; out_m = []
    for mic in (1, 2):
        torch.manual_seed(5); mm.zero_grad(); l0_, c0_ = train_step(mm, xb4[:, :20], xb4[:, 1:21], None, 16, mic); l1_, c1_ = train_step(mm, xb4[:, 20:40], xb4[:, 21:41], c0_, 16, mic)
        out_m.append((l0_ + l1_, [p_.grad.clone() if p_.grad is not None else None for p_ in mm.parameters()], c1_['A']))
    res['micro_loss_diff'] = abs(out_m[0][0] - out_m[1][0]); res['micro_grad_diff'] = max(float((g1 - g2).abs().max()) for g1, g2 in zip(out_m[0][1], out_m[1][1]) if g1 is not None)
    res['micro_carry_diff'] = float((out_m[0][2] - out_m[1][2]).abs().max())
    ok_e = ok_e and res['micro_loss_diff'] < 1e-5 and res['micro_grad_diff'] < 1e-5 and res['micro_carry_diff'] < 1e-5
    # cross-lane credit: with fixed parameters the summed credit of CarryCredit(H) equals backprop through H + 1 concatenated carried lanes
    torch.manual_seed(0); mx = RMR(V, cls, words, dict(cerl, level1='slots', slots=16, slot_m=3, slot_read_topk=4), events=True, **dict(NAT, write_explore=0, read_explore=0)); mx.eval()
    for t_ in mx.base.tm: t_.CHUNK = 8
    with torch.no_grad():
        for p_ in mx.parameters(): p_.add_(torch.randn_like(p_)*0.1)
    xs = torch.randint(0, 8, (2, 61)); lanes_ = [(xs[:, i*20:(i + 1)*20], xs[:, i*20 + 1:(i + 1)*20 + 1]) for i in range(3)]
    for H, nl in ((1, 2), (2, 3)):
        mx.zero_grad(); pf = None; tot = 0.0
        for xl, yl in lanes_[:nl]:
            tot = tot + mx.lane_nll(xl, yl, prefix=pf).mean(); pf = mx.carry_out(16, live=True)
        ref = torch.autograd.grad(tot, list(mx.parameters()), allow_unused=True)
        ccx = CarryCredit(mx, H); car = None; got = None
        for xl, yl in lanes_[:nl]:
            _, car, dn = ccx.step(xl, yl, car, 16)
            if dn is not None: got = [g_ for g_ in dn[2]] if got is None else [a_ + b_ for a_, b_ in zip(got, dn[2])]
        for dn in ccx.flush(): got = [g_ for g_ in dn[2]] if got is None else [a_ + b_ for a_, b_ in zip(got, dn[2])]
        res[f'carry_credit_H{H}_equals_bptt'] = max(float(((r_ if r_ is not None else torch.zeros_like(g_)) - g_).abs().max()) for r_, g_ in zip(ref, got))
        mx.zero_grad(); pf = None; tot = 0.0
        for xl, yl in lanes_[:nl]:
            tot = tot + mx.lane_nll(xl, yl, prefix=pf).mean(); pf = mx.carry_out(16)
        det = torch.autograd.grad(tot, list(mx.parameters()), allow_unused=True)
        res[f'carry_credit_H{H}_cross_lane_part'] = max(float(((r_ if r_ is not None else 0) - (d_ if d_ is not None else 0)).abs().max()) if (r_ is not None or d_ is not None) else 0.0 for r_, d_ in zip(ref, det))
    ok_e = ok_e and res['carry_credit_H1_equals_bptt'] < 1e-5 and res['carry_credit_H2_equals_bptt'] < 1e-5 and res['carry_credit_H1_cross_lane_part'] > 0
    # online adaptation with persistent memory: causal (a later token changes no earlier score), the first segment equals frozen carried scoring,
    # later segments adapt; weights are restored
    torch.manual_seed(0); mo2 = RMR(V, cls, words, dict(cerl, level1='slots', slots=16, slot_m=3, slot_read_topk=4), events=True, **dict(NAT, write_explore=0, read_explore=0)); mo2.eval()
    for t_ in mo2.base.tm: t_.CHUNK = 8
    blk = np.random.default_rng(3).integers(0, 8, (2, 49)); msz = [torch.ones_like(p_) for p_ in mo2.parameters()]; w0 = [p_.detach().clone() for p_ in mo2.parameters()]
    r1_, o1_ = online_carried(mo2, blk, 16, msz, 1e-2, 0.01, sub=16, adjacent=True); blk2 = blk.copy(); blk2[1, 40] = (blk2[1, 40] + 1) % 8
    r2_, o2_ = online_carried(mo2, blk2, 16, msz, 1e-2, 0.01, sub=16, adjacent=True)
    q_ = 48 + 39 - 1                                                                               # scored targets before the changed input (row 0: 48; row 1: 39 after its unscored first)
    res['online_causal_prefix_max_diff'] = float((o1_[:q_] - o2_[:q_]).abs().max())
    with torch.no_grad(): fz = mo2.lane_nll(torch.from_numpy(blk[:1, :16]), torch.from_numpy(blk[:1, 1:17]))[0]
    res['online_first_segment_equals_frozen'] = float((o1_[:16] - fz).abs().max()); _, o0_ = online_carried(mo2, blk, 16, msz, 0.0, 0.0, sub=16, adjacent=True)
    res['online_adapts'] = float((o1_ - o0_).abs().max()); res['online_weights_restored'] = max(float((p_ - w_).abs().max()) for p_, w_ in zip(mo2.parameters(), w0))
    ok_e = ok_e and res['online_causal_prefix_max_diff'] < 1e-5 and res['online_first_segment_equals_frozen'] < 1e-5 and res['online_adapts'] > 0 and res['online_weights_restored'] == 0
    # per-correction clocks: with all K rates equal the member equals the shared-rate member exactly (the one-hot choice sums to 1); exact law,
    # causal, train = eval, carry exact; credit reaches the clock selector and the rates
    torch.manual_seed(0); CK = dict(levels=3, dk=16, top_m=3, chunk=8, old_per_target=True, write_race=True)
    mk1 = RMR(V, cls, words, cerl, old_decay=0.05, **CK); mkc = RMR(V, cls, words, cerl, old_clocks=4, **CK)
    with torch.no_grad():
        for p_ in mk1.parameters(): p_.add_(torch.randn_like(p_)*0.1)
    mkc.load_state_dict(mk1.state_dict(), strict=False)
    with torch.no_grad():
        for l1_, lc_ in zip(mk1.levels, mkc.levels): lc_.log_clock.fill_(float(F.softplus(l1_.log_beta).log())); lc_.clock_sel.weight.normal_(0, 0.5)
    mk1.eval(); mkc.eval()
    with torch.no_grad(): res['clocks_equal_rates_equal_shared'] = float((mkc.lane_nll(x.clone(), y) - mk1.lane_nll(x.clone(), y)).abs().max())
    with torch.no_grad():
        for lc_ in mkc.levels: lc_.log_clock.copy_(torch.linspace(math.log(1/4), math.log(1/64), 4))
        fc = mkc.lane_nll(x.clone(), y); res['clocks_normalisation_sum'] = sum(math.exp(-mkc.lane_nll(x.clone(), torch.cat([y[:, :25], torch.tensor([[v]]), y[:, 26:]], 1))[0, 25].item()) for v in range(V))
        res['clocks_causal_prefix_max_diff'] = float((mkc.lane_nll(x2, y2)[0, :p - 1] - fc[0, :p - 1]).abs().max())
        mkc.train(); res['clocks_train_equals_eval'] = float((mkc.lane_nll(x.clone(), y) - fc).abs().max()); mkc.eval()
        full2 = mkc.lane_nll(xa, ya); l1 = mkc.lane_nll(xa[:, :40], ya[:, :40]); pf = mkc.carry_out(40); l2 = mkc.lane_nll(xa[:, 40:], ya[:, 40:], prefix=pf)
        res['clocks_carry_equals_long_lane'] = max(float((l1 - full2[:, :40]).abs().max()), float((l2 - full2[:, 40:]).abs().max()))
    mkc.train(); mkc.zero_grad(); mkc.lane_nll(x.clone(), y).mean().backward()
    res['clocks_credit_to_selector'] = float(mkc.levels[0].clock_sel.weight.grad.abs().max()); res['clocks_credit_to_rates'] = float(mkc.levels[0].log_clock.grad.abs().max())
    ok_e = ok_e and res['clocks_equal_rates_equal_shared'] < 1e-5 and abs(res['clocks_normalisation_sum'] - 1) < 1e-4 and res['clocks_causal_prefix_max_diff'] < 1e-5 and res['clocks_train_equals_eval'] < 1e-5 \
        and res['clocks_carry_equals_long_lane'] < 1e-4 and res['clocks_credit_to_selector'] > 0 and res['clocks_credit_to_rates'] > 0
    # silence (theory 174 P2): a level that never fires and has zero gates receives no gradient at all (absorbing); with a nonzero gate its firing
    # decision is credited at silent steps (straight-through value of firing), so it can revive
    torch.manual_seed(0); ms_ = RMR(V, cls, words, cerl, levels=3, dk=16, top_m=3, chunk=8, write_race=True, fire=True); ms_.train()
    with torch.no_grad():
        for p_ in ms_.parameters(): p_.add_(torch.randn_like(p_)*0.1)
        for l_ in ms_.levels: l_.fire_lin.weight.zero_(); l_.fire_lin.bias.fill_(-10.0); l_.alpha_new.zero_(); l_.alpha_old.zero_()
    ms_.zero_grad(); ms_.lane_nll(x.clone(), y).mean().backward()
    res['silence_zero_gate_absorbing'] = max(float(p_.grad.abs().max()) if p_.grad is not None else 0.0 for p_ in ms_.levels[1].parameters())
    with torch.no_grad():
        for l_ in ms_.levels: l_.alpha_new.fill_(0.5); l_.alpha_old.fill_(0.5)
    ms_.zero_grad(); ms_.lane_nll(x.clone(), y).mean().backward(); res['silence_revival_credit'] = float(ms_.levels[1].fire_lin.weight.grad.abs().max())
    ok_e = ok_e and res['silence_zero_gate_absorbing'] == 0.0 and res['silence_revival_credit'] > 0
    mg.eval(); mg.n_active = 2; ru = lanes_nll_usage(mg, xx.numpy()[:, :41], 16)                 # usable capacity: unit responsibilities add up to the concept cause's share
    res['capacity_units_used'] = ru['capacity']['units_used']; res['capacity_mass_matches_resp_concept'] = abs(ru['capacity']['concept_mass_per_target'] - ru['native']['resp_concept'])
    ok_g = res['local_clip_banks_partition'] and res['local_clip_direction_preserved'] < 1e-5 and ok_e and res['conditioning_credit_new_slot'] > 0 and res['conditioning_credit_old_slot'] > 0 and res['conditioning_credit_share'] > 0 and res['capacity_units_used'] > 0 and res['capacity_mass_matches_resp_concept'] < 1e-6 and res['grow_insertion_exact'] < 1e-6 and res['grow_zero_levels_equal_cerl'] < 1e-5 and res['grow_enter_after_carry_exact'] < 1e-6 and res['grow_inactive_level_no_credit'] and res['grow_active_level_credit']
    ok_s = ok_g and res['slots_zero_corrections_equal_cerl'] < 1e-5 and abs(res['slots_normalisation_sum'] - 1) < 1e-4 and res['slots_causal_prefix_max_diff'] < 1e-5 and res['slots_carry_equals_long_lane'] < 1e-4
    ok = ok_k and ok_i and ok_r and ok_n and ok_f and ok_v and ok_s and ok_w and res['carry_adjacent_protocol'] < 1e-5 and res['carry_equals_long_lane'] < 1e-4 and res['carry_row_chunk_max_diff'] < 1e-4 and res['carry_window_finite'] and res['algebra_per_target_max_diff'] < 1e-4 and res['per_target_zero_corrections_equal_cerl'] < 1e-5 and abs(res['per_target_normalisation_sum'] - 1) < 1e-4 \
        and res['per_target_causal_prefix_max_diff'] < 1e-5 and res['per_target_row_chunk_max_diff'] < 1e-4 and res['per_target_corrections_change_law'] > 0
    ok = ok and res['algebra_decay_max_diff'] < 1e-4 and res['algebra_scores_max_diff'] < 1e-4 and res['algebra_values_max_diff'] < 1e-4 and res['zero_corrections_equal_cerl'] < 1e-5 and abs(res['normalisation_sum'] - 1) < 1e-4 and res['causal_prefix_max_diff'] < 1e-5 and res['row_chunk_max_diff'] < 1e-4 and res['corrections_change_law'] > 0
    res['status'] = 'passed' if ok else 'failed'; return res


def online_carried(model, block, keep, ms, lr, decay, sub=512, adjacent=False):
    """Causal online adaptation with persistent memory (AGENTS.md: online learning on non-stationary streams; dynamic evaluation, Krause et
    al.): the lanes are one stream scored in segments of `sub` tokens with carried memory; after each segment is scored, one RMS-normalised
    step on it with decay toward the trained weights (ms: the optimizer's second moments). Every prediction uses weights adapted only on
    earlier segments and memory written only by earlier tokens; weights are restored afterwards."""
    model.eval(); ps = list(model.parameters()); th0 = [p_.detach().clone() for p_ in ps]; rms = []
    for m_ in ms:
        r = m_.sqrt(); nz = r[r > 0]; rms.append(r.clamp_min(0.5*nz.mean().item() if nz.numel() else 1e-3) + 1e-8)
    mr = torch.cat([r.flatten() for r in rms]).mean(); dec = [(decay*r/mr).clamp(max=1.0) for r in rms]; pf = None; tot = n = 0.0; out = []
    for i in range(len(block)):
        row = block[i] if not (adjacent and i) else np.concatenate([block[i - 1][-1:], block[i]]); first = 1 if (adjacent and i) else 0
        for s0 in range(0, len(row) - 1, sub):
            w = torch.from_numpy(row[s0:s0 + sub + 1][None].astype(np.int64)); x, y = w[:, :-1], w[:, 1:]
            nll = model.lane_nll(x, y, prefix=pf); sc = nll[:, first:] if s0 == 0 else nll; first_ = 0
            tot += float(sc.sum()); n += sc.numel(); out.append(sc.detach()); model.zero_grad(); sc.mean().backward(); pf = model.carry_out(keep)
            with torch.no_grad():
                for p_, r, d_, t0 in zip(ps, rms, dec, th0):
                    if p_.grad is not None: p_.add_(-lr*p_.grad/r + d_*(t0 - p_))
    with torch.no_grad():
        for p_, t0 in zip(ps, th0): p_.copy_(t0)
    model.zero_grad(); return dict(nll=tot/n, targets=int(n), lr=lr, decay=decay, segment=sub, carry_keep=keep), torch.cat([o_.reshape(-1) for o_ in out])


def online_protocols(model, val, keep, opt):
    """Online column (separately labelled from the fixed-weight scores): lr and decay chosen on the selection slice only, then applied to the
    65,528-target slice and the flagship windows."""
    ms = [opt.state[p_]['exp_avg_sq'] if p_ in opt.state and 'exp_avg_sq' in opt.state[p_] else torch.zeros_like(p_) for p_ in model.parameters()]
    sel = np.asarray(val[10485760:10485760 + 8*2048]).reshape(8, 2048); sc = np.asarray(val[20971520:20971520 + 8*8192]).reshape(8, 8192)
    fl = np.stack([np.asarray(val[p:p + 1025]) for p in range(25165824, 25165824 + 8192, 1024)])
    grid = [dict(lr=lr, decay=dc, nll=online_carried(model, sel, keep, ms, lr, dc, adjacent=True)[0]['nll']) for lr in (1e-4, 3e-4, 1e-3) for dc in (0.0, 0.01)]
    b = min(grid, key=lambda r: r['nll'] if math.isfinite(r['nll']) else math.inf)
    return dict(online=dict(scope='causal online adaptation with persistent memory; lr/decay chosen on the selection slice', grid=grid,
                            select=online_carried(model, sel, keep, ms, b['lr'], b['decay'], adjacent=True)[0], score_65528=online_carried(model, sc, keep, ms, b['lr'], b['decay'], adjacent=True)[0],
                            flagship=online_carried(model, fl, keep, ms, b['lr'], b['decay'])[0]))


def carried_protocols(model, val, row_chunk, keep):
    """The three FineWeb protocols scored with persistent memory across their consecutive lanes (same targets)."""
    sel = np.asarray(val[10485760:10485760 + 8*2048]).reshape(8, 2048); sc = np.asarray(val[20971520:20971520 + 8*8192]).reshape(8, 8192)
    fl = np.stack([np.asarray(val[p:p + 1025]) for p in range(25165824, 25165824 + 8192, 1024)])
    return dict(carried=dict(select=lanes_nll_carry(model, sel, row_chunk, keep, True, adjacent=True), score_65528=lanes_nll_carry(model, sc, row_chunk, keep, True, adjacent=True),
                             flagship=lanes_nll_carry(model, fl, row_chunk, keep, True)))


def main():
    ap = argparse.ArgumentParser(); A = ap.add_argument
    A('--mode', choices=('train', 'contract'), default='train'); A('--tag', default=''); A('--seed', type=int, default=0)
    A('--d', type=int, default=128); A('--modes', type=int, default=64); A('--layers', type=int, default=2); A('--dk', type=int, default=64)
    A('--units', type=int, default=1024); A('--topk', type=int, default=4); A('--explore', type=int, default=2); A('--classes', type=int, default=256)
    A('--levels', type=int, default=2); A('--read-dk', type=int, default=64); A('--top-m', type=int, default=4); A('--chunk', type=int, default=64); A('--read-topk', type=int, default=0); A('--rezero', type=int, default=1); A('--old-decay', type=float, default=0.0); A('--old-per-target', type=int, default=0); A('--write-race', type=int, default=0); A('--carry', type=int, default=0); A('--fire', type=int, default=0); A('--read-explore', type=int, default=0); A('--index-regions', type=int, default=0); A('--index-read', type=int, default=0); A('--readout-index', type=int, default=0); A('--readout-read', type=int, default=0); A('--readout-tok', type=int, default=32); A('--index-pk', type=int, default=0); A('--index-balance', type=float, default=0.0); A('--slot-read-topk', type=int, default=0); A('--slot-pk', type=int, default=0); A('--old-norm', type=int, default=0); A('--act-cost', type=float, default=0.0); A('--level1', choices=('tm', 'slots'), default='tm'); A('--product-keys', type=int, default=0); A('--slots', type=int, default=1024); A('--slot-m', type=int, default=4); A('--slot-explore', type=int, default=0); A('--write-explore', type=int, default=0); A('--wd-groups', type=int, default=0); A('--unit-init', type=float, default=0.0); A('--balance', type=float, default=0.0)
    A('--old-clocks', type=int, default=0, help='per-correction clocks: each write step races over K learned decay rates (replaces the shared --old-decay rate)'); A('--carry-credit', type=int, default=0, help='credit across lane boundaries: lanes of future credit for persistent memory (CarryCredit; updates lag by this many lanes)'); A('--micro', type=int, default=1, help='execution: microbatches per update (gradient accumulation; activation memory / micro)'); A('--clip', choices=('global', 'local'), default='global', help='local: calibrated bank-local radial clipping (CALIBRATED_LOCAL_CLIPPING_ADOPTION.md)'); A('--events', type=int, default=0, help='event channel between levels (theory 173 section 17): pointer events, composed pointers, event-driven firing, events into the readout race'); A('--grow', default='', help='depth growth: comma-separated data fractions at which correction levels 2..K enter (one per level, nondecreasing); empty = all from the start')
    A('--train-tokens', type=int, default=4194304); A('--lane', type=int, default=256); A('--batch', type=int, default=4); A('--lr', type=float, default=3e-3); A('--wd', type=float, default=0.01)
    A('--row-chunk', type=int, default=512); A('--checkpoint-every', type=int, default=256); A('--max-steps', type=int, default=0); A('--log-every', type=int, default=100)
    a = ap.parse_args(); torch.set_num_threads(1); torch.manual_seed(a.seed)
    src = {n: hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in ('experiments/r1_rmr.py', 'experiments/r1_cerl.py', 'experiments/r1_deep_keyed_lm.py', 'experiments/native_slot_memory.py')}
    if a.mode == 'contract':
        res = dict(battle='R1', tag=a.tag, **contracts(), source_sha256=src); print(json.dumps(res))
        if a.tag:
            with (ROOT/f'experiments/results/diagnostics/{a.tag}.json').open('x') as f: json.dump(res, f, indent=1); f.write('\n')
        return
    train_all = load_fineweb(ROOT/'data/fineweb_gpt2/fineweb_train_000001.bin'); val = load_fineweb(ROOT/'data/fineweb_gpt2/fineweb_val_000000.bin')
    n_lanes = a.train_tokens//a.lane; train = np.asarray(train_all[:n_lanes*a.lane]).reshape(n_lanes, a.lane).astype(np.int64)
    cls, words = word_classes(np.bincount(train.reshape(-1), minlength=50257).astype(float), a.classes)
    if a.carry:                                                                                    # carried lanes overlap by one token, so the inputs of consecutive lanes are one contiguous
        n_lanes = (a.train_tokens - 1)//(a.lane - 1)                                               # stream (non-overlapping rows drop each lane's last token from the inputs)
        train = np.asarray(train_all[:n_lanes*(a.lane - 1) + 1]).astype(np.int64)[np.arange(n_lanes)[:, None]*(a.lane - 1) + np.arange(a.lane)[None, :]]
    cerl = dict(d=a.d, modes=a.modes, layers=a.layers, dk=a.dk, units=a.units, topk=a.topk, explore=a.explore, product_keys=bool(a.product_keys), unit_init=a.unit_init, balance=a.balance, level1=a.level1, slots=a.slots, slot_m=a.slot_m, slot_explore=a.slot_explore, slot_read_topk=a.slot_read_topk, slot_pk=bool(a.slot_pk))
    model = RMR(50257, cls, words, cerl, levels=a.levels, dk=a.read_dk, top_m=a.top_m, chunk=a.chunk, read_topk=a.read_topk, rezero=bool(a.rezero), old_decay=a.old_decay, old_per_target=bool(a.old_per_target), write_race=bool(a.write_race), write_explore=a.write_explore, fire=bool(a.fire), act_cost=a.act_cost, old_norm=bool(a.old_norm), read_explore=a.read_explore, index_regions=a.index_regions, index_read=a.index_read, readout_index=a.readout_index, readout_read=a.readout_read, readout_tok=a.readout_tok, index_pk=bool(a.index_pk), index_balance=a.index_balance, events=bool(a.events), old_clocks=a.old_clocks); params = sum(p.numel() for p in model.parameters())
    order = np.random.default_rng(a.seed).permutation(n_lanes); total = math.ceil(n_lanes/a.batch); quarter = math.ceil(total/4)
    if a.carry:                                                                                    # persistent memory: batch row b walks its own contiguous stream of lanes (one pass, every lane once)
        per = n_lanes//a.batch; total = per; quarter = math.ceil(total/4)
        order = (np.arange(a.batch)[None, :]*per + np.arange(per)[:, None]).reshape(-1)        # step s uses lanes b*per + s
    opt = torch.optim.AdamW(param_groups(model, a.wd, bool(a.wd_groups)), lr=a.lr, weight_decay=a.wd); sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=total, pct_start=0.05)
    out_dir = ROOT/'experiments/results/token_language'; out = out_dir/f'{a.tag}.json'; state_path = out_dir/f'{a.tag}.resume.pt'; assert a.tag and not out.exists()
    ident = {k: v for k, v in vars(a).items() if k not in EXECUTION_ONLY}; step, losses, milestones, wall, resumes = 0, [], [], 0.0, 0; carry = None
    clipper = LocalClip(model) if a.clip == 'local' else None
    assert not a.carry_credit or (a.carry and a.micro == 1), '--carry-credit needs --carry and --micro 1'
    cc = CarryCredit(model, a.carry_credit) if a.carry_credit else None
    if state_path.exists():
        s = torch.load(state_path, weights_only=False); assert same_run(s['args'], ident)
        model.load_state_dict(s['model']); opt.load_state_dict(s['opt']); sched.load_state_dict(s['sched']); torch.set_rng_state(s['torch_rng'])
        step, losses, milestones, wall, resumes = s['step'], s['losses'], s['milestones'], s['wall'], s['resumes'] + 1; carry = s.get('carry')
        if clipper is not None: clipper.load(s['clip'])
    began = time.time() - wall
    def save():
        tmp = state_path.with_suffix('.tmp'); torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(), torch_rng=torch.get_rng_state(), step=step, losses=losses,
                   milestones=milestones, wall=time.time() - began, resumes=resumes, args=ident, carry=carry, **({'clip': clipper.state()} if clipper is not None else {})), tmp); os.replace(tmp, state_path)
    sel_l = np.asarray(val[10485760:10485760 + 8*2048]).reshape(8, 2048)
    grow = [float(f) for f in a.grow.split(',')] if a.grow else []
    assert not grow or (len(grow) == a.levels - 1 and grow == sorted(grow) and 0 <= grow[0] and grow[-1] < 1), '--grow needs one nondecreasing fraction in [0, 1) per correction level'
    n_active_at = lambda s_: sum(s_ >= f*total for f in grow) if grow else a.levels - 1
    while step < total:
        model.n_active = n_active_at(step)
        ids = order[step*a.batch:(step + 1)*a.batch]; model.train(); w = torch.from_numpy(train[ids if a.carry else np.sort(ids)]); x, y = w[:, :-1], w[:, 1:]
        if cc is not None:                                                                         # cross-lane credit: the update applied now is the lane leaving the credit window
            loss, carry, done = cc.step(x, y, carry, a.carry)
            if done is not None: opt.zero_grad(); CarryCredit.apply(model, done); (clipper(model) if clipper is not None else nn.utils.clip_grad_norm_(model.parameters(), 1.0)); opt.step(); sched.step()
        else:
            opt.zero_grad(); loss, carry = train_step(model, x, y, carry, a.carry, a.micro)
            (clipper(model) if clipper is not None else nn.utils.clip_grad_norm_(model.parameters(), 1.0)); opt.step(); sched.step()
        step += 1; losses.append(loss)
        if cc is not None and step == total:                                                       # the last H lanes have no later lanes: apply them with the credit they have
            for done in cc.flush(): opt.zero_grad(); CarryCredit.apply(model, done); (clipper(model) if clipper is not None else nn.utils.clip_grad_norm_(model.parameters(), 1.0)); opt.step(); sched.step()
        if step % a.log_every == 0:
            print(json.dumps(dict(step=step, of=total, prequential_last=float(np.mean(losses[-a.log_every:])), levels=model._mem[2], elapsed_s=round(time.time() - began))), flush=True)
        if step % quarter == 0 or step == total:
            milestones.append(dict(step=step, tokens=step*a.batch*a.lane, select=lanes_nll_usage(model, sel_l, a.row_chunk), **({'select_carried': lanes_nll_carry(model, sel_l, a.row_chunk, a.carry, adjacent=True)} if a.carry else {}), prequential_segment=float(np.mean(losses[-quarter:])), elapsed_s=round(time.time() - began)))
            print(json.dumps(milestones[-1]), flush=True)
        if step % a.checkpoint_every == 0 or step == total: save()
        if a.max_steps and step >= a.max_steps:
            print(json.dumps(dict(smoke_steps=step, s_per_step=(time.time() - began)/step, max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, parameters=params)), flush=True)
            print(json.dumps(dict(smoke_select_2lanes=lanes_nll(model, sel_l[:2], a.row_chunk, stats=True))), flush=True); return
    model.n_active = a.levels - 1                                                              # final work and scoring at full depth
    from torch.utils.flop_counter import FlopCounterMode
    pf_ = None
    if a.carry:                                                                                    # count a carried lane: the prefix slots are read, refined and pointed at
        model.train(); w0 = torch.from_numpy(train[:1]); model.lane_nll(w0[:, :-1], w0[:, 1:]); pf_ = model.carry_out(a.carry)
    with FlopCounterMode(display=False) as fc:
        model.train(); w = torch.from_numpy(train[1:2]); model.lane_nll(w[:, :-1], w[:, 1:], prefix=pf_).mean().backward()
    model.zero_grad(); fpt = fc.get_total_flops()/(a.lane - 1); seg = max(1, total//8); grow_rec = None
    if grow:                                                                                       # grown depth: training work counted at the depth in use at each step
        steps_k = {}
        for s_ in range(total): steps_k[n_active_at(s_)] = steps_k.get(n_active_at(s_), 0) + 1
        fpt_k = {}
        for k_ in steps_k:
            model.n_active = k_
            with FlopCounterMode(display=False) as fck:
                model.train(); model.lane_nll(w[:, :-1], w[:, 1:], prefix=pf_).mean().backward()
            model.zero_grad(); fpt_k[k_] = fck.get_total_flops()/(a.lane - 1)
        model.n_active = a.levels - 1; fpt = sum(fpt_k[k_]*n_ for k_, n_ in steps_k.items())/total
        grow_rec = dict(fractions=grow, steps_at_depth={str(k_ + 1): n_ for k_, n_ in sorted(steps_k.items())}, train_flops_per_target_at_depth={str(k_ + 1): v for k_, v in sorted(fpt_k.items())})
    fl0 = np.asarray(val[25165824:25165824 + 1025]).astype(np.int64)[None]; xf, yf = torch.from_numpy(fl0[:, :-1]), torch.from_numpy(fl0[:, 1:]); model.eval()
    with torch.no_grad(), FlopCounterMode(display=False) as fci: model.lane_nll(xf, yf)              # measured inference work of the whole member (level 1, every level, readout) per target at history up to 1,023
    inference = dict(flops_per_target_window_1024=fci.get_total_flops()/1024, scope='measured, eval mode, one flagship window, reset lane (dense reference execution)')
    fire_rates = [st_.get('fire_rate') for st_ in model._mem[2]] if any(l_.fire for l_ in model.levels) else None   # measured on that flagship window
    active = member_work(model, 1023, fire_rates, a.carry)
    result = dict(status='completed', battle='R1', stage=STAGE, tag=a.tag, args=vars(a), parameters=params, steps=step,
                  presentations=step*a.batch*(a.lane - 1), train_wall_s=time.time() - began, resumes=resumes, milestones=milestones, train_flops_per_target=fpt, train_flops=fpt*step*a.batch*(a.lane - 1),
                  prequential_by_eighth=[float(np.mean(losses[i:i + seg])) for i in range(0, len(losses), seg)], prequential_last_quarter=float(np.mean(losses[-quarter:])),
                  **fineweb_protocols(model, val, a.row_chunk), **(carried_protocols(model, val, a.row_chunk, a.carry) if a.carry else {}), **(online_protocols(model, val, a.carry, opt) if a.carry else {}), work_level1_per_target=work_per_target(model.base, 255), inference_work=inference, **({'local_clip': dict(caps=clipper.caps, calibration_updates=clipper.calib, quantile=clipper.q, budget=clipper.budget)} if clipper is not None else {}), active_work=active, **({'grow': grow_rec} if grow_rec else {}),
                  reference=dict(cerl_4M='curie_r1_cerl_4M_s0_20261016_v1 (flagship 5.4665)', diagnostic_planB_L8_4M_flagship=5.8414), max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, source_sha256=src)
    torch.save(model.state_dict(), out_dir/f'{a.tag}.pt'); result['checkpoint'] = f'experiments/results/token_language/{a.tag}.pt'
    out.write_text(json.dumps(result, indent=1) + '\n'); state_path.unlink()
    print('RESULT', json.dumps({k: result[k] for k in ('select', 'score_65528', 'flagship', 'train_flops_per_target', 'parameters')}), flush=True)


if __name__ == '__main__':
    main()
