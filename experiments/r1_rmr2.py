#!/usr/bin/env python3
"""R1 residual-memory race model, version 2 (RMR-2; theory 172 sections 9-10): the asynchronous side of the memory stack.

Adds to RMR-1 (r1_rmr.py), both orthogonal to the memory-decomposition dimension:
- FIRING CLOCKS (events in, events out). Level k fires at step t with probability phi_k(t) from its event inputs: the
  current slot under M^(<k), the size of the correction level k-1 just wrote and level k-1's own firing. Training writes the
  expected correction phi * delta (exact credit to the clock); an activity cost lambda * mean(phi) makes firing sparse;
  evaluation can fire hard (phi > 0.5) so measured work counts actual firings. Level k's events out (phi, its correction
  size) are level k+1's events in, so cadence composes level by level.
- STANDING SLOTS (pooling). A shared pool of persistent slots with learned keys and per-slot decay. A firing level pools a
  bounded correction into the top-m standing slots its query selects (a race), and every later read races over event slots
  and standing slots together. Many events write into one standing slot: document-level concepts and entities, capacity
  beyond activity (dormant slots cost storage only).
Corrections stay zero-initialised in effect (ReZero gates at 0): at initialisation RMR-2 equals CERL exactly.
"""
import argparse, math, sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
ROOT = Path(__file__).resolve().parents[1]; sys.path[:0] = [str(ROOT/'experiments'), str(ROOT/'experiments/tpp')]
import r1_rmr as R  # noqa: E402
from r1_cerl import word_classes  # noqa: E402


class Level2(R.CorrectionLevel):
    def __init__(self, d, dk, top_m, read_topk, standing_m):
        super().__init__(d, dk, top_m, read_topk, rezero=True); self.standing_m = standing_m
        self.g = nn.Sequential(nn.Linear(2*d, 2*d), nn.GELU(), nn.Linear(2*d, 3*d))                 # delta_new, delta_old, delta_standing
        self.alpha_s = nn.Parameter(torch.tensor(0.0)); self.fire = nn.Linear(d + 2, 1); nn.init.zeros_(self.fire.weight); nn.init.constant_(self.fire.bias, 2.0)


class RMR2(R.RMR):
    def __init__(self, V, cls, words, cerl, levels=4, dk=64, top_m=4, chunk=64, read_topk=0, standing=256, standing_m=2, act_cost=0.0, hard_fire_eval=False):
        nn.Module.__init__(self); from r1_cerl import CERL
        self.base = CERL(V, cls, words, **cerl); d = cerl['d']; self.d, self.chunk, self.act_cost, self.hard_fire_eval = d, chunk, act_cost, hard_fire_eval
        self.levels = nn.ModuleList(Level2(d, dk, top_m, read_topk, standing_m) for _ in range(levels - 1))
        self.st_key = nn.Parameter(torch.randn(standing, dk)/math.sqrt(dk)); self.st_decay = nn.Parameter(torch.full((standing,), -3.0))   # rate = softplus(-3) ~ 0.05 per step
        self.aux = None; self._px = None

    def standing_coeff(self, a_st, ws_list, rows, T, own_from=None):
        """Read of standing slots: sum_u a[t,u] sum_tau gamma_u^(t-tau) ws[tau,u] ds_tau. The decay factorises,
        e^{-k_u (t - tau)} = e^{-k_u (t - c0)} e^{-k_u (c0 - tau)}, so the coefficients over tau are one matrix product
        [B, R, U] @ [B, U, T]; causality masks depend on (t, tau) only and are applied after it. Rates are capped at 0.5 per
        step so the chunk-relative factors stay finite."""
        rate = F.softplus(self.st_decay).clamp(max=0.5); c0 = int(rows[0]); out = 0.0; tt = torch.arange(T, device=rows.device)
        left = a_st*torch.exp(-rate[None, None, :]*(rows - c0).to(a_st.dtype)[None, :, None])             # [B, R, U]
        for ds, ws, lim in ws_list:
            right = ws*torch.exp(-rate[None, None, :]*(c0 - tt).to(ws.dtype)[None, :, None])               # [B, T, U]
            ok = (tt[None, :] <= rows[:, None]) if lim == 'le' else (tt[None, :] < lim).expand(len(rows), T)
            coef = (left @ right.transpose(1, 2))*ok.to(left.dtype)
            out = out + coef @ ds
        return out

    def build_memory(self, x):
        A = self.base.states(x); B, T, d = A.shape; dv = A.device; new_total = A; writes = []; st_writes = []; stats = []; act_terms = []
        prev_corr = torch.zeros(B, T, device=dv); prev_fire = torch.ones(B, T, device=dv)              # level 1 fires on every token
        for lev in self.levels:
            own_new = torch.zeros_like(A); own_dold = torch.zeros_like(A); own_W = torch.zeros(B, T, T, device=dv)
            own_ds = torch.zeros_like(A); own_ws = torch.zeros(B, T, self.st_key.shape[0], device=dv); own_phi = torch.zeros(B, T, device=dv)
            for c0 in range(0, T, self.chunk):
                c1 = min(c0 + self.chunk, T); rows = torch.arange(c0, c1, device=dv); prev = (torch.arange(T, device=dv) < c0).to(A.dtype)
                base_prime = new_total + own_new*prev[None, :, None]; cur = new_total[:, c0:c1]
                wr = writes + [(own_dold*prev[None, :, None], own_W)]
                u = lev.nq(cur); q = lev.q(u)
                S = (q @ lev.k(base_prime).transpose(1, 2) + self.corr_scores(q, wr, rows, lev.k))/math.sqrt(lev.dk)
                dist = (rows[:, None] - torch.arange(T, device=dv)[None, :]).to(A.dtype)
                S = (S - F.softplus(lev.log_rate)*torch.log1p(dist.clamp_min(0))).masked_fill(dist < 0, float('-inf'))
                if lev.read_topk and lev.read_topk < T:
                    kth = S.topk(min(lev.read_topk, T), -1).values[..., -1:]; S = S.masked_fill(S < kth, float('-inf'))
                S_st = (q @ self.st_key.T)/math.sqrt(lev.dk)                                          # standing slots race in the same read
                a_all = torch.softmax(torch.cat([S, S_st], -1), -1); a, a_st = a_all[..., :T], a_all[..., T:]
                st_list = [(ds_, ws_, 'le') for ds_, ws_ in st_writes] + [(own_ds, own_ws, c0)]
                r = a @ base_prime + self.corr_values(a, wr, rows) + self.standing_coeff(a_st, st_list, rows, T)
                dnew, dold, dst = lev.g(torch.cat([u, lev.nr(r)], -1)).chunk(3, -1)
                phi = torch.sigmoid(lev.fire(torch.cat([u, prev_corr[:, c0:c1, None], prev_fire[:, c0:c1, None]], -1))).squeeze(-1)   # firing clock from events in
                if self.hard_fire_eval and not self.training: phi = (phi > 0.5).to(phi.dtype)
                dnew = phi[..., None]*lev.alpha_new*dnew; dold = phi[..., None]*lev.alpha_old*torch.tanh(dold); dst = phi[..., None]*lev.alpha_s*torch.tanh(dst)
                Wc = torch.zeros(B, c1 - c0, T, device=dv)
                if lev.top_m:
                    ao = a.masked_fill(dist <= 0, 0.0); top = ao.topk(min(lev.top_m, T), -1); Wc = Wc.scatter(-1, top.indices, top.values)
                tops = S_st.topk(lev.standing_m, -1); wsc = torch.zeros(B, c1 - c0, self.st_key.shape[0], device=dv).scatter(-1, tops.indices, torch.softmax(tops.values, -1))
                own_new = own_new.index_copy(1, rows, dnew); own_dold = own_dold.index_copy(1, rows, dold); own_W = own_W.index_copy(1, rows, Wc)
                own_ds = own_ds.index_copy(1, rows, dst); own_ws = own_ws.index_copy(1, rows, wsc); own_phi = own_phi.index_copy(1, rows, phi)
            new_total = new_total + own_new; writes = writes + [(own_dold, own_W)]; st_writes = st_writes + [(own_ds, own_ws)]
            prev_corr = own_new.norm(dim=-1); prev_fire = own_phi; act_terms.append(own_phi.mean())
            stats.append(dict(fire_rate=float(own_phi.detach().mean()), new_norm=float(own_new.detach().norm(dim=-1).mean()), standing_norm=float(own_ds.detach().norm(dim=-1).mean())))
        self._act = torch.stack(act_terms).mean() if act_terms else None
        return new_total, writes, stats

    def lane_nll(self, x, y, rows=None, stats=False, msg=None):
        out = super().lane_nll(x, y, rows, stats, msg)
        if self.training and self.act_cost and self._act is not None:                                # activity cost: sparse firing (temporal computation pays only when useful)
            self.aux = (self.aux if self.aux is not None else 0.0) + self.act_cost*self._act
        return out


def contracts():
    torch.manual_seed(0); V = 61; counts = np.random.default_rng(0).integers(0, 50, V).astype(float); cls, words = word_classes(counts, 6, s_max=16)
    cerl = dict(d=32, modes=8, layers=1, dk=16, units=32, topk=3, explore=0); res = {}
    m = RMR2(V, cls, words, cerl, levels=3, dk=16, top_m=3, chunk=8, standing=16, standing_m=2); m.eval()
    with torch.no_grad():
        x = torch.randint(0, 8, (1, 40)); y = torch.cat([x[:, 1:], torch.randint(0, 8, (1, 1))], 1)
        res['zero_gates_equal_cerl'] = float((m.lane_nll(x.clone(), y) - m.base.lane_nll(x, y)).abs().max())
        for p_ in m.parameters(): p_.add_(torch.randn_like(p_)*0.1)
        full = m.lane_nll(x.clone(), y); tot = sum(math.exp(-m.lane_nll(x.clone(), torch.cat([y[:, :25], torch.tensor([[v]]), y[:, 26:]], 1))[0, 25].item()) for v in range(V))
        res['normalisation_sum'] = tot
        x2, y2 = x.clone(), y.clone(); p = 30; x2[0, p] = (x2[0, p] + 1) % V; y2[0, p - 1] = x2[0, p]
        res['causal_prefix_max_diff'] = float((m.lane_nll(x2, y2)[0, :p - 1] - full[0, :p - 1]).abs().max())
        xc = x.clone(); res['row_chunk_max_diff'] = float((torch.cat([m.lane_nll(xc, y, (r, min(r + 7, 40))) for r in range(0, 40, 7)], 1) - full).abs().max())
        m.hard_fire_eval = True; hard = m.lane_nll(x.clone(), y); res['hard_fire_differs_finite'] = bool(torch.isfinite(hard).all())
        x3, y3 = x.clone(), y.clone(); x3[0, p] = (x3[0, p] + 1) % V; y3[0, p - 1] = x3[0, p]
        res['hard_fire_causal_prefix'] = float((m.lane_nll(x3, y3)[0, :p - 1] - hard[0, :p - 1]).abs().max())
    ok = res['zero_gates_equal_cerl'] < 1e-5 and abs(res['normalisation_sum'] - 1) < 1e-4 and res['causal_prefix_max_diff'] < 1e-5 and res['row_chunk_max_diff'] < 1e-4 and res['hard_fire_causal_prefix'] < 1e-5
    res['status'] = 'passed' if ok else 'failed'; return res


if __name__ == '__main__':
    import json
    if '--mode' in sys.argv and sys.argv[sys.argv.index('--mode') + 1] == 'contract':
        print(json.dumps(contracts())); sys.exit(0)
    # training: reuse r1_rmr.main with RMR2 (extra flags parsed here, the rest by r1_rmr)
    ap = argparse.ArgumentParser(add_help=False); ap.add_argument('--standing', type=int, default=256); ap.add_argument('--standing-m', type=int, default=2)
    ap.add_argument('--act-cost', type=float, default=0.0); ap.add_argument('--hard-fire-eval', type=int, default=0); extra, rest = ap.parse_known_args()
    orig = R.RMR
    class Bound(RMR2):
        def __init__(self, V, cls, words, cerl, levels=2, dk=64, top_m=4, chunk=64, read_topk=0, rezero=True, old_decay=0.0, old_per_target=False):
            assert old_decay == 0 and not old_per_target, 'RMR-2 implements neither old-slot decay nor per-target corrections'
            super().__init__(V, cls, words, cerl, levels, dk, top_m, chunk, read_topk, extra.standing, extra.standing_m, extra.act_cost, bool(extra.hard_fire_eval))
    R.RMR = Bound; R.STAGE = f'native residual-memory race model with firing clocks and standing slots (RMR-2, theory 172 sections 9-10; standing {extra.standing}, act_cost {extra.act_cost}), one pass'; sys.argv = [sys.argv[0]] + rest
    R.main()
