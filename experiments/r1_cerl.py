#!/usr/bin/env python3
"""R1 native language member: the causal event race (CERL), experiments/theory/169_causal_race_language.md.

A token is an event and every event has a cause. At each position t (issuing token t+1) a race of clocks among candidate
causes decides who emits the next token:
- episodic REPEAT causes: every past event j <= t proposes its own token x_j (entity and topic recurrence);
- episodic CONTINUE causes: every past event j < t proposes the token that followed it, x_{j+1} (induction / span
  continuation); its clock is excited by the length of the context match ending at j and at t (a copied span stays
  primed: the Hawkes self-excitation of the event battles);
- CONCEPT causes: a bank of dormant units; the race wakes the top-k units whose keys match the current state (plus random
  exploration units in training, the paid counterfactual alternatives); a woken unit proposes from the two-level word law
  shifted by its own vector (capacity beyond activity: parameters grow with the bank, work with k);
- the BACKGROUND route: a two-level word race (frequency-mass word class, then a word inside the class; classes capped at
  512 words) from the state, giving every token support at ~(C + 512) rows of work instead of V.
Race win probabilities pi_c = softmax(clock log-rates); each cause's proposal is an exactly normalised law, so
p(y) = sum_c pi_c p_c(y) is exactly normalised (competing risks, theory note 168). Episodic clocks age with the elapsed
events since j (log1p(age) penalty, learned per mechanism).

State: token events written into persistent decaying/rotating temporal memory (r1_deep_keyed_lm.ParallelTemporalMemory,
exact kernel, chunk 64) with small local maps; the state forms the race queries and the background/concept proposals. It
has no attention and no dense V-way softmax; predictions are delivered by the race.

Reported with every score: native share (loss reduction over the background route alone, and posterior responsibility by
cause type) and active work per target (candidates scored, units woken, class/word rows).

Corpora: fineweb (GPT-2 tokens; one pass over fresh 256-token lanes from fineweb_train_000001 offset 0, the plan-B
diagnostic's data; selection 8x2048 lanes @10,485,760; 65,528-target slice 8x8192 @20,971,520; flagship 8x1024 windows
@25,165,824), ptb / wt2 (word level, epochs, validation in windows; test sealed).
"""
import argparse, hashlib, json, math, os, resource, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
ROOT = Path(__file__).resolve().parents[1]; sys.path[:0] = [str(ROOT/'experiments'), str(ROOT/'experiments/tpp')]
from r1_deep_keyed_lm import ParallelTemporalMemory  # noqa: E402
from native_slot_memory import SlotMemory  # noqa: E402

S_MAX = 512
EXECUTION_ONLY = ('max_steps', 'checkpoint_every', 'mode', 'row_chunk', 'log_every', 'run', 'split', 'select_from', 'online_stride', 'grid_targets', 'row_chunk_unused')


def param_groups(model, wd, groups=False):
    """AdamW groups. groups=False: decay everything (the original runs). groups=True: no decay on scalars/vectors (biases,
    norms, learned clock rates and slopes, ReZero gates; decoupled decay pulls softplus-parameterised rates toward 0.69) and
    none on dormant-capacity tables (concept keys/vectors: decaying them erodes capacity that is meant to wait unused)."""
    if not groups: return [dict(params=list(model.parameters()), weight_decay=wd)]
    dormant = ('unit_key', 'unit_out', 'unit_val', 'key1', 'key2', 'wkey', 'rkey', 'wkey1', 'wkey2', 'rkey1', 'rkey2')
    dec, nod = [], []
    for n, p in model.named_parameters():
        (nod if p.dim() < 2 or n.split('.')[-1] in dormant else dec).append(p)
    return [dict(params=dec, weight_decay=wd), dict(params=nod, weight_decay=0.0)]


# flags added after a run started, at their default (= the run's actual behaviour); resumes stay exact
LATER_FLAG_DEFAULTS = dict(product_keys=0, wd_groups=0, unit_init=0.0, balance=0.0, old_per_target=0, write_race=0, write_explore=0, carry=0, level1='tm', slots=1024, slot_m=4, slot_explore=0, fire=0, act_cost=0.0, old_norm=0, read_explore=0, slot_read_topk=0, slot_pk=0, index_regions=0, index_read=0, readout_index=0, readout_read=0, readout_tok=32, index_pk=0, index_balance=0.0)


def same_run(saved, ident):
    """Resume identity: equal settings, where flags absent from the saved run must be at their (behaviour-preserving) defaults."""
    return {k: v for k, v in ident.items() if k in saved} == saved and all(k in LATER_FLAG_DEFAULTS and LATER_FLAG_DEFAULTS[k] == v for k, v in ident.items() if k not in saved)


def word_classes(counts, n_classes, s_max=S_MAX, trim=False):
    """Frequency-mass word classes: words by descending training count; a class closes at 1/n_classes of the mass or at
    s_max words. Returns class_of [V] and class_words [C, s_max] (-1 padded)."""
    V = len(counts); order = np.argsort(-counts, kind='stable'); p = (counts + 0.5)/(counts + 0.5).sum(); cls, cur, mass, out = np.zeros(V, np.int64), [], 0.0, []
    for w in order:
        cur.append(w); mass += p[w]
        if mass >= 1.0/n_classes or len(cur) == s_max: out.append(cur); cur, mass = [], 0.0
    if cur: out.append(cur)
    words = np.full((len(out), s_max), -1, np.int64)
    for c, ws in enumerate(out): cls[ws] = c; words[c, :len(ws)] = ws
    if trim: words = words[:, :max(len(ws) for ws in out)]                                       # drop all-padding columns (small alphabets: higher cause levels)
    return cls, words


def match_lengths(x, r0, r1, cap=8):
    """l[b, t, j] for rows t in [r0, r1) and columns j < r1: length of the common suffix of x[..t] and x[..j], capped. Only
    rows r0-cap+1 .. r1 are formed (uint8), so long evaluation lanes cost O(rows x r1) bytes, not O(r1^2) words."""
    a = max(0, r0 - cap + 1); eq = (x[:, a:r1, None] == x[:, None, :r1]).to(torch.uint8); l = eq.clone()
    for _ in range(cap - 1):
        sh = torch.zeros_like(l); sh[:, 1:, 1:] = l[:, :-1, :-1]; l = eq*(1 + sh)
    return l[:, r0 - a:].clamp_max(cap)


class CERL(nn.Module):
    def __init__(self, V, class_of, class_words, d=128, modes=64, layers=2, dk=64, units=1024, topk=4, explore=2, tm_chunk=64, age_slope=1.5, dropout=0.0, product_keys=False, cause_messages=False, unit_init=0.0, balance=0.0, level1='tm', slots=1024, slot_m=4, slot_explore=0, slot_read_topk=0, slot_pk=False):
        super().__init__(); self.V, self.d, self.dk, self.U, self.topk, self.explore, self.p_drop = V, d, dk, units, topk, explore, dropout
        self.register_buffer('class_of', torch.as_tensor(class_of)); self.register_buffer('class_words', torch.as_tensor(class_words)); C = class_words.shape[0]
        self.embed = nn.Embedding(V, d); nn.init.normal_(self.embed.weight, std=0.02); self.word_bias = nn.Parameter(torch.zeros(V))
        self.class_emb = nn.Parameter(torch.randn(C, d)*0.02); self.class_bias = nn.Parameter(torch.zeros(C))
        self.n1 = nn.ModuleList(nn.LayerNorm(d) for _ in range(layers)); self.level1 = level1
        if level1 == 'slots':                                                                        # native level 1: sparse addressed slots, learned clocks, race read (native_slot_memory.py)
            self.tm = nn.ModuleList(SlotMemory(d, slots, slot_m, dk, slot_explore, slot_read_topk, slot_pk) for _ in range(layers)); self.tm_out = nn.ModuleList(nn.Linear(d, d) for _ in range(layers))
        else:                                                                                        # dense complex-diagonal linear recurrence (labelled: not native)
            self.tm = nn.ModuleList(ParallelTemporalMemory(d, modes) for _ in range(layers)); self.tm_out = nn.ModuleList(nn.Linear(2*modes, d) for _ in range(layers))
        self.n2 = nn.ModuleList(nn.LayerNorm(d) for _ in range(layers))
        self.mlp = nn.ModuleList(nn.Sequential(nn.Linear(d, 2*d), nn.GELU(), nn.Linear(2*d, d)) for _ in range(layers)); self.out_norm = nn.LayerNorm(d)
        for t in self.tm: t.CHUNK = min(tm_chunk, 64) if level1 == 'slots' else tm_chunk
        # race clocks
        self.bg = nn.Linear(d, 1); nn.init.zeros_(self.bg.weight); nn.init.constant_(self.bg.bias, 2.0)
        self.q_unit = nn.Linear(d, dk); self.pk = product_keys
        if product_keys:                                                                             # two-stage race over half-keys: unit (a, b) scores q1.K1[a] + q2.K2[b]
            n = int(round(math.sqrt(units))); assert n*n == units, 'product keys need a square bank'; self.n_half = n
            self.key1, self.key2 = nn.Parameter(torch.randn(n, dk//2)/math.sqrt(dk)), nn.Parameter(torch.randn(n, dk//2)/math.sqrt(dk))
            self.bias1, self.bias2 = nn.Parameter(torch.zeros(n)), nn.Parameter(torch.zeros(n))
        else:
            self.unit_key = nn.Parameter(torch.randn(units, dk)/math.sqrt(dk)); self.unit_bias = nn.Parameter(torch.zeros(units))
        self.unit_out = nn.Parameter(torch.randn(units, d)*unit_init if unit_init > 0 else torch.zeros(units, d)); self.cause_msgs = cause_messages; self.balance = balance   # unit_init > 0 breaks the symmetry of identical concept laws
        if cause_messages:
            self.v_ep = nn.Linear(d, d); nn.init.zeros_(self.v_ep.weight); nn.init.zeros_(self.v_ep.bias); self.unit_val = nn.Parameter(torch.zeros(units, d))
        self.q_rep, self.k_rep, self.q_cont, self.k_cont = (nn.Linear(d, dk) for _ in range(4))
        self.b_rep, self.b_cont = nn.Parameter(torch.tensor(-2.0)), nn.Parameter(torch.tensor(-2.0))
        a0 = math.log(math.expm1(age_slope))   # power-law excitation kernel age^-slope: integrable for slope > 1, so total episodic hazard stays bounded as history grows
        self.age_rep, self.age_cont = nn.Parameter(torch.tensor(a0)), nn.Parameter(torch.tensor(a0)); self.span = nn.Parameter(torch.tensor(1.0))

    def states(self, x, msg=None, tm_state=None):
        """tm_state: per-layer temporal-memory states carried from the previous lane (None = fresh); final states in self.tm_state."""
        dr = lambda z: F.dropout(z, self.p_drop, self.training) if self.p_drop else z          # small-corpus regularisation (0 on FineWeb)
        h = dr(self.embed(x) + (msg if msg is not None else 0))
        for i, (n1, tm, to, n2, mlp) in enumerate(zip(self.n1, self.tm, self.tm_out, self.n2, self.mlp)):
            h = h + dr(to(tm(n1(h), None if tm_state is None else tm_state[i]))); h = h + dr(mlp(n2(h)))
        self.tm_state = [t.last_state for t in self.tm]; return dr(self.out_norm(h))

    def unit_score(self, q, idx):
        """Race log-rates of units idx [..., k] for queries q [..., dk]."""
        if not self.pk: return (q[..., None, :]*self.unit_key[idx]).sum(-1) + self.unit_bias[idx]
        h2 = self.dk//2; a, b = idx//self.n_half, idx % self.n_half
        return (q[..., None, :h2]*self.key1[a]).sum(-1) + self.bias1[a] + (q[..., None, h2:]*self.key2[b]).sum(-1) + self.bias2[b]

    def wake(self, q):
        """Exact top-k units: full scoring for a flat bank; for product keys the top-k of each half then the top-k of the k x k
        sums (the k best sums always lie in that grid), costing 2 sqrt(U) half-key scores instead of U."""
        if not self.pk:
            su = q @ self.unit_key.T + self.unit_bias; sel = su.topk(self.topk, -1).indices; return sel, su.gather(-1, sel)
        h2, k = self.dk//2, self.topk
        s1 = q[..., :h2] @ self.key1.T + self.bias1; s2 = q[..., h2:] @ self.key2.T + self.bias2
        v1, i1 = s1.topk(k, -1); v2, i2 = s2.topk(k, -1); grid = (v1[..., :, None] + v2[..., None, :]).flatten(-2)
        vs, g = grid.topk(k, -1); sel = i1.gather(-1, g//k)*self.n_half + i2.gather(-1, g % k); return sel, vs

    def two_level(self, H, y):
        """log p(y) under the two-level word law of each state row in H [N, K, d] for targets y [N]."""
        lc = torch.log_softmax(H @ self.class_emb.T + self.class_bias, -1)                              # [N, K, C]
        c = self.class_of[y]; words = self.class_words[c]; ok = words >= 0; wi = words.clamp_min(0)      # [N, S]
        lw = torch.einsum('nkd,nsd->nks', H, self.embed.weight[wi]) + self.word_bias[wi][:, None, :]
        lw = torch.log_softmax(lw.masked_fill(~ok[:, None, :], float('-inf')), -1)
        pos = (words == y[:, None]).float().argmax(-1)                                                 # target's slot in its class
        return lc.gather(-1, c[:, None, None].expand(-1, H.shape[1], 1)).squeeze(-1) + lw.gather(-1, pos[:, None, None].expand(-1, H.shape[1], 1)).squeeze(-1)

    def race(self, x, y, h, r0, r1, explore, td=None):
        """The race at query rows [r0, r1) given states h: per-target NLL, the competing-risk parts and the woken set.
        td (optional, from a higher cause level; r1_cerl_h.py): extra top-down-woken units 'sel_extra' [B, R, k] and additive
        log-rate bonuses 'unit_bonus' (callable on the woken ids), 'b_bg', 'b_rep', 'b_cont' [B, R, 1]. Bonuses enter the race
        normaliser and the parts alike, so the law stays exactly normalised."""
        B, L = x.shape; R = r1 - r0; hq = h[:, r0:r1]
        dv = x.device; t = torch.arange(r0, r1, device=dv)[:, None]; j = torch.arange(r1, device=dv)[None, :]; age = (t - j).clamp_min(0).float()
        ml = match_lengths(x, r0, r1)                                                                # [B, R, r1]
        S_rep = (self.q_rep(hq) @ self.k_rep(h[:, :r1]).transpose(1, 2))/math.sqrt(self.dk) + self.b_rep - F.softplus(self.age_rep)*torch.log1p(age)
        S_rep = S_rep.masked_fill(~(j <= t), float('-inf'))
        S_cont = (self.q_cont(hq) @ self.k_cont(h[:, :r1]).transpose(1, 2))/math.sqrt(self.dk) + self.b_cont - F.softplus(self.age_cont)*torch.log1p(age) + self.span*torch.log1p(ml.float())
        S_cont = S_cont.masked_fill(~(j < t), float('-inf'))
        sel, s_sel = self.wake(self.q_unit(hq))                                                      # woken units: the predictive race (train = eval)
        s_bg = self.bg(hq)                                                                           # [B, R, 1]
        if td is not None and 'unit_bonus' in td:
            if td.get('sel_extra') is not None:                                                      # top-down waking: units the level above expects
                ex_ = td['sel_extra']; s_ex_ = self.unit_score(self.q_unit(hq), ex_).masked_fill((ex_[..., :, None] == sel[..., None, :]).any(-1), float('-inf'))
                sel, s_sel = torch.cat([sel, ex_], -1), torch.cat([s_sel, s_ex_], -1)
            s_sel = s_sel + td['unit_bonus'](sel); s_bg = s_bg + td['b_bg']; S_rep = S_rep + td['b_rep']; S_cont = S_cont + td['b_cont']
        if td is not None and td.get('S_rep_add') is not None:                                      # residual-memory refinements of old event slots (r1_rmr.py)
            S_rep = S_rep + td['S_rep_add']; S_cont = S_cont + td['S_cont_add']
        logZ = torch.logsumexp(torch.cat([s_bg, s_sel, S_rep, S_cont], -1), -1)
        yq = y[:, r0:r1]; H = torch.cat([hq[:, :, None], hq[:, :, None] + self.unit_out[sel]], 2)      # [B, R, 1+K, d]
        if explore:
            # Paid counterfactual alternatives: random dormant units ask "would including me have raised p(y)?" against the
            # actual race held fixed (detached). Credit reaches only their keys, biases, vectors and the unit query map; they
            # never enter the predictive mixture, so training and evaluation predict with the same race.
            ex = torch.randint(0, self.U, (B, R, self.explore), device=dv); hd = hq.detach()
            s_ex = self.unit_score(self.q_unit(hd), ex)
            s_ex = s_ex.masked_fill((ex[..., :, None] == sel[..., None, :]).any(-1), float('-inf'))
            H = torch.cat([H, hd[:, :, None] + self.unit_out[ex]], 2)                                  # one word-law pass for all causes
        Hf, yf = H.reshape(B*R, -1, self.d), yq.reshape(-1); n_ = Hf.shape[0]            # word laws in row chunks under checkpointing: [rows, class words, d] never all at once
        lp = torch.cat([torch.utils.checkpoint.checkpoint(self.two_level, Hf[i:i + 256], yf[i:i + 256], use_reentrant=False) if torch.is_grad_enabled() else self.two_level(Hf[i:i + 256], yf[i:i + 256])
                        for i in range(0, n_, 256)]).view(B, R, -1)
        K = s_sel.shape[-1]; lp_ex = lp[..., 1 + K:]; lp = lp[..., :1 + K]
        xj = x[:, :r1]; nxt = torch.cat([x[:, 1:r1], yq[:, -1:] if r1 == L else x[:, r1:r1 + 1]], 1)  # x_{j+1}; the last j has no successor among inputs (masked by j < t)
        hit_rep = xj[:, None, :] == yq[:, :, None]; hit_cont = nxt[:, None, :] == yq[:, :, None]
        neg = torch.full_like(S_rep, float('-inf'))
        parts = torch.cat([s_bg + lp[..., :1], s_sel + lp[..., 1:], torch.where(hit_rep, S_rep, neg), torch.where(hit_cont, S_cont, neg)], -1)
        nll = logZ - torch.logsumexp(parts, -1)
        self.aux = None
        if self.training and self.balance > 0:                                                      # load balance (Switch-style): fraction woken x mean wake probability, so dormant units get used
            s0 = sel[..., :self.topk].reshape(-1); one = torch.ones(s0.numel(), device=dv); q_ = self.q_unit(hq)
            if not self.pk:
                P = torch.softmax(q_ @ self.unit_key.T + self.unit_bias, -1).reshape(-1, self.U).mean(0)
                f = torch.zeros(self.U, device=dv).scatter_add(0, s0, one)/s0.numel(); bal = self.U*(f*P).sum()
            else:                                                                                   # product keys: balance each half-bank (2 sqrt(U) scores, never U)
                h2, n = self.dk//2, self.n_half; bal = 0.0
                for qh, key, bias, half in ((q_[..., :h2], self.key1, self.bias1, s0//n), (q_[..., h2:], self.key2, self.bias2, s0 % n)):
                    P = torch.softmax(qh @ key.T + bias, -1).reshape(-1, n).mean(0); f = torch.zeros(n, device=dv).scatter_add(0, half, one)/s0.numel(); bal = bal + 0.5*n*(f*P).sum()
            self.aux = (self.balance*bal)*torch.ones_like(nll)
        if explore:
            race = torch.cat([s_bg, s_sel, S_rep, S_cont], -1).detach()
            ex_aux = torch.logsumexp(torch.cat([race, s_ex], -1), -1) - torch.logsumexp(torch.cat([parts.detach(), s_ex + lp_ex], -1), -1)
            self.aux = ex_aux if self.aux is None else self.aux + ex_aux
        return nll, parts, s_sel, sel, lp, s_bg, S_rep, S_cont

    def cand_logp(self, x, h, r0, r1, cands):
        """Sparse evaluation of the race law at rows [r0, r1) for candidate symbols cands [B, R, M]: log p(c) without forming the
        distribution over the alphabet (used by a higher cause level to bias the level below; r1_cerl_h.py)."""
        B, L = x.shape; R = r1 - r0; hq = h[:, r0:r1]; dv = x.device
        t = torch.arange(r0, r1, device=dv)[:, None]; j = torch.arange(r1, device=dv)[None, :]; age = (t - j).clamp_min(0).float(); ml = match_lengths(x, r0, r1)
        S_rep = ((self.q_rep(hq) @ self.k_rep(h[:, :r1]).transpose(1, 2))/math.sqrt(self.dk) + self.b_rep - F.softplus(self.age_rep)*torch.log1p(age)).masked_fill(~(j <= t), float('-inf'))
        S_cont = ((self.q_cont(hq) @ self.k_cont(h[:, :r1]).transpose(1, 2))/math.sqrt(self.dk) + self.b_cont - F.softplus(self.age_cont)*torch.log1p(age)
                  + self.span*torch.log1p(ml.float())).masked_fill(~(j < t), float('-inf'))
        sel, s_sel = self.wake(self.q_unit(hq)); s_bg = self.bg(hq); logZ = torch.logsumexp(torch.cat([s_bg, s_sel, S_rep, S_cont], -1), -1)
        H = torch.cat([hq[:, :, None], hq[:, :, None] + self.unit_out[sel]], 2).reshape(B*R, -1, self.d); M = cands.shape[-1]
        lp = torch.stack([self.two_level(H, cands[..., m].reshape(-1)).view(B, R, -1) for m in range(M)], 2)        # [B, R, M, 1+K]
        xj = x[:, :r1]; nxt = torch.cat([x[:, 1:r1], x[:, -1:]], 1)                                      # last j masked (j < t)
        hr = xj[:, None, None, :] == cands[..., None]; hc = nxt[:, None, None, :] == cands[..., None]; neg = float('-inf')
        parts = torch.cat([s_bg[:, :, None] + lp[..., :1], s_sel[:, :, None] + lp[..., 1:], torch.where(hr, S_rep[:, :, None], neg), torch.where(hc, S_cont[:, :, None], neg)], -1)
        return torch.logsumexp(parts, -1) - logZ[..., None]

    def messages(self, x, y, chunk=512):
        """Cause messages (theory 169 section 3): once token t+1 is observed, the posterior attribution of its cause sends that
        cause's content into memory at position t+1: m_t = sum_j post_j v(h_j) over episodic causes (repeat and continue both
        attribute to event j) + sum_u post_u val_u over woken concepts. Pass-1 states form the attribution (weights detached:
        an observation of who caused the event); values carry gradient. Causal: m_t depends on tokens <= t+1 and enters the
        state at input position t+1, which predicts t+2."""
        B, L = x.shape; h1 = self.states(x); out = []
        for r0 in range(0, L, chunk):
            r1 = min(r0 + chunk, L)
            with torch.no_grad():
                _, parts, s_sel, sel, *_ = self.race(x, y, h1.detach(), r0, r1, False); post = torch.softmax(parts, -1); K = s_sel.shape[-1]
            w_ep = post[..., 1 + K:1 + K + r1] + post[..., 1 + K + r1:]
            out.append(w_ep @ self.v_ep(h1[:, :r1]) + (post[..., 1:1 + K, None]*self.unit_val[sel]).sum(-2))
        m = torch.cat(out, 1); return torch.cat([torch.zeros_like(m[:, :1]), m[:, :-1]], 1)

    def lane_nll(self, x, y, rows=None, stats=False, msg=None):
        """Per-target NLL [B, R] for query rows `rows` of lanes x [B, L] (inputs) with targets y [B, L]."""
        B, L = x.shape
        if self.cause_msgs and msg is None: msg = self.messages(x, y)
        h = self.states(x, msg); r0, r1 = (0, L) if rows is None else rows
        nll, parts, s_sel, sel, lp, s_bg, S_rep, S_cont = self.race(x, y, h, r0, r1, self.training and self.explore > 0)
        if getattr(self, 'record_winner', False):                                                    # analysis: posterior-winning cause type per target (0 bg, 1 concept, 2 repeat, 3 continue)
            with torch.no_grad():
                post = torch.softmax(parts, -1); K_ = s_sel.shape[-1]
                grp = torch.stack([post[..., 0], post[..., 1:1 + K_].sum(-1), post[..., 1 + K_:1 + K_ + r1].sum(-1), post[..., 1 + K_ + r1:].sum(-1)], -1); self.last_winner = grp.argmax(-1)
        if not stats: return nll
        with torch.no_grad():
            post = torch.softmax(parts, -1); K = s_sel.shape[-1]
            st = dict(bg_only_nll=(-lp[..., 0]).mean().item(), resp_bg=post[..., 0].mean().item(), resp_concept=post[..., 1:1 + K].sum(-1).mean().item(),
                      resp_repeat=post[..., 1 + K:1 + K + r1].sum(-1).mean().item(), resp_continue=post[..., 1 + K + r1:].sum(-1).mean().item(),
                      win_bg=torch.softmax(torch.cat([s_bg, s_sel, S_rep, S_cont], -1), -1)[..., 0].mean().item())
        return nll, st


def work_per_target(m, history):
    """Active multiply-adds per target for likelihood scoring (dense state maps, race scores over `history` candidates,
    all U unit keys, the background and k concept two-level laws at mean class size)."""
    d, dk, C = m.d, m.dk, m.class_words.shape[0]; s_mean = float((m.class_words >= 0).sum())/C
    state = sum(p.numel() for n, p in m.named_parameters() if n.startswith(('tm', 'tm_out', 'mlp')))   # dense reference execution (slots: every slot key scored, every slot read)
    slots = dict(level1='slots', slots_per_layer=m.tm[0].S, slots_written_per_event=m.tm[0].m) if getattr(m, 'level1', 'tm') == 'slots' else dict(level1='temporal-memory recurrence')
    return dict(state_macs=state, **slots, episodic_candidates=2*history, episodic_macs=4*d*dk + 2*history*dk, unit_scores=(2*m.n_half if m.pk else m.U), unit_macs=d*dk + (2*m.n_half*dk//2 + m.topk**2 if m.pk else m.U*dk), concepts_woken=m.topk,
                proposal_rows=(1 + m.topk)*(C + s_mean), proposal_macs=(1 + m.topk)*(C + s_mean)*d)


# ---------------------------------------------------------------- data and protocols
def load_fineweb(path):
    h = np.fromfile(path, dtype=np.int32, count=256); assert h[0] == 20240520
    return np.memmap(path, dtype=np.uint16, mode='r', offset=1024)


def lanes_nll(model, block, row_chunk, stats=False):
    """Mean NLL over lanes [N, L+1] (each target conditioned on its lane's history), query rows in chunks."""
    model.eval(); tot = n = 0.0; acc = {}
    with torch.no_grad():
        for i in range(len(block)):
            w = torch.from_numpy(block[i:i + 1].astype(np.int64)); x, y = w[:, :-1], w[:, 1:]; L = x.shape[1]
            msg = model.messages(x, y) if model.cause_msgs else None                                  # cause messages once per lane, shared by its row chunks
            for r0 in range(0, L, row_chunk):
                out = model.lane_nll(x, y, (r0, min(r0 + row_chunk, L)), stats=stats, msg=msg)
                nll, st = out if stats else (out, None); tot += nll.sum().item(); n += nll.numel()
                if st:
                    for k, v in st.items(): acc[k] = acc.get(k, 0.0) + v*nll.numel()
    model.train(); res = dict(nll=tot/n, targets=int(n))
    if acc: res['native'] = {k: v/n for k, v in acc.items()}; res['native']['native_gain'] = res['native']['bg_only_nll'] - res['nll']
    return res


def fineweb_protocols(model, val, row_chunk, stats=True):
    sel = np.asarray(val[10485760:10485760 + 8*2048]).reshape(8, 2048); sc = np.asarray(val[20971520:20971520 + 8*8192]).reshape(8, 8192)
    fl = np.stack([np.asarray(val[p:p + 1025]) for p in range(25165824, 25165824 + 8192, 1024)])
    return dict(select=lanes_nll(model, sel, row_chunk, stats), score_65528=lanes_nll(model, sc, row_chunk, stats), flagship=lanes_nll(model, fl, row_chunk, stats))


# ---------------------------------------------------------------- word-level corpora (PTB, WikiText-2): experiments/PTB_WT2_BATTLE.md
def window_segments(model, stream, ctx, stride, start):
    """Per-target NLL tensors for consecutive segments of `stride` targets covering every token (the first conditioned on a
    start <eos>); each segment's window holds up to ctx tokens of history, which the episodic causes race over."""
    x = np.concatenate([[start], stream]); n = len(stream)
    for t0 in range(0, n, stride):
        t1 = min(t0 + stride, n); w0 = max(0, t1 - ctx); inp = torch.from_numpy(x[w0:t1][None]); tgt = torch.from_numpy(x[w0 + 1:t1 + 1][None])
        yield model.lane_nll(inp, tgt, (t0 - w0, t1 - w0))[0]


def frozen_words(model, stream, ctx, stride, start):
    model.eval()
    with torch.no_grad(): out = torch.cat(list(window_segments(model, stream, ctx, stride, start))).numpy()
    model.train(); return out


def online_words(model, stream, ctx, stride, start, ms, lr, decay):
    """Causal online adaptation (dynamic evaluation): score a segment, then one RMS-normalised step on it with decay toward
    the trained weights; every prediction uses weights adapted only on earlier segments."""
    model.eval(); ps = [p for p in model.parameters() if p.requires_grad]; th0 = [p.detach().clone() for p in ps]; out = []
    rms = []
    for m in ms:
        r = m.sqrt(); nz = r[r > 0]; rms.append(r.clamp_min(0.5*nz.mean().item() if nz.numel() else 1e-3) + 1e-8)
    mr = torch.cat([r.flatten() for r in rms]).mean(); dec = [(decay*r/mr).clamp(max=1.0) for r in rms]
    for nll in window_segments(model, stream, ctx, stride, start):
        out.append(nll.detach().numpy()); model.zero_grad(); nll.mean().backward()
        with torch.no_grad():
            for p, r, d_, t0 in zip(ps, rms, dec, th0):
                if p.grad is not None: p.add_(-lr*p.grad/r + d_*(t0 - p))
    with torch.no_grad():
        for p, t0 in zip(ps, th0): p.copy_(t0)
    model.train(); return np.concatenate(out)


def training_ms(model, train, lane, batches, seed):
    model.eval(); ps = [p for p in model.parameters() if p.requires_grad]; ms = [torch.zeros_like(p) for p in ps]; rng = np.random.default_rng(seed)
    for _ in range(batches):
        i = rng.integers(0, len(train) - lane - 1); w = torch.from_numpy(train[i:i + lane + 1][None]); model.zero_grad(); model.lane_nll(w[:, :-1], w[:, 1:]).mean().backward()
        for m, p in zip(ms, ps):
            if p.grad is not None: m += p.grad**2/batches
    model.zero_grad(); model.train(); return ms


def words_main(a, src):
    from r1_ptb_native import read_corpus, CORPORA
    ids, vocab, eos = read_corpus(a.corpus); V = CORPORA[a.corpus]['vocab']; out_dir = ROOT/'experiments/results/word_language'; out_dir.mkdir(exist_ok=True)
    counts = np.bincount(ids['train'], minlength=V).astype(float); cls, words = word_classes(counts, a.classes)
    model = CERL(V, cls, words, a.d, a.modes, a.layers, a.dk, a.units, a.topk, a.explore, age_slope=a.age_slope, dropout=a.dropout, product_keys=bool(getattr(a, 'product_keys', 0)), cause_messages=bool(getattr(a, 'cause_messages', 0))); params = sum(p.numel() for p in model.parameters())
    if a.mode == 'score':
        rec = json.loads((out_dir/f'{a.run}.json').read_text()); model.load_state_dict(torch.load(ROOT/rec['checkpoint'])); path = out_dir/f'{a.tag}.json'; assert not path.exists()
        split = ids[a.split]; res = dict(battle='R1', tag=a.tag, run=a.run, corpus=a.corpus, split=a.split, targets=len(split), parameters=params, ctx=a.ctx, stride=a.stride)
        fz = frozen_words(model, split, a.ctx, a.stride, eos); res['frozen'] = dict(nll=float(fz.mean()), ppl=math.exp(fz.mean())); print(json.dumps(res['frozen']), flush=True)
        ms = training_ms(model, ids['train'], a.lane, 32, a.seed)
        if a.split == 'test':
            sel = json.loads((out_dir/f'{a.select_from}.json').read_text()); assert sel['split'] == 'valid' and sel['run'] == a.run; grid = [(sel['online']['lr'], sel['online']['decay'])]
        else:
            res['online_grid'] = []
            for lr in (1e-5, 3e-5, 1e-4, 3e-4):
                for dc in (0.0, 0.002):
                    o = online_words(model, split[:a.grid_targets], a.ctx, a.online_stride, eos, ms, lr, dc); res['online_grid'].append(dict(lr=lr, decay=dc, nll=float(o.mean())))
                    print(json.dumps(res['online_grid'][-1]), flush=True)
            b = min(res['online_grid'], key=lambda r: r['nll'] if math.isfinite(r['nll']) else math.inf); grid = [(b['lr'], b['decay'])]
        o = online_words(model, split, a.ctx, a.online_stride, eos, ms, *grid[0]); res['online'] = dict(lr=grid[0][0], decay=grid[0][1], stride=a.online_stride, nll=float(o.mean()), ppl=math.exp(o.mean()))
        res.update(status='completed', source_sha256=src)
        with path.open('x') as f: json.dump(res, f, indent=1); f.write('\n')
        print('RESULT', json.dumps(dict(frozen=res['frozen'], online=res['online'])), flush=True); return
    out = out_dir/f'{a.tag}.json'; state_path = out_dir/f'{a.tag}.resume.pt'; ckpt = out_dir/f'{a.tag}.pt'; assert a.tag and not out.exists()
    train = ids['train']; spe = (len(train) - 1)//(a.lane*a.batch); total = spe*a.epochs; warm = max(1, int(0.02*total))
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd, betas=(0.9, 0.98))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s_: min(1.0, (s_ + 1)/warm)*0.5*(1 + math.cos(math.pi*min(1.0, s_/total))))
    ident = {k: v for k, v in vars(a).items() if k not in EXECUTION_ONLY}; epoch, step, best, best_epoch, hist, wall = 0, 0, math.inf, -1, [], 0.0; rng = np.random.default_rng(a.seed)
    if state_path.exists():
        s_ = torch.load(state_path, weights_only=False); assert same_run(s_['args'], ident)
        model.load_state_dict(s_['model']); opt.load_state_dict(s_['opt']); sched.load_state_dict(s_['sched']); torch.set_rng_state(s_['torch_rng']); rng.bit_generator.state = s_['np_rng']
        epoch, step, best, best_epoch, hist, wall = s_['epoch'], s_['step'], s_['best'], s_['best_epoch'], s_['hist'], s_['wall']
    t_start = time.time() - wall
    while epoch < a.epochs:
        e0 = time.time(); off = int(rng.integers(0, a.lane)); n_l = (len(train) - 1 - off)//a.lane
        lanes = np.stack([train[off + i*a.lane: off + (i + 1)*a.lane + 1] for i in range(n_l)]); perm = rng.permutation(n_l); tl = 0.0
        for bi in range(spe):
            w = torch.from_numpy(lanes[perm[bi*a.batch:(bi + 1)*a.batch]]); nll = model.lane_nll(w[:, :-1], w[:, 1:]); loss = nll.mean()
            obj = loss + (model.aux.mean() if model.aux is not None else 0.0); opt.zero_grad(); obj.backward(); nn.utils.clip_grad_norm_(model.parameters(), 0.5); opt.step(); sched.step(); step += 1; tl += loss.item()
            if bi % 50 == 0: print(json.dumps(dict(epoch=epoch, step=bi, of=spe, train_nll=tl/(bi + 1), elapsed_s=round(time.time() - t_start))), flush=True)
            if a.max_steps and step >= a.max_steps:
                vn = frozen_words(model, ids['valid'][:4096], a.ctx, a.stride, eos)
                print(json.dumps(dict(smoke_steps=step, s_per_step=(time.time() - e0)/(bi + 1), smoke_valid4096_nll=float(vn.mean()), max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, parameters=params)), flush=True); return
        vn = float(frozen_words(model, ids['valid'], a.ctx, a.stride, eos).mean()); hist.append(dict(epoch=epoch, train_nll=tl/spe, valid_nll=vn, valid_ppl=math.exp(vn), epoch_s=round(time.time() - e0)))
        print(json.dumps(hist[-1]), flush=True); epoch += 1
        if vn < best: best, best_epoch = vn, epoch - 1; torch.save(model.state_dict(), ckpt)
        tmp = state_path.with_suffix('.tmp'); torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(), torch_rng=torch.get_rng_state(), np_rng=rng.bit_generator.state,
                   epoch=epoch, step=step, best=best, best_epoch=best_epoch, hist=hist, wall=time.time() - t_start, args=ident), tmp); os.replace(tmp, state_path)
    model.load_state_dict(torch.load(ckpt)); m_eval = frozen_words(model, ids['valid'][:8192], a.ctx, a.stride, eos)
    model.eval()
    with torch.no_grad():                                                                          # native share on validation windows
        x = np.concatenate([[eos], ids['valid'][:4096]]); w = torch.from_numpy(x[None]); _, st = model.lane_nll(w[:, :-1], w[:, 1:], stats=True)
    result = dict(status='completed', battle='R1', stage=f'native causal event race member (theory 169): {a.corpus} development fit', tag=a.tag, args=vars(a), parameters=params, classes=int(words.shape[0]),
                  best_epoch=best_epoch, best_valid_nll=best, best_valid_ppl=math.exp(best), history=hist, steps=step, train_wall_s=time.time() - t_start, native_valid4096=st,
                  checkpoint=str(ckpt.relative_to(ROOT)), test_scored=False, max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, source_sha256=src)
    out.write_text(json.dumps(result, indent=1) + '\n'); state_path.unlink()
    print('RESULT', json.dumps({k: result[k] for k in ('best_valid_ppl', 'best_epoch', 'parameters', 'native_valid4096')}), flush=True)


def contracts(cause_messages=False):
    """Exact normalisation, causality, row chunking and exploration-free evaluation on a small synthetic model."""
    torch.manual_seed(0); V = 61; counts = np.random.default_rng(0).integers(0, 50, V).astype(float); cls, words = word_classes(counts, 6, s_max=16)
    m = CERL(V, cls, words, d=32, modes=8, layers=2, dk=16, units=32, topk=3, explore=2, cause_messages=cause_messages); m.eval(); res = {}
    with torch.no_grad():
        for p in m.parameters(): p.add_(torch.randn_like(p)*0.05)
        x = torch.randint(0, 8, (1, 40))                                                             # small alphabet: many repeats/continuations
        t = 25; tot = 0.0
        for v in range(V):
            y = x.clone(); y[0, t] = v; tot += math.exp(-m.lane_nll(x, y)[0, t].item())
        res['normalisation_sum'] = tot
        y = torch.cat([x[:, 1:], torch.randint(0, 8, (1, 1))], 1); full = m.lane_nll(x, y)
        x2, y2 = x.clone(), y.clone(); p = 30; x2[0, p] = (x2[0, p] + 1) % V; y2[0, p - 1] = x2[0, p]
        res['causal_prefix_max_diff'] = float((m.lane_nll(x2, y2)[0, :p - 1] - full[0, :p - 1]).abs().max())
        res['row_chunk_max_diff'] = float((torch.cat([m.lane_nll(x, y, (r, min(r + 7, 40))) for r in range(0, 40, 7)], 1) - full).abs().max())
        res['repeat_eval_identical'] = bool(torch.equal(m.lane_nll(x, y), full))
    mp = CERL(V, cls, words, d=32, modes=8, layers=1, dk=16, units=64, topk=3, explore=0, product_keys=True)
    with torch.no_grad():
        for p_ in mp.parameters(): p_.add_(torch.randn_like(p_)*0.3)
        q_ = torch.randn(5, 7, 16); sel, vs = mp.wake(q_); allu = torch.arange(64).expand(5, 7, 64); full = mp.unit_score(q_, allu)
        res['product_key_topk_max_diff'] = float((vs - full.topk(3, -1).values).abs().max())
    # word-level arms: windowed frozen scoring and causal online adaptation never let a later token change an earlier score
    s1 = np.random.default_rng(1).integers(0, 8, 200); s2 = s1.copy(); q = 120; s2[q] = (s2[q] + 1) % V
    ms = [torch.full_like(p_, 1e-4) for p_ in m.parameters() if p_.requires_grad]; w0 = [p_.detach().clone() for p_ in m.parameters()]
    f1, f2 = frozen_words(m, s1, 64, 16, 0), frozen_words(m, s2, 64, 16, 0); o1, o2 = online_words(m, s1, 64, 16, 0, ms, 1e-3, 0.01), online_words(m, s2, 64, 16, 0, ms, 1e-3, 0.01)
    res.update(window_targets=len(f1), window_prefix_max_diff=float(np.abs(f1[:q] - f2[:q]).max()), online_prefix_max_diff=float(np.abs(o1[:q] - o2[:q]).max()),
               online_first_segment_equals_frozen=float(np.abs(o1[:16] - f1[:16]).max()), online_adapts=float(np.abs(o1[16:] - f1[16:]).max()),
               weights_restored=all(torch.equal(a_, b_) for a_, b_ in zip(m.parameters(), w0)))
    ok = (abs(res['normalisation_sum'] - 1) < 1e-4 and res['causal_prefix_max_diff'] < 1e-5 and res['row_chunk_max_diff'] < 1e-4 and res['repeat_eval_identical'] and res['window_targets'] == 200
          and res['product_key_topk_max_diff'] < 1e-5 and res['window_prefix_max_diff'] == 0 and res['online_prefix_max_diff'] == 0 and res['online_first_segment_equals_frozen'] < 1e-5 and res['online_adapts'] > 0 and res['weights_restored'])
    res['status'] = 'passed' if ok else 'failed'; return res


def main():
    ap = argparse.ArgumentParser(); A = ap.add_argument
    A('--mode', choices=('train', 'contract', 'score'), default='train'); A('--tag', default=''); A('--corpus', default='fineweb'); A('--seed', type=int, default=0)
    A('--d', type=int, default=128); A('--modes', type=int, default=64); A('--layers', type=int, default=2); A('--dk', type=int, default=64)
    A('--units', type=int, default=1024); A('--topk', type=int, default=4); A('--explore', type=int, default=2); A('--classes', type=int, default=256); A('--age-slope', type=float, default=1.5); A('--product-keys', type=int, default=0); A('--cause-messages', type=int, default=0); A('--wd-groups', type=int, default=0); A('--unit-init', type=float, default=0.0); A('--balance', type=float, default=0.0); A('--level1', choices=('tm', 'slots'), default='tm'); A('--slots', type=int, default=1024); A('--slot-m', type=int, default=4); A('--slot-explore', type=int, default=0); A('--slot-read-topk', type=int, default=0); A('--slot-pk', type=int, default=0)
    A('--train-tokens', type=int, default=4194304); A('--lane', type=int, default=256); A('--batch', type=int, default=4); A('--lr', type=float, default=3e-3); A('--wd', type=float, default=0.01)
    A('--row-chunk', type=int, default=512); A('--dropout', type=float, default=0.0); A('--epochs', type=int, default=30); A('--ctx', type=int, default=512); A('--stride', type=int, default=128)
    A('--run', default=''); A('--split', default='valid'); A('--select-from', default=''); A('--online-stride', type=int, default=32); A('--grid-targets', type=int, default=8192)
    A('--checkpoint-every', type=int, default=256); A('--max-steps', type=int, default=0); A('--log-every', type=int, default=100)
    a = ap.parse_args(); torch.set_num_threads(1); torch.manual_seed(a.seed)
    src = {n: hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in ('experiments/r1_cerl.py', 'experiments/r1_deep_keyed_lm.py')}
    if a.mode == 'contract':
        base, msgs = contracts(), contracts(cause_messages=True)
        res = dict(battle='R1', tag=a.tag, **base, cause_messages=msgs, status_all='passed' if base['status'] == msgs['status'] == 'passed' else 'failed', source_sha256=src); print(json.dumps(res))
        if a.tag:
            with (ROOT/f'experiments/results/diagnostics/{a.tag}.json').open('x') as f: json.dump(res, f, indent=1); f.write('\n')
        return
    if a.corpus != 'fineweb': return words_main(a, src)
    train_all = load_fineweb(ROOT/'data/fineweb_gpt2/fineweb_train_000001.bin'); val = load_fineweb(ROOT/'data/fineweb_gpt2/fineweb_val_000000.bin')
    n_lanes = a.train_tokens//a.lane; train = np.asarray(train_all[:n_lanes*a.lane]).reshape(n_lanes, a.lane).astype(np.int64)
    counts = np.bincount(train.reshape(-1), minlength=50257).astype(float); cls, words = word_classes(counts, a.classes)
    model = CERL(50257, cls, words, a.d, a.modes, a.layers, a.dk, a.units, a.topk, a.explore, age_slope=a.age_slope, product_keys=bool(a.product_keys), cause_messages=bool(a.cause_messages), unit_init=a.unit_init, balance=a.balance, level1=a.level1, slots=a.slots, slot_m=a.slot_m, slot_explore=a.slot_explore, slot_read_topk=a.slot_read_topk, slot_pk=bool(a.slot_pk)); params = sum(p.numel() for p in model.parameters())
    order = np.random.default_rng(a.seed).permutation(n_lanes); total = math.ceil(n_lanes/a.batch); quarter = math.ceil(total/4)
    opt = torch.optim.AdamW(param_groups(model, a.wd, bool(a.wd_groups)), lr=a.lr, weight_decay=a.wd); sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=total, pct_start=0.05)
    out_dir = ROOT/'experiments/results/token_language'; out = out_dir/f'{a.tag}.json'; state_path = out_dir/f'{a.tag}.resume.pt'; assert a.tag and not out.exists()
    ident = {k: v for k, v in vars(a).items() if k not in EXECUTION_ONLY}; step, losses, milestones, wall, resumes = 0, [], [], 0.0, 0
    if state_path.exists():
        s = torch.load(state_path, weights_only=False); assert same_run(s['args'], ident), 'resume with different settings'
        model.load_state_dict(s['model']); opt.load_state_dict(s['opt']); sched.load_state_dict(s['sched']); torch.set_rng_state(s['torch_rng'])
        step, losses, milestones, wall, resumes = s['step'], s['losses'], s['milestones'], s['wall'], s['resumes'] + 1; print(json.dumps(dict(resumed_at_step=step)), flush=True)
    began = time.time() - wall
    def save():
        tmp = state_path.with_suffix('.tmp'); torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(), torch_rng=torch.get_rng_state(), step=step, losses=losses,
                   milestones=milestones, wall=time.time() - began, resumes=resumes, args=ident), tmp); os.replace(tmp, state_path)
    while step < total:
        model.train(); w = torch.from_numpy(train[np.sort(order[step*a.batch:(step + 1)*a.batch])]); x, y = w[:, :-1], w[:, 1:]
        nll = model.lane_nll(x, y); loss = nll.mean(); obj = loss + (model.aux.mean() if model.aux is not None else 0.0)
        opt.zero_grad(); obj.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step(); step += 1
        losses.append(loss.item())                                                                   # fresh lanes: the pre-update loss is held out
        if step % a.log_every == 0: print(json.dumps(dict(step=step, of=total, prequential_last=float(np.mean(losses[-a.log_every:])), elapsed_s=round(time.time() - began))), flush=True)
        if step % quarter == 0 or step == total:
            sel = np.asarray(val[10485760:10485760 + 8*2048]).reshape(8, 2048)
            milestones.append(dict(step=step, tokens=step*a.batch*a.lane, select=lanes_nll(model, sel, a.row_chunk, stats=True), prequential_segment=float(np.mean(losses[-quarter:])), elapsed_s=round(time.time() - began)))
            print(json.dumps(milestones[-1]), flush=True)
        if step % a.checkpoint_every == 0 or step == total: save()
        if a.max_steps and step >= a.max_steps:
            print(json.dumps(dict(smoke_steps=step, s_per_step=(time.time() - began)/step, max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, parameters=params)), flush=True)
            print(json.dumps(dict(smoke_select_2lanes=lanes_nll(model, np.asarray(val[10485760:10485760 + 2*2048]).reshape(2, 2048), a.row_chunk, stats=True))), flush=True)
            print(json.dumps(dict(smoke_select_same_tokens_256lanes=lanes_nll(model, np.asarray(val[10485760:10485760 + 2*2048 - 2048 % 256]).reshape(-1, 256), a.row_chunk, stats=True))), flush=True)
            tr = train[order[step*a.batch:step*a.batch + 16]]; print(json.dumps(dict(smoke_unseen_train_lanes=lanes_nll(model, tr, a.row_chunk, stats=True))), flush=True); return
    from torch.utils.flop_counter import FlopCounterMode
    with FlopCounterMode(display=False) as fc:
        model.train(); w = torch.from_numpy(train[:1]); model.lane_nll(w[:, :-1], w[:, 1:]).mean().backward()
    model.zero_grad(); fpt = fc.get_total_flops()/(a.lane - 1); seg = max(1, total//8)
    result = dict(status='completed', battle='R1', stage='native causal event race member (theory 169), one pass', tag=a.tag, args=vars(a), parameters=params, steps=step,
                  presentations=step*a.batch*(a.lane - 1), train_wall_s=time.time() - began, resumes=resumes, milestones=milestones, train_flops_per_target=fpt, train_flops=fpt*step*a.batch*(a.lane - 1),
                  prequential_by_eighth=[float(np.mean(losses[i:i + seg])) for i in range(0, len(losses), seg)], prequential_last_quarter=float(np.mean(losses[-quarter:])),
                  **fineweb_protocols(model, val, a.row_chunk), work_per_target_at_history_255=work_per_target(model, 255), classes=int(words.shape[0]),
                  reference=dict(diagnostic_planB_L8_4M_flagship=5.8414, diagnostic_planB_L2_4M_flagship=5.8648, diagnostic_planB_L8_train_flops_per_target=25847808.0),
                  max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, source_sha256=src)
    torch.save(model.state_dict(), out_dir/f'{a.tag}.pt'); result['checkpoint'] = f'experiments/results/token_language/{a.tag}.pt'
    out.write_text(json.dumps(result, indent=1) + '\n'); state_path.unlink()
    print('RESULT', json.dumps({k: result[k] for k in ('select', 'score_65528', 'flagship', 'train_flops_per_target', 'parameters')}), flush=True)


if __name__ == '__main__':
    main()
