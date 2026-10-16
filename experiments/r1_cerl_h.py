#!/usr/bin/env python3
"""R1 compositional hierarchy of causal event races (CERL-H; theory/169 section 8).

Level 1 is the causal event race over tokens (r1_cerl.CERL). Its posterior attributes every observed token to one cause:
a woken concept unit u, the background route, an episodic repeat or an episodic continuation. The stream of attributed
causes is compressed in time: a level-2 event occurs only when the cause changes (a copied span or a persistent concept is
one event). Level 2 is the same race (r1_cerl.CERL) over that compressed cause stream, so its episodic pointers repeat and
continue *patterns of causes* (structure-level induction) and its dormant concepts propose which lower causes come next.

Top-down composition, sparse: at token row t, level 2's state after the event that holds the cause of x_t predicts the
next lower cause. The prior over the cause of x_{t+1} is
    stay with the current cause:      log sigmoid(g_t),            g_t = a + b log1p(age of the current event);
    switch to cause c:                log(1 - sigmoid(g_t)) + log p2(c | cause events so far),
evaluated only for the candidate causes (level-1 key-woken units, a few top-down-woken units proposed by level 2's
strongest continuation pointers, and the three route types). lambda * prior enters the level-1 race log-rates
(r1_cerl.CERL.race td hook), so the token law stays exactly normalised; lambda starts at 0, where CERL-H equals level 1.
Learning is local per level: level 1 on tokens (plus the top-down path), level 2 on predicting the next attributed cause;
no long chain of gradients through depth. Nothing dense is added: no attention, no full-alphabet distributions, higher
levels compute only at their events.
"""
import argparse, hashlib, json, math, os, resource, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
ROOT = Path(__file__).resolve().parents[1]; sys.path[:0] = [str(ROOT/'experiments'), str(ROOT/'experiments/tpp')]
from r1_cerl import CERL, word_classes, match_lengths, load_fineweb, lanes_nll, fineweb_protocols, work_per_target  # noqa: E402

EXECUTION_ONLY = ('max_steps', 'checkpoint_every', 'log_every', 'row_chunk')


class CERLH(nn.Module):
    cause_msgs = False

    def __init__(self, V, cls, words, l1, l2, k_td=2, alpha=0.5):
        super().__init__(); self.L1 = CERL(V, cls, words, **l1); self.U1 = l1['units']; self.V2 = self.U1 + 3; self.k_td, self.alpha = k_td, alpha
        self.BG, self.REP, self.CONT = self.U1, self.U1 + 1, self.U1 + 2
        c2, w2 = word_classes(np.ones(self.V2), 32, trim=True); self.L2 = CERL(self.V2, c2, w2, **l2)
        self.td_gain = nn.Parameter(torch.tensor(0.0)); self.td_wake = nn.Parameter(torch.tensor(-12.0)); self.stay_a = nn.Parameter(torch.tensor(1.0)); self.stay_b = nn.Parameter(torch.tensor(0.5))
        self._px = None; self.aux = None; self.last = {}

    # ---- level-1 attribution: who caused each observed token (posterior argmax; an observation, no gradient)
    @torch.no_grad()
    def attribute(self, x, y, h, chunk=512):
        B, L = x.shape; out = []; h = h.detach()                                                  # an observation: no graph (also keeps FLOP-counter module hooks valid)
        for r0 in range(0, L, chunk):
            r1 = min(r0 + chunk, L); _, parts, s_sel, sel, *_ = self.L1.race(x, y, h, r0, r1, False); post = torch.softmax(parts, -1); K = s_sel.shape[-1]
            g = torch.cat([post[..., :1], post[..., 1:1 + K], post[..., 1 + K:1 + K + r1].sum(-1, keepdim=True), post[..., 1 + K + r1:].sum(-1, keepdim=True)], -1)
            sym = torch.cat([torch.full_like(sel[..., :1], self.BG), sel, torch.full_like(sel[..., :1], self.REP), torch.full_like(sel[..., :1], self.CONT)], -1)
            out.append(sym.gather(-1, g.argmax(-1, keepdim=True)).squeeze(-1))
        return torch.cat(out, 1)

    # ---- level-2 race components at every event row (sparse: candidates are scored, no full distribution)
    def l2_components(self, e):
        m = self.L2; x2 = e[None]; h2 = m.states(x2); M = e.shape[0]; dv = e.device
        t = torch.arange(M, device=dv)[:, None]; j = torch.arange(M, device=dv)[None, :]; age = (t - j).clamp_min(0).float(); ml = match_lengths(x2, 0, M)
        S_rep = ((m.q_rep(h2) @ m.k_rep(h2).transpose(1, 2))/math.sqrt(m.dk) + m.b_rep - F.softplus(m.age_rep)*torch.log1p(age)).masked_fill(~(j <= t), float('-inf'))[0]
        S_cont = ((m.q_cont(h2) @ m.k_cont(h2).transpose(1, 2))/math.sqrt(m.dk) + m.b_cont - F.softplus(m.age_cont)*torch.log1p(age)
                  + m.span*torch.log1p(ml.float())).masked_fill(~(j < t), float('-inf'))[0]
        sel, s_sel = m.wake(m.q_unit(h2)); s_bg = m.bg(h2)
        logZ = torch.logsumexp(torch.cat([s_bg, s_sel, S_rep[None], S_cont[None]], -1), -1)[0]
        H = torch.cat([h2[:, :, None], h2[:, :, None] + m.unit_out[sel]], 2)[0]                         # [M, 1+K, d2]
        nxt = torch.cat([e[1:], e[-1:]])
        k = min(self.k_td, M); top = S_cont.topk(k, -1).indices; extras = nxt[top]                       # symbols proposed by the strongest continuation pointers
        if k < self.k_td: extras = torch.cat([extras, extras[:, :1].expand(-1, self.k_td - k)], -1)
        return dict(S_rep=S_rep, S_cont=S_cont, s_sel=s_sel[0], s_bg=s_bg[0], logZ=logZ, H=H, e=e, nxt=nxt, extras=extras)

    def l2_logp(self, c2, rows, cands):
        """log p2(cand) at level-2 rows `rows` [N] for candidates [N, Mc]."""
        m = self.L2; H = c2['H'][rows]; Mc = cands.shape[-1]
        lp = torch.stack([m.two_level(H, cands[:, i]) for i in range(Mc)], 1)                           # [N, Mc, 1+K]
        Sr, Sc = c2['S_rep'][rows], c2['S_cont'][rows]; hr = c2['e'][None, None, :] == cands[..., None]; hc = c2['nxt'][None, None, :] == cands[..., None]
        parts = torch.cat([c2['s_bg'][rows][:, None] + lp[..., :1], c2['s_sel'][rows][:, None] + lp[..., 1:], torch.where(hr, Sr[:, None], float('-inf')), torch.where(hc, Sc[:, None], float('-inf'))], -1)
        return torch.logsumexp(parts, -1) - c2['logZ'][rows][:, None]

    def prepare(self, x, y):
        """Attribution, compression, level-2 local loss and the top-down prior for every token row of lanes x."""
        B, L = x.shape; dv = x.device; h = self.L1.states(x); sym = self.attribute(x, y, h)
        sel_key = self.L1.wake(self.L1.q_unit(h))[0]                                                    # [B, L, k1] (recomputed identically inside the race)
        k1 = sel_key.shape[-1]; nC = k1 + self.k_td + 3; logprior = torch.zeros(B, L, nC, device=dv); extras_all = sel_key[..., :1].expand(-1, -1, self.k_td).clone()
        l2_losses, l2_aux, comp = [], [], []
        for b in range(B):
            s = sym[b]; change = torch.ones_like(s, dtype=torch.bool); change[1:] = s[1:] != s[:-1]; e = s[change]; M = e.shape[0]
            k_row = torch.cumsum(change.long(), 0) - 1; start = torch.nonzero(change).squeeze(-1); comp.append(M/L)
            if M >= 2:
                l2 = self.L2.lane_nll(e[None, :-1], e[None, 1:]); l2_losses.append(l2.mean())
                if self.L2.aux is not None: l2_aux.append(self.L2.aux.mean())
            c2 = self.l2_components(e)
            rows_t = torch.arange(1, L, device=dv); kk = k_row[rows_t - 1]                                # token row t uses the event holding the cause of x_t
            ex = c2['extras'][kk]; ex_units = torch.where(ex < self.U1, ex, sel_key[b, rows_t, :1].expand(-1, self.k_td))   # route-type extras map to a duplicate (masked in the race)
            extras_all[b, rows_t] = ex_units
            cands = torch.cat([sel_key[b, rows_t], ex_units, torch.tensor([self.BG, self.REP, self.CONT], device=dv).expand(L - 1, 3)], -1)
            lp2 = self.l2_logp(c2, kk, cands)
            age = (rows_t - 1 - start[kk]).float(); g = self.stay_a + self.stay_b*torch.log1p(age)
            prev = s[rows_t - 1]; stay = cands == prev[:, None]
            logprior[b, 1:] = torch.where(stay, F.logsigmoid(g)[:, None], F.logsigmoid(-g)[:, None] + lp2)
        extra = (self.alpha*torch.stack(l2_losses).mean() if l2_losses else 0.0) + (torch.stack(l2_aux).mean() if l2_aux else 0.0)
        self._prep = dict(h=h, logprior=logprior, extras=extras_all, k1=k1, extra=extra, comp=float(np.mean(comp)), l2_nll=float(torch.stack(l2_losses).mean()) if l2_losses else float('nan'))
        self._px = x

    def lane_nll(self, x, y, rows=None, stats=False, msg=None):
        if self._px is not x or not self.training is self._prep_training: self.prepare(x, y); self._prep_training = self.training
        p = self._prep; B, L = x.shape; r0, r1 = (0, L) if rows is None else rows; k1 = p['k1']; bonus = self.td_gain*p['logprior'][:, r0:r1]
        ub = torch.cat([bonus[..., :k1], bonus[..., k1:k1 + self.k_td] + self.td_wake], -1)             # top-down-woken units enter through a learned gate (closed at init)
        td = dict(sel_extra=p['extras'][:, r0:r1], unit_bonus=lambda sel, b_=ub: b_, b_bg=bonus[..., -3:-2], b_rep=bonus[..., -2:-1], b_cont=bonus[..., -1:])
        nll, parts, s_sel, sel, lp, s_bg, S_rep, S_cont = self.L1.race(x, y, p['h'], r0, r1, self.training and self.L1.explore > 0, td=td)
        a1 = self.L1.aux; self.aux = (a1 if a1 is not None else torch.zeros_like(nll)) + (p['extra']*(R := (r1 - r0))/R if torch.is_tensor(p['extra']) else 0.0)
        if not stats: return nll
        with torch.no_grad():
            post = torch.softmax(parts, -1); K = s_sel.shape[-1]
            st = dict(bg_only_nll=(-lp[..., 0]).mean().item(), resp_bg=post[..., 0].mean().item(), resp_concept=post[..., 1:1 + K].sum(-1).mean().item(),
                      resp_repeat=post[..., 1 + K:1 + K + r1].sum(-1).mean().item(), resp_continue=post[..., 1 + K + r1:].sum(-1).mean().item(),
                      td_gain=self.td_gain.item(), l2_local_nll=p['l2_nll'], compression=p['comp'])
        return nll, st

    _prep_training = None


def build(a, cls, words, V=50257):
    l1 = dict(d=a.d, modes=a.modes, layers=a.layers, dk=a.dk, units=a.units, topk=a.topk, explore=a.explore, product_keys=bool(a.product_keys))
    l2 = dict(d=a.d2, modes=a.modes2, layers=a.layers2, dk=a.dk2, units=a.units2, topk=4, explore=1, product_keys=False)
    return CERLH(V, cls, words, l1, l2, k_td=a.k_td, alpha=a.alpha)


def contracts():
    """Exact normalisation, causality through attribution/compression/top-down, row chunking, and equality with level 1 at lambda 0."""
    torch.manual_seed(0); V = 61; counts = np.random.default_rng(0).integers(0, 50, V).astype(float); cls, words = word_classes(counts, 6, s_max=16)
    l1 = dict(d=32, modes=8, layers=1, dk=16, units=32, topk=3, explore=0); l2 = dict(d=16, modes=8, layers=1, dk=8, units=16, topk=2, explore=0)
    m = CERLH(V, cls, words, l1, l2, k_td=2); m.eval(); res = {}
    with torch.no_grad():
        for p_ in m.parameters(): p_.add_(torch.randn_like(p_)*0.05)
        x = torch.randint(0, 8, (1, 40)); y = torch.cat([x[:, 1:], torch.randint(0, 8, (1, 1))], 1)
        m.td_gain.fill_(0.0); base = m.L1.lane_nll(x, y); res['lambda0_equals_level1'] = float((m.lane_nll(x.clone(), y) - base).abs().max())
        m.td_gain.fill_(1.0); full = m.lane_nll(x.clone(), y)
        t = 25; tot = 0.0
        for v in range(V):
            yy = y.clone(); yy[0, t] = v; tot += math.exp(-m.lane_nll(x.clone(), yy)[0, t].item())
        res['normalisation_sum'] = tot
        x2, y2 = x.clone(), y.clone(); p = 30; x2[0, p] = (x2[0, p] + 1) % V; y2[0, p - 1] = x2[0, p]
        res['causal_prefix_max_diff'] = float((m.lane_nll(x2, y2)[0, :p - 1] - full[0, :p - 1]).abs().max())
        xc = x.clone(); res['row_chunk_max_diff'] = float((torch.cat([m.lane_nll(xc, y, (r, min(r + 7, 40))) for r in range(0, 40, 7)], 1) - full).abs().max())
        res['topdown_changes_law'] = float((full - base).abs().max())
    ok = res['lambda0_equals_level1'] < 1e-4 and abs(res['normalisation_sum'] - 1) < 1e-4 and res['causal_prefix_max_diff'] < 1e-5 and res['row_chunk_max_diff'] < 1e-4 and res['topdown_changes_law'] > 0
    res['status'] = 'passed' if ok else 'failed'; return res


def main():
    ap = argparse.ArgumentParser(); A = ap.add_argument
    A('--mode', choices=('train', 'contract'), default='train'); A('--tag', default=''); A('--seed', type=int, default=0)
    A('--d', type=int, default=128); A('--modes', type=int, default=64); A('--layers', type=int, default=2); A('--dk', type=int, default=64)
    A('--units', type=int, default=1024); A('--topk', type=int, default=4); A('--explore', type=int, default=2); A('--product-keys', type=int, default=0); A('--classes', type=int, default=256)
    A('--d2', type=int, default=64); A('--modes2', type=int, default=32); A('--layers2', type=int, default=1); A('--dk2', type=int, default=32); A('--units2', type=int, default=256)
    A('--k-td', type=int, default=2); A('--alpha', type=float, default=0.5)
    A('--train-tokens', type=int, default=4194304); A('--lane', type=int, default=256); A('--batch', type=int, default=4); A('--lr', type=float, default=3e-3); A('--wd', type=float, default=0.01)
    A('--row-chunk', type=int, default=512); A('--checkpoint-every', type=int, default=256); A('--max-steps', type=int, default=0); A('--log-every', type=int, default=100)
    a = ap.parse_args(); torch.set_num_threads(1); torch.manual_seed(a.seed)
    src = {n: hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in ('experiments/r1_cerl_h.py', 'experiments/r1_cerl.py', 'experiments/r1_deep_keyed_lm.py')}
    if a.mode == 'contract':
        res = dict(battle='R1', tag=a.tag, **contracts(), source_sha256=src); print(json.dumps(res))
        if a.tag:
            with (ROOT/f'experiments/results/diagnostics/{a.tag}.json').open('x') as f: json.dump(res, f, indent=1); f.write('\n')
        return
    train_all = load_fineweb(ROOT/'data/fineweb_gpt2/fineweb_train_000001.bin'); val = load_fineweb(ROOT/'data/fineweb_gpt2/fineweb_val_000000.bin')
    n_lanes = a.train_tokens//a.lane; train = np.asarray(train_all[:n_lanes*a.lane]).reshape(n_lanes, a.lane).astype(np.int64)
    cls, words = word_classes(np.bincount(train.reshape(-1), minlength=50257).astype(float), a.classes); model = build(a, cls, words); params = sum(p.numel() for p in model.parameters())
    order = np.random.default_rng(a.seed).permutation(n_lanes); total = math.ceil(n_lanes/a.batch); quarter = math.ceil(total/4)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd); sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=total, pct_start=0.05)
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
            print(json.dumps(dict(step=step, of=total, prequential_last=float(np.mean(losses[-a.log_every:])), td_gain=model.td_gain.item(), l2_nll=model._prep['l2_nll'], compression=model._prep['comp'],
                                  elapsed_s=round(time.time() - began))), flush=True)
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
    result = dict(status='completed', battle='R1', stage='native compositional hierarchy of causal event races (CERL-H, theory 169 section 8), one pass', tag=a.tag, args=vars(a), parameters=params, steps=step,
                  presentations=step*a.batch*(a.lane - 1), train_wall_s=time.time() - began, resumes=resumes, milestones=milestones, train_flops_per_target=fpt, train_flops=fpt*step*a.batch*(a.lane - 1),
                  prequential_by_eighth=[float(np.mean(losses[i:i + seg])) for i in range(0, len(losses), seg)], prequential_last_quarter=float(np.mean(losses[-quarter:])),
                  **fineweb_protocols(model, val, a.row_chunk), work_level1_per_target=work_per_target(model.L1, 255), td_gain=model.td_gain.item(),
                  reference=dict(cerl_level1_4M='curie_r1_cerl_4M_s0_20261016_v1', diagnostic_planB_L8_4M_flagship=5.8414), max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, source_sha256=src)
    torch.save(model.state_dict(), out_dir/f'{a.tag}.pt'); result['checkpoint'] = f'experiments/results/token_language/{a.tag}.pt'
    out.write_text(json.dumps(result, indent=1) + '\n'); state_path.unlink()
    print('RESULT', json.dumps({k: result[k] for k in ('select', 'score_65528', 'flagship', 'train_flops_per_target', 'parameters', 'td_gain')}), flush=True)


if __name__ == '__main__':
    main()
