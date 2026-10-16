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

S_MAX = 512
EXECUTION_ONLY = ('max_steps', 'checkpoint_every', 'mode', 'row_chunk', 'log_every')


def word_classes(counts, n_classes, s_max=S_MAX):
    """Frequency-mass word classes: words by descending training count; a class closes at 1/n_classes of the mass or at
    s_max words. Returns class_of [V] and class_words [C, s_max] (-1 padded)."""
    V = len(counts); order = np.argsort(-counts, kind='stable'); p = (counts + 0.5)/(counts + 0.5).sum(); cls, cur, mass, out = np.zeros(V, np.int64), [], 0.0, []
    for w in order:
        cur.append(w); mass += p[w]
        if mass >= 1.0/n_classes or len(cur) == s_max: out.append(cur); cur, mass = [], 0.0
    if cur: out.append(cur)
    words = np.full((len(out), s_max), -1, np.int64)
    for c, ws in enumerate(out): cls[ws] = c; words[c, :len(ws)] = ws
    return cls, words


def match_lengths(x, cap=8):
    """l[b, t, j] = length of the common suffix of x[..t] and x[..j] (capped), for j < t."""
    eq = (x[:, :, None] == x[:, None, :]); l = eq.long()
    for _ in range(cap - 1):
        sh = torch.zeros_like(l); sh[:, 1:, 1:] = l[:, :-1, :-1]; l = eq.long()*(1 + sh)
    return l.clamp_max(cap)


class CERL(nn.Module):
    def __init__(self, V, class_of, class_words, d=128, modes=64, layers=2, dk=64, units=1024, topk=4, explore=2, tm_chunk=64, age_slope=1.5):
        super().__init__(); self.V, self.d, self.dk, self.U, self.topk, self.explore = V, d, dk, units, topk, explore
        self.register_buffer('class_of', torch.as_tensor(class_of)); self.register_buffer('class_words', torch.as_tensor(class_words)); C = class_words.shape[0]
        self.embed = nn.Embedding(V, d); nn.init.normal_(self.embed.weight, std=0.02); self.word_bias = nn.Parameter(torch.zeros(V))
        self.class_emb = nn.Parameter(torch.randn(C, d)*0.02); self.class_bias = nn.Parameter(torch.zeros(C))
        self.n1 = nn.ModuleList(nn.LayerNorm(d) for _ in range(layers)); self.tm = nn.ModuleList(ParallelTemporalMemory(d, modes) for _ in range(layers))
        self.tm_out = nn.ModuleList(nn.Linear(2*modes, d) for _ in range(layers)); self.n2 = nn.ModuleList(nn.LayerNorm(d) for _ in range(layers))
        self.mlp = nn.ModuleList(nn.Sequential(nn.Linear(d, 2*d), nn.GELU(), nn.Linear(2*d, d)) for _ in range(layers)); self.out_norm = nn.LayerNorm(d)
        for t in self.tm: t.CHUNK = tm_chunk
        # race clocks
        self.bg = nn.Linear(d, 1); nn.init.zeros_(self.bg.weight); nn.init.constant_(self.bg.bias, 2.0)
        self.q_unit = nn.Linear(d, dk); self.unit_key = nn.Parameter(torch.randn(units, dk)/math.sqrt(dk)); self.unit_bias = nn.Parameter(torch.zeros(units))
        self.unit_out = nn.Parameter(torch.zeros(units, d))
        self.q_rep, self.k_rep, self.q_cont, self.k_cont = (nn.Linear(d, dk) for _ in range(4))
        self.b_rep, self.b_cont = nn.Parameter(torch.tensor(-2.0)), nn.Parameter(torch.tensor(-2.0))
        a0 = math.log(math.expm1(age_slope))   # power-law excitation kernel age^-slope: integrable for slope > 1, so total episodic hazard stays bounded as history grows
        self.age_rep, self.age_cont = nn.Parameter(torch.tensor(a0)), nn.Parameter(torch.tensor(a0)); self.span = nn.Parameter(torch.tensor(1.0))

    def states(self, x):
        h = self.embed(x)
        for n1, tm, to, n2, mlp in zip(self.n1, self.tm, self.tm_out, self.n2, self.mlp):
            h = h + to(tm(n1(h))); h = h + mlp(n2(h))
        return self.out_norm(h)

    def two_level(self, H, y):
        """log p(y) under the two-level word law of each state row in H [N, K, d] for targets y [N]."""
        lc = torch.log_softmax(H @ self.class_emb.T + self.class_bias, -1)                              # [N, K, C]
        c = self.class_of[y]; words = self.class_words[c]; ok = words >= 0; wi = words.clamp_min(0)      # [N, S]
        lw = torch.einsum('nkd,nsd->nks', H, self.embed.weight[wi]) + self.word_bias[wi][:, None, :]
        lw = torch.log_softmax(lw.masked_fill(~ok[:, None, :], float('-inf')), -1)
        pos = (words == y[:, None]).float().argmax(-1)                                                 # target's slot in its class
        return lc.gather(-1, c[:, None, None].expand(-1, H.shape[1], 1)).squeeze(-1) + lw.gather(-1, pos[:, None, None].expand(-1, H.shape[1], 1)).squeeze(-1)

    def lane_nll(self, x, y, rows=None, stats=False):
        """Per-target NLL [B, R] for query rows `rows` of lanes x [B, L] (inputs) with targets y [B, L]."""
        B, L = x.shape; h = self.states(x); r0, r1 = (0, L) if rows is None else rows; R = r1 - r0; hq = h[:, r0:r1]
        t = torch.arange(r0, r1)[:, None]; j = torch.arange(r1)[None, :]; age = (t - j).clamp_min(0).float()
        ml = match_lengths(x[:, :r1])[:, r0:r1]                                                      # [B, R, r1]
        S_rep = (self.q_rep(hq) @ self.k_rep(h[:, :r1]).transpose(1, 2))/math.sqrt(self.dk) + self.b_rep - F.softplus(self.age_rep)*torch.log1p(age)
        S_rep = S_rep.masked_fill(~(j <= t), float('-inf'))
        S_cont = (self.q_cont(hq) @ self.k_cont(h[:, :r1]).transpose(1, 2))/math.sqrt(self.dk) + self.b_cont - F.softplus(self.age_cont)*torch.log1p(age) + self.span*torch.log1p(ml.float())
        S_cont = S_cont.masked_fill(~(j < t), float('-inf'))
        su = self.q_unit(hq) @ self.unit_key.T + self.unit_bias                                      # [B, R, U]
        sel = su.topk(self.topk, -1).indices; s_sel = su.gather(-1, sel)                               # woken units: the predictive race (train = eval)
        s_bg = self.bg(hq)                                                                           # [B, R, 1]
        logZ = torch.logsumexp(torch.cat([s_bg, s_sel, S_rep, S_cont], -1), -1)
        yq = y[:, r0:r1]; H = torch.cat([hq[:, :, None], hq[:, :, None] + self.unit_out[sel]], 2)      # [B, R, 1+K, d]
        explore = self.training and self.explore > 0
        if explore:
            # Paid counterfactual alternatives: random dormant units ask "would including me have raised p(y)?" against the
            # actual race held fixed (detached). Credit reaches only their keys, biases, vectors and the unit query map; they
            # never enter the predictive mixture, so training and evaluation predict with the same race.
            ex = torch.randint(0, self.U, (B, R, self.explore)); hd = hq.detach()
            s_ex = (self.q_unit(hd)[:, :, None, :]*self.unit_key[ex]).sum(-1) + self.unit_bias[ex]
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
        if explore:
            race = torch.cat([s_bg, s_sel, S_rep, S_cont], -1).detach()
            self.aux = torch.logsumexp(torch.cat([race, s_ex], -1), -1) - torch.logsumexp(torch.cat([parts.detach(), s_ex + lp_ex], -1), -1)
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
    state = sum(p.numel() for n, p in m.named_parameters() if n.startswith(('tm', 'tm_out', 'mlp')))
    return dict(state_macs=state, episodic_candidates=2*history, episodic_macs=4*d*dk + 2*history*dk, unit_scores=m.U, unit_macs=d*dk + m.U*dk, concepts_woken=m.topk,
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
            for r0 in range(0, L, row_chunk):
                out = model.lane_nll(x, y, (r0, min(r0 + row_chunk, L)), stats=stats)
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


def contracts():
    """Exact normalisation, causality, row chunking and exploration-free evaluation on a small synthetic model."""
    torch.manual_seed(0); V = 61; counts = np.random.default_rng(0).integers(0, 50, V).astype(float); cls, words = word_classes(counts, 6, s_max=16)
    m = CERL(V, cls, words, d=32, modes=8, layers=2, dk=16, units=32, topk=3, explore=2); m.eval(); res = {}
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
    ok = abs(res['normalisation_sum'] - 1) < 1e-4 and res['causal_prefix_max_diff'] < 1e-5 and res['row_chunk_max_diff'] < 1e-4 and res['repeat_eval_identical']
    res['status'] = 'passed' if ok else 'failed'; return res


def main():
    ap = argparse.ArgumentParser(); A = ap.add_argument
    A('--mode', choices=('train', 'contract'), default='train'); A('--tag', default=''); A('--corpus', default='fineweb'); A('--seed', type=int, default=0)
    A('--d', type=int, default=128); A('--modes', type=int, default=64); A('--layers', type=int, default=2); A('--dk', type=int, default=64)
    A('--units', type=int, default=1024); A('--topk', type=int, default=4); A('--explore', type=int, default=2); A('--classes', type=int, default=256); A('--age-slope', type=float, default=1.5)
    A('--train-tokens', type=int, default=4194304); A('--lane', type=int, default=256); A('--batch', type=int, default=4); A('--lr', type=float, default=3e-3); A('--wd', type=float, default=0.01)
    A('--row-chunk', type=int, default=512); A('--checkpoint-every', type=int, default=256); A('--max-steps', type=int, default=0); A('--log-every', type=int, default=100)
    a = ap.parse_args(); torch.set_num_threads(1); torch.manual_seed(a.seed)
    src = {n: hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in ('experiments/r1_cerl.py', 'experiments/r1_deep_keyed_lm.py')}
    if a.mode == 'contract':
        res = dict(battle='R1', tag=a.tag, **contracts(), source_sha256=src); print(json.dumps(res))
        if a.tag:
            with (ROOT/f'experiments/results/diagnostics/{a.tag}.json').open('x') as f: json.dump(res, f, indent=1); f.write('\n')
        return
    assert a.corpus == 'fineweb', 'word-level corpora: next driver revision'
    train_all = load_fineweb(ROOT/'data/fineweb_gpt2/fineweb_train_000001.bin'); val = load_fineweb(ROOT/'data/fineweb_gpt2/fineweb_val_000000.bin')
    n_lanes = a.train_tokens//a.lane; train = np.asarray(train_all[:n_lanes*a.lane]).reshape(n_lanes, a.lane).astype(np.int64)
    counts = np.bincount(train.reshape(-1), minlength=50257).astype(float); cls, words = word_classes(counts, a.classes)
    model = CERL(50257, cls, words, a.d, a.modes, a.layers, a.dk, a.units, a.topk, a.explore, age_slope=a.age_slope); params = sum(p.numel() for p in model.parameters())
    order = np.random.default_rng(a.seed).permutation(n_lanes); total = math.ceil(n_lanes/a.batch); quarter = math.ceil(total/4)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd); sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=total, pct_start=0.05)
    out_dir = ROOT/'experiments/results/token_language'; out = out_dir/f'{a.tag}.json'; state_path = out_dir/f'{a.tag}.resume.pt'; assert a.tag and not out.exists()
    ident = {k: v for k, v in vars(a).items() if k not in EXECUTION_ONLY}; step, losses, milestones, wall, resumes = 0, [], [], 0.0, 0
    if state_path.exists():
        s = torch.load(state_path, weights_only=False); assert s['args'] == ident, 'resume with different settings'
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
