#!/usr/bin/env python3
"""B10: Poisson, multivariate Hawkes and race-of-clocks models of the BTCUSDT aggTrades stream under the frozen protocol
(experiments/market/B10_MARKET.md).

Windows: each day's events cut into consecutive windows of 1,024; times in seconds from the window's first event (keeps
the 1 µs recording cell exact in float64). Scored events: positions 128..1023 of every window (context 0..127).
Training: every 20th window of the training days (all positions); validation: every 10th window of the validation days
(scored positions); test (--score-test): every window of the test days (scored positions), once.
Per event: log p = log[S(g) − S(g + c)] + log π_k(g + c/2), c = 1 µs: the interval likelihood of the recording cell
holding the gap, and the mark law at the cell's midpoint, identical for every model.
"""
import argparse
import datetime as dt
import hashlib
import json
import math
import sys
import time
import resource
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'experiments/tpp'))
from race_tpp_v5 import RaceTPP  # noqa: E402

NPZ = ROOT / 'data/binance/npz'
CELL = 1e-6; W = 1024; CTX = 128; K = 6
TRAIN = [dt.date(2026, 8, 1) + dt.timedelta(d) for d in range(24)]
VAL = [dt.date(2026, 8, 25) + dt.timedelta(d) for d in range(7)]
TEST = [dt.date(2026, 9, 1) + dt.timedelta(d) for d in range(7)]


def windows(day, every):
    z = np.load(NPZ / f'{day}.npz'); t, m = z['t'], z['mark'].astype(np.int64)
    n = len(t) // W; out = []
    for w in range(0, n, every):
        tt = t[w * W:(w + 1) * W]; out.append(((tt - tt[0]) * 1e-6, m[w * W:(w + 1) * W]))
    return out


def log_interval(log_s0, log_s1):
    """log(S0 − S1) for log S0 ≥ log S1."""
    d = (log_s1 - log_s0).clamp_max(-1e-12)
    return log_s0 + torch.log(-torch.expm1(d))


class Poisson(nn.Module):
    def __init__(self, rates):
        super().__init__(); self.log_lam = nn.Parameter(torch.log(torch.tensor(rates)))

    def terms(self, t, m):
        g = t[:, 1:] - t[:, :-1]; lam = self.log_lam.exp(); tot = lam.sum()
        log_s0, log_s1 = -tot * g, -tot * (g + CELL)
        log_pi = (self.log_lam - torch.logsumexp(self.log_lam, 0))[m[:, 1:]]
        return log_interval(log_s0, log_s1) + log_pi


class Hawkes(nn.Module):
    """λ_k(t) = μ_k + Σ_r Σ_j α_{r,k,m_j} β_r e^{−β_r (t − t_j)}; exact compensator."""

    def __init__(self, rates, betas=(100.0, 1.0)):
        super().__init__()
        self.log_mu = nn.Parameter(torch.log(torch.tensor(rates) * 0.5))
        self.log_alpha = nn.Parameter(torch.full((len(betas), K, K), math.log(0.05)))
        self.log_beta = nn.Parameter(torch.log(torch.tensor(betas)))

    def terms(self, t, m):
        B, L = t.shape; mu = self.log_mu.exp(); al = self.log_alpha.exp(); be = self.log_beta.exp()   # [R,K,K], [R]
        A = torch.zeros(B, len(be), K, dtype=t.dtype)                         # Σ_j e^{−β_r (t − t_j)} per source mark
        oh = torch.nn.functional.one_hot(m, K).to(t.dtype)
        out = []
        for i in range(1, L):
            if i == 1:
                A = oh[:, 0].unsqueeze(1).expand(B, len(be), K).clone()           # state right after event 0
            else:
                A = A * torch.exp(-be.view(1, -1, 1) * (t[:, i - 1] - t[:, i - 2]).view(B, 1, 1)) + oh[:, i - 1].unsqueeze(1)
            g = (t[:, i] - t[:, i - 1]).unsqueeze(1)                           # [B,1]
            exc = torch.einsum('rkm,brm->brk', al, A)                          # Σ_m α A  [B,R,K]
            def comp(x):
                return mu.sum() * x + (exc.sum(-1) * (1 - torch.exp(-be * x))).sum(-1, keepdim=True)
            log_s0, log_s1 = -comp(g), -comp(g + CELL)
            lam = mu + (exc * be.view(1, -1, 1) * torch.exp(-be.view(1, -1, 1) * (g + CELL / 2).unsqueeze(-1))).sum(1)
            log_pi = torch.log(lam.gather(1, m[:, i:i + 1])) - torch.log(lam.sum(1, keepdim=True))
            out.append((log_interval(log_s0, log_s1) + log_pi).squeeze(1))
        return torch.stack(out, 1)


class IdentityGrowthLayer(nn.Module):
    """Live temporal features with an initially zero correction output map."""
    def __init__(self, temporal, d):
        super().__init__()
        self.temporal=temporal
        self.correction=nn.Linear(d,d,bias=False).to(device=temporal.read.weight.device,dtype=temporal.read.weight.dtype)
        nn.init.zeros_(self.correction.weight)

    def forward(self,x,dt):
        return x+self.correction(self.temporal(x,dt)-x)


def grow_from_parent(model, state):
    own=model.state_dict()
    missing=set(own)-set(state)
    if set(state)-set(own) or any(not key.startswith('m.layers.') or int(key.split('.')[2])<2 for key in missing):
        raise ValueError('Growth requires the matching completed two-layer native model')
    model.load_state_dict(state,strict=False)
    for index in range(2,len(model.m.layers)):
        model.m.layers[index]=IdentityGrowthLayer(model.m.layers[index],model.m.embed.embedding_dim)


class Race(nn.Module):
    def __init__(self, pos_gaps, d, n_ln, layers=2):
        super().__init__()
        scale = float(np.median(pos_gaps)); qs = np.log(np.quantile(pos_gaps, np.linspace(0.05, 0.95, n_ln))).tolist()
        self.m = RaceTPP(K, d, 16, layers, 1, n_ln, 4, 0.1, scale, qs, CELL)

    def terms(self, t, m):
        time,mark=self.component_terms(t,m)
        return time+mark

    def component_terms(self, t, m):
        mask = torch.ones_like(m, dtype=torch.bool)
        h, slots = self.m.encode(t, m, mask); p = self.m.clocks(h[:, :-1], slots[:, :-1])
        g = (t[:, 1:] - t[:, :-1]).clamp_min(0)
        _, ls0 = self.m.clock_terms(g, *p[:4]); _, ls1 = self.m.clock_terms(g + CELL, *p[:4])
        lh, _ = self.m.clock_terms(g + CELL / 2, *p[:4])
        log_pk = p[4].gather(-1, m[:, 1:, None, None].expand(-1, -1, self.m.M, 1)).squeeze(-1)
        log_pi = torch.logsumexp(lh + log_pk, -1) - torch.logsumexp(lh, -1)
        return log_interval(ls0.sum(-1), ls1.sum(-1)),log_pi


def batches(ws, bs, rng=None):
    idx = np.arange(len(ws)) if rng is None else rng.permutation(len(ws))
    for b in range(0, len(idx), bs):
        sel = [ws[i] for i in idx[b:b + bs]]
        yield torch.tensor(np.stack([s[0] for s in sel])), torch.tensor(np.stack([s[1] for s in sel]))


@torch.no_grad()
def score(model, ws, bs):
    model.eval(); tot, n = 0.0, 0
    parts={name:dict(targets=0,time_ll_sum=0.,mark_ll_sum=0.) for name in ('tied','positive')}
    for t, m in batches(ws, bs):
        time_ll,mark_ll=model.component_terms(t,m)
        time_ll=time_ll[:,CTX-1:];mark_ll=mark_ll[:,CTX-1:]
        lp=time_ll+mark_ll
        tot += float(lp.sum()); n += lp.numel()
        tied=(t[:,1:]-t[:,:-1])[:,CTX-1:]==0
        for name,mask in [('tied',tied),('positive',~tied)]:
            parts[name]['targets']+=int(mask.sum())
            parts[name]['time_ll_sum']+=float(time_ll[mask].sum())
            parts[name]['mark_ll_sum']+=float(mark_ll[mask].sum())
    model.validation_components=parts
    assert sum(row['targets'] for row in parts.values())==n
    assert math.isclose(sum(row['time_ll_sum']+row['mark_ll_sum'] for row in parts.values()),tot,abs_tol=1e-7,rel_tol=1e-12)
    return tot / n


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--tag', required=True)
    ap.add_argument('--model', choices=('poisson', 'hawkes', 'race'), required=True)
    ap.add_argument('--seed', type=int, default=0); ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--patience', type=int, default=5); ap.add_argument('--lr', type=float, default=3e-3)
    ap.add_argument('--bs', type=int, default=16); ap.add_argument('--d', type=int, default=32)
    ap.add_argument('--layers',type=int,default=8); ap.add_argument('--smoke-updates',type=int,default=0)
    ap.add_argument('--resume'); ap.add_argument('--grow-from')
    ap.add_argument('--n-ln', type=int, default=8); ap.add_argument('--train-every', type=int, default=20)
    ap.add_argument('--max-wall-s', type=float, default=5 * 3600); ap.add_argument('--score-test', action='store_true')
    a = ap.parse_args()
    if a.score_test:raise ValueError('Development depth pipeline does not admit sealed TEST')
    if a.model!='race' or a.layers<2:raise ValueError('Native multi-layer race only')
    rd=ROOT/'experiments/results/market'
    if any((rd/f'{a.tag}{suffix}').exists() for suffix in ('.json','.pt')):
        raise FileExistsError('Never overwrite a prior run')
    torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed)
    torch.set_num_threads(1); torch.set_default_dtype(torch.float64); t0 = time.time()
    tr = [w for d in TRAIN for w in windows(d, a.train_every)]; va = [w for d in VAL for w in windows(d, 10)]
    gaps = np.concatenate([np.diff(w[0]) for w in tr]); pos = gaps[gaps > 0]
    counts = np.bincount(np.concatenate([w[1] for w in tr]), minlength=K); rates = counts / sum(w[0][-1] for w in tr)
    model = dict(poisson=lambda: Poisson(rates.tolist()), hawkes=lambda: Hawkes(rates.tolist()),
                 race=lambda: Race(pos, a.d, a.n_ln, a.layers))[a.model]()
    if a.grow_from:
        grow_from_parent(model,torch.load(a.grow_from,map_location='cpu',weights_only=True))
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4 if a.model == 'race' else 0.0)
    print(f'train windows {len(tr)} val windows {len(va)} zero-gap frac {float((gaps == 0).mean()):.3f} '
          f'params {sum(p.numel() for p in model.parameters())}', flush=True)
    hist, best, state, bad = [], -math.inf, None, 0
    best_components=None
    updates=0; stage_gradients=[0.]*a.layers; train_wall_s=0.; trained_targets=0
    identity=dict(parent=hashlib.sha256(Path(a.grow_from).read_bytes()).hexdigest() if a.grow_from else None,driver=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  core=hashlib.sha256((ROOT/'experiments/tpp/race_tpp_v5.py').read_bytes()).hexdigest(),
                  data={str(day):hashlib.sha256((NPZ/f'{day}.npz').read_bytes()).hexdigest() for day in TRAIN+VAL})
    settings={k:v for k,v in vars(a).items() if k not in ('tag','resume')}
    first_epoch=0;previous_wall=0.
    if a.resume:
        ck=torch.load(a.resume,map_location='cpu',weights_only=False)
        if ck['identity']!=identity or ck['settings']!=settings:raise ValueError('Exact recovery identity/settings mismatch')
        model.load_state_dict(ck['model']);opt.load_state_dict(ck['optimizer'])
        rng.bit_generator.state=ck['numpy_rng'];torch.set_rng_state(ck['torch_rng'])
        hist,best,state,bad,first_epoch,updates,stage_gradients,train_wall_s,trained_targets,previous_wall=[ck[k] for k in
            ('history','best','best_state','bad','next_epoch','updates','stage_gradients','training_wall_s','training_targets','elapsed_wall_s')]
        best_components=ck['best_components']
        if first_epoch >= (1 if a.smoke_updates else a.epochs):raise ValueError('Recovery checkpoint already reached budget')
    reason='epoch_budget'
    for ep in range(first_epoch,1 if a.smoke_updates else a.epochs):
        model.train(); tl, tn = 0.0, 0
        for t, m in batches(tr, a.bs, rng):
            if a.smoke_updates and updates>=a.smoke_updates:break
            began=time.monotonic()
            lp = model.terms(t, m); loss = -lp.mean()
            if not bool(torch.isfinite(loss)):raise FloatingPointError('Nonfinite market fit')
            opt.zero_grad(); loss.backward()
            for index,layer in enumerate(model.m.layers):
                norm=math.sqrt(sum(float(p.grad.square().sum()) for p in layer.parameters() if p.grad is not None))
                if not math.isfinite(norm):raise FloatingPointError('Nonfinite stage gradient')
                stage_gradients[index]+=norm
            gradient=float(nn.utils.clip_grad_norm_(model.parameters(), 1.0))
            if not math.isfinite(gradient):raise FloatingPointError('Nonfinite full gradient')
            opt.step();updates+=1;trained_targets+=lp.numel();train_wall_s+=time.monotonic()-began
            tl += float(lp.sum()); tn += lp.numel()
        v = score(model, va[:2] if a.smoke_updates else va, 32)
        if not math.isfinite(v):raise FloatingPointError('Nonfinite DEV score')
        hist.append(dict(epoch=ep, train_ll=tl / tn, val_ll=v, elapsed_s=time.time() - t0))
        print(json.dumps(hist[-1]), flush=True)
        if v > best:
            best, state, bad = v, {k: x.clone() for k, x in model.state_dict().items()}, 0
            best_components=model.validation_components
        else:
            bad += 1
        ck=dict(best_components=best_components,identity=identity,settings=settings,model=model.state_dict(),optimizer=opt.state_dict(),
                numpy_rng=rng.bit_generator.state,torch_rng=torch.get_rng_state(),history=hist,best=best,best_state=state,
                bad=bad,next_epoch=ep+1,updates=updates,stage_gradients=stage_gradients,training_wall_s=train_wall_s,
                training_targets=trained_targets,elapsed_wall_s=previous_wall+time.time()-t0)
        rd.mkdir(parents=True,exist_ok=True);temporary=rd/f'{a.tag}.resume.tmp'
        torch.save(ck,temporary);temporary.replace(rd/f'{a.tag}.resume.pt')
        if bad >= a.patience or previous_wall+time.time() - t0 > a.max_wall_s:
            reason='early_stopping' if bad>=a.patience else 'wall_limit'
            break
    model.load_state_dict(state)
    out = dict(status='completed', battle='B10', tag=a.tag, model=a.model, args=vars(a), parameters=sum(p.numel() for p in model.parameters()),
               train_windows=len(tr), val_windows=len(va), best_val_ll_per_event=best, history=hist,
               best_validation_components=best_components,
               evidence='numerical/resource smoke' if a.smoke_updates else 'development single seed',
               completion_reason=reason,updates=updates,training_targets=trained_targets,training_wall_s=train_wall_s,
               val_scored_windows=min(2,len(va)) if a.smoke_updates else len(va),
               stage_gradient_norm_totals=stage_gradients,
               identity=identity,max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
               memory_stages=a.layers,mark_memory_slots=K,latent_clocks=model.m.M,
               core_source_sha256=hashlib.sha256((ROOT/'experiments/tpp/race_tpp_v5.py').read_bytes()).hexdigest(),
               mechanism_scope='Stacked temporal decay/rotation memories and addressed mark writes; exact clock-race likelihood teaches all clocks. Full BPTT/all-clock scoring, no hard sparse hidden-layer routing or learned backward-credit replacement.',
               test='not scored (development run)', source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    if a.score_test:
        per_day = {}
        for d in TEST:
            per_day[str(d)] = score(model, windows(d, 1), 32)
        out['test'] = dict(per_day=per_day, mean=float(np.mean(list(per_day.values()))))
    out['wall_s'] = time.time() - t0
    rd = ROOT / 'experiments/results/market'; rd.mkdir(parents=True, exist_ok=True)
    torch.save(state, rd / f'{a.tag}.pt'); (rd / f'{a.tag}.json').write_text(json.dumps(out, indent=1) + '\n')
    print('RESULT', json.dumps(dict(best_val=best, test=out['test'], wall_s=out['wall_s'])))


if __name__ == '__main__':
    main()
