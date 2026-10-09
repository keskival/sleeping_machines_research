#!/usr/bin/env python3
"""v2 (9 Oct, founder direction: forward and backward directions need not share weights, but they must train each
other). The backward path through the readout and clock head (the only place the trace learner uses transposed weights)
gets its OWN feedback matrices B, separate from the forward W:
  online_kp     the two directions train each other: B receives exactly the update W receives (the same local
                pre/post-activity product, Kolen & Pollack 1994; Akrout et al. 2019) with the same decoupled decay, so
                W - B shrinks by (1 - lr wd) per step: duality is learned, not assumed. No weight transport anywhere.
  online_fa     static unshared feedback: fixed random B (feedback alignment, Lillicrap et al. 2016)
  online_trace  shared (exact transposes), as v1
Cosine alignment between each W and its B is logged every epoch. The memory's own credit runs forward (traces use the
forward weights in the forward direction), so with learned feedback the whole learner is local.

Theory note 160, prediction 4: learning from each event as it arrives, without backpropagation through time.

A race-of-clocks temporal point process over one complex-diagonal temporal memory is trained in three ways on EasyTPP data:
  bptt          reference: full-sequence backpropagation through time, batches of sequences, one update per batch
  online_trace  PROPOSED: per-event updates. Forward eligibility traces S_t = a_t S_(t-1) + db_t/dtheta (note 160 sect. 1,
                the exact dual of the time-reversed adjoint) carry every memory parameter's sensitivity forward; when event
                t+1 arrives its loss is differentiated locally (head, readout) and the memory credit lambda_t = dl/dz_t is
                applied to the traces. No stored history, no backward sweep, no sequence-level synchrony.
  online_local  ablation: per-event updates with the traces dropped (instantaneous gradient only; e-prop-like truncation)
For fixed weights the trace gradient equals BPTT exactly (contract: --contract). With weights changing every event it is
real-time recurrent learning, exact up to the weight drift. --streams parallel sequences share each per-event update
(an average over concurrent events, not over a sequence).

Model: u_t = E[m_t] + G f(dt_t); memory z_t = a_t z_(t-1) + (W_r u + i W_i u) sigmoid(V u), a_t = exp((-r + i w) dt_t);
h_t = u_t + R [Re z, Im z]; clocks from h_t: exponential clocks and defective log-normal clocks with per-clock mark
distributions; log-likelihood log lambda_k(tau) - sum_i H_i(tau) (exact compensator). Scored events 2..N (EasyTPP), DEV only.
Run under run_safe.
"""
import argparse, hashlib, json, math, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'experiments/tpp'))
from race_tpp_v8 import load_split, batches  # noqa: E402
LOG2PI = math.log(2 * math.pi)
class FBLinear(torch.autograd.Function):
    """y = x W^T forward; the backward pass sends the error through its own matrix B (B = W means exact backprop)."""
    @staticmethod
    def forward(ctx, x, W, B):
        ctx.save_for_backward(x, B)
        return x @ W.T

    @staticmethod
    def backward(ctx, g):
        x, B = ctx.saved_tensors
        gW = g.reshape(-1, g.shape[-1]).T @ x.reshape(-1, x.shape[-1])
        return g @ B, gW, None


FB = ('R', 'P', 'H', 'Mk')                          # readout/head matrices whose transposes carry credit


def lin(p, x, name):
    fb = p.get('_fb_' + name)
    return FBLinear.apply(x, p[name], fb) if fb is not None else x @ p[name].T


MEM = ('Wr', 'br', 'Wi', 'bi', 'V', 'bv', 'E', 'Gw', 'Gb', 'log_rate', 'freq')   # parameters that reach the memory


def init_params(K, d, n, ne, nl, log_q, dtype, gen):
    r = lambda *s, sc=1.0: (torch.randn(*s, generator=gen, dtype=dtype) * sc)
    M = ne + nl
    p = dict(E=r(K, d, sc=0.3), Gw=r(d, 2, sc=0.3), Gb=torch.zeros(d, dtype=dtype),
             Wr=r(n, d, sc=d ** -0.5), br=torch.zeros(n, dtype=dtype), Wi=r(n, d, sc=d ** -0.5), bi=torch.zeros(n, dtype=dtype),
             V=r(n, d, sc=d ** -0.5), bv=torch.zeros(n, dtype=dtype),
             log_rate=torch.logspace(-2, 1, n, dtype=dtype).log(), freq=r(n, sc=0.5),
             R=r(d, 2 * n, sc=(2 * n) ** -0.5), Rb=torch.zeros(d, dtype=dtype),
             P=r(d, d, sc=d ** -0.5), Pb=torch.zeros(d, dtype=dtype),
             H=r(ne + 3 * nl, d, sc=0.01), Hb=torch.zeros(ne + 3 * nl, dtype=dtype),
             Mk=r(M * K, d, sc=0.01), Mb=torch.zeros(M * K, dtype=dtype))
    p['Hb'][ne:ne + nl] = torch.as_tensor(log_q, dtype=dtype); p['Hb'][ne + nl:ne + 2 * nl] = -0.5
    for v in p.values():
        v.requires_grad_(True)
    return p


def embed(p, m, dt):
    f = torch.stack([torch.log1p(dt), (dt == 0).to(dt.dtype)], -1)
    return p['E'][m] + f @ p['Gw'].T + p['Gb'], f


def write(p, u):
    wr = u @ p['Wr'].T + p['br']; wi = u @ p['Wi'].T + p['bi']; g = torch.sigmoid(u @ p['V'].T + p['bv'])
    return wr * g, wi * g, (wr, wi, g)


def decay(p, dt):
    r = p['log_rate'].exp(); e = torch.exp(-r * dt[:, None]); ang = p['freq'] * dt[:, None]
    return e * torch.cos(ang), e * torch.sin(ang), r


def event_ll(p, u, zr, zi, tau, mk, ne, nl, K):
    """log-likelihood (time part, mark part) of the next event (gap tau > 0, mark mk) given the state after this event."""
    h = u + lin(p, torch.cat([zr, zi], -1), 'R') + p['Rb']
    hid = F.gelu(lin(p, h, 'P') + p['Pb']); o = lin(p, hid, 'H') + p['Hb']
    lrate = o[:, :ne]; mu, lsig, lfire = o[:, ne:ne + nl], o[:, ne + nl:ne + 2 * nl], o[:, ne + 2 * nl:]
    sig = F.softplus(lsig) + 0.05; lt = torch.log(tau)[:, None]; zz = (lt - mu) / sig
    lp = F.logsigmoid(lfire); log_cdf = torch.special.log_ndtr(zz)
    log_S = torch.log1p(-torch.exp(lp + log_cdf).clamp(max=1 - 1e-12))
    log_h = torch.cat([lrate, lp - 0.5 * zz ** 2 - 0.5 * LOG2PI - torch.log(sig) - lt - log_S], -1)
    H = torch.cat([lrate.exp() * tau[:, None], -log_S], -1).sum(-1)
    lpk = F.log_softmax((lin(p, hid, 'Mk') + p['Mb']).view(len(u), ne + nl, K), -1)
    lam = torch.logsumexp(log_h, -1)
    lam_k = torch.logsumexp(log_h + lpk.gather(-1, mk[:, None, None].expand(-1, ne + nl, 1)).squeeze(-1), -1)
    return lam - H, lam_k - lam


class Traces:
    """Forward eligibility traces dz_t/dtheta for the memory parameters (real, imaginary parts), per stream."""

    def __init__(self, B, n, d, K, dtype):
        z = lambda *s: (torch.zeros(B, n, *s, dtype=dtype), torch.zeros(B, n, *s, dtype=dtype))
        self.S = dict(Wr=z(d), br=z(), Wi=z(d), bi=z(), V=z(d), bv=z(), E=z(K, d), Gw=z(d, 2), Gb=z(d), log_rate=z(), freq=z())

    def reset(self, rows):
        for Sr, Si in self.S.values():
            Sr[rows] = 0; Si[rows] = 0

    def step(self, p, ar, ai, r, dt, zpr, zpi, u, f, m, parts):
        """S_t = a_t S_(t-1) + local derivative of (a_t z_(t-1) + b_t)."""
        with torch.no_grad():
            wr, wi, g = parts; gp = g * (1 - g)
            loc = {}
            loc['Wr'] = ((g[..., None] * u[:, None]), torch.zeros(()))
            loc['br'] = (g, torch.zeros(()))
            loc['Wi'] = (torch.zeros(()), g[..., None] * u[:, None])
            loc['bi'] = (torch.zeros(()), g)
            loc['V'] = ((wr * gp)[..., None] * u[:, None], (wi * gp)[..., None] * u[:, None])
            loc['bv'] = (wr * gp, wi * gp)
            Jr = g[..., None] * p['Wr'] + (wr * gp)[..., None] * p['V']           # db/du  [B, n, d]
            Ji = g[..., None] * p['Wi'] + (wi * gp)[..., None] * p['V']
            Er = torch.zeros_like(self.S['E'][0]); Ei = torch.zeros_like(Er); bix = torch.arange(len(m))
            Er[bix, :, m] = Jr; Ei[bix, :, m] = Ji
            loc['E'] = (Er, Ei)
            loc['Gw'] = (Jr[..., None] * f[:, None, None], Ji[..., None] * f[:, None, None])
            loc['Gb'] = (Jr, Ji)
            azr, azi = ar * zpr - ai * zpi, ar * zpi + ai * zpr                     # a z_(t-1)
            k = -r * dt[:, None]
            loc['log_rate'] = (k * azr, k * azi)                                   # d(a z)/d log r = -r dt a z
            loc['freq'] = (-dt[:, None] * azi, dt[:, None] * azr)                  # d(a z)/d w = i dt a z
            for name, (Sr, Si) in self.S.items():
                sh = (len(m), ar.shape[1]) + (1,) * (Sr.dim() - 2)
                a_r, a_i = ar.view(sh), ai.view(sh)
                nr, ni = a_r * Sr - a_i * Si, a_r * Si + a_i * Sr
                lr_, li_ = loc[name]
                self.S[name] = (nr + lr_, ni + li_)

    def grads(self, lr, li, active):
        """dl/dtheta through the memory: sum over streams and modes of Re(lambda) Re(S) + Im(lambda) Im(S)."""
        out = {}
        w = active.to(lr.dtype)[:, None]
        lr, li = lr * w, li * w
        for name, (Sr, Si) in self.S.items():
            sh = lr.shape + (1,) * (Sr.dim() - 2)
            gfull = lr.view(sh) * Sr + li.view(sh) * Si
            out[name] = gfull.sum(0) if name in ('Wr', 'br', 'Wi', 'bi', 'V', 'bv', 'log_rate', 'freq') else gfull.sum((0, 1))
        return out


def run_batch(p, t, m, mask, a, scale, mode, opt=None, tr=None):
    """One group of sequences. mode: 'bptt' (returns graph loss), 'eval', 'online_trace', 'online_local'.
    Returns (time_ll_sum, mark_ll_sum, n_scored) in original time units."""
    B, L = m.shape; dtype = t.dtype; n = a.modes
    dt = torch.diff(t, dim=1, prepend=t[:, :1]).clamp_min(0) / scale
    zr = torch.zeros(B, n, dtype=dtype); zi = torch.zeros_like(zr)
    tl_sum = ml_sum = 0.0; cnt = 0; tot = torch.zeros((), dtype=dtype)
    online = mode.startswith('online')
    if online and mode == 'online_trace':
        tr.reset(slice(None))
    u_prev = None
    for i in range(L):
        u, f = embed(p, m[:, i], dt[:, i])
        if i > 0:                                                    # score event i from the state after event i-1
            valid = mask[:, i]
            tau = dt[:, i].clamp_min(1e-9)
            if online:
                zr_l = zr.detach().requires_grad_(True); zi_l = zi.detach().requires_grad_(True)
                up, _ = embed(p, m[:, i - 1], dt[:, i - 1])
                tl, ml = event_ll(p, up, zr_l, zi_l, tau, m[:, i], a.n_exp, a.n_ln, a.K)
                lsum = -((tl + ml) * valid).sum(); nv = int(valid.sum())
                if nv:
                    opt.zero_grad()
                    lsum.backward()
                    if mode == 'online_trace':
                        g = tr.grads(zr_l.grad, zi_l.grad, valid)
                        for k_, v in g.items():
                            p[k_].grad = p[k_].grad + v if p[k_].grad is not None else v.clone()
                    for k_, v in p.items():
                        if v.grad is not None and not k_.startswith('_fb_'):
                            v.grad /= nv
                    if '_fb_R' in p and p['_fb_R'].requires_grad:
                        for nm in FB:                     # the backward direction learns from the forward's own update
                            p['_fb_' + nm].grad = p[nm].grad.clone()
                    opt.step()
            else:
                tl, ml = event_ll(p, u_prev, zr, zi, tau, m[:, i], a.n_exp, a.n_ln, a.K)
                if mode == 'bptt':
                    tot = tot - ((tl + ml) * valid).sum()
            with torch.no_grad():
                tl_sum += float(((tl - math.log(scale)) * valid).sum()); ml_sum += float((ml * valid).sum()); cnt += int(valid.sum())
        # advance the memory with event i (weights as they are now)
        ctx = torch.no_grad() if (online or mode == 'eval') else torch.enable_grad()
        with ctx:
            if online:
                u, f = embed(p, m[:, i], dt[:, i])
            br_, bi_, parts = write(p, u)
            ar, ai, r = decay(p, dt[:, i])
            if mode == 'online_trace':
                tr.step(p, ar, ai, r, dt[:, i], zr, zi, u, f, m[:, i], parts)
            zr, zi = ar * zr - ai * zi + br_, ar * zi + ai * zr + bi_
            u_prev = u
    return tl_sum, ml_sum, cnt, tot


def evaluate(p, seqs, a, scale):
    T = M = N = 0
    with torch.no_grad():
        for t, m, mask in batches(seqs, 64, False, None):
            tl, ml, c, _ = run_batch(p, t.to(DT), m, mask, a, scale, 'eval')
            T += tl; M += ml; N += c
    return T / N, M / N, (T + M) / N


DT = torch.float64


def contract(a):
    """Fixed weights: summed per-event trace gradients == BPTT autograd gradient."""
    gen = torch.Generator().manual_seed(0); K = 5; a.K = K
    p = init_params(K, 8, a.modes, a.n_exp, a.n_ln, [0.0] * a.n_ln, DT, gen)
    with torch.no_grad():
        for v in p.values():
            v.add_(torch.randn(v.shape, generator=gen, dtype=DT) * 0.1)
    B, L = 3, 12
    t = torch.cumsum(torch.rand(B, L, generator=gen, dtype=DT) * 2 + 0.01, 1); m = torch.randint(0, K, (B, L), generator=gen)
    mask = torch.ones(B, L, dtype=torch.bool); mask[0, 9:] = False
    for v in p.values():
        v.grad = None
    _, _, nv, tot = run_batch(p, t, m, mask, a, 1.0, 'bptt'); tot.backward()
    ref = {k: v.grad.clone() for k, v in p.items()}

    class Acc:                                                     # accumulate instead of stepping: weights stay fixed
        def __init__(s): s.acc = {k: torch.zeros_like(v) for k, v in p.items()}
        def zero_grad(s):
            for v in p.values():
                v.grad = None
        def step(s):
            pass
    acc = Acc(); tr = Traces(B, a.modes, 8, K, DT)
    orig_step = acc.step

    def step():
        for k, v in p.items():
            if v.grad is not None:
                acc.acc[k] += v.grad * cur['nv']
    acc.step = step; cur = {}
    # run online with per-step valid counts recorded (gradients are divided by nv before step)
    dt = torch.diff(t, dim=1, prepend=t[:, :1]).clamp_min(0)
    cur_nv = [int(mask[:, i].sum()) for i in range(L)]
    i_ref = {'i': 0}

    def step_wrap():
        i_ref['i'] += 1
        cur['nv'] = cur_nv[i_ref['i']]
        step()
    acc.step = step_wrap
    run_batch(p, t, m, mask, a, 1.0, 'online_trace', opt=acc, tr=tr)
    err = max(((acc.acc[k] - ref[k]).abs().max() / ref[k].abs().max().clamp_min(1e-30)).item() for k in p)
    worst = max(p, key=lambda k: ((acc.acc[k] - ref[k]).abs().max() / ref[k].abs().max().clamp_min(1e-30)).item())
    print(f'contract: max relative difference trace vs BPTT over all parameters {err:.2e} (worst {worst})',
          'PASS' if err < 1e-9 else 'FAIL')
    return err


def add_feedback(p, arm, seed, equal=False):
    """Separate backward matrices for the readout/head (unshared with the forward weights)."""
    if arm not in ('online_kp', 'online_fa'):
        return
    g = torch.Generator().manual_seed(10_000 + seed)
    for nm in FB:
        W = p[nm].detach()
        B = W.clone() if equal else torch.randn(W.shape, generator=g, dtype=W.dtype) * W.std().clamp_min(1e-2)
        p['_fb_' + nm] = B.requires_grad_(arm == 'online_kp')


def make_opt(p, a):
    """AdamW: identical decay on the readout/head matrices and (KP) their feedback, so W - B shrinks every step."""
    decayed = [p[nm] for nm in FB] + ([p['_fb_' + nm] for nm in FB] if a.arm == 'online_kp' else [])
    ids = {id(v) for v in decayed}
    rest = [v for k, v in p.items() if id(v) not in ids and not k.startswith('_fb_')]
    return torch.optim.AdamW([dict(params=decayed, weight_decay=a.wd), dict(params=rest, weight_decay=0.0)], lr=a.lr)


def alignment(p):
    """cosine between each forward matrix and its feedback matrix (1 = exact duality)."""
    return {nm: float(F.cosine_similarity(p[nm].detach().flatten(), p['_fb_' + nm].detach().flatten(), 0))
            for nm in FB if '_fb_' + nm in p}


def kp_contract(a):
    """Identical weights and data: exact transposes vs KP feedback initialized at B = W. KP must keep B = W and
    reproduce every parameter of the exact learner (identical gradients, identical AdamW steps)."""
    gen = torch.Generator().manual_seed(0); K = 5; a.K = K; d = 8; B_, L = 3, 10
    t = torch.cumsum(torch.rand(B_, L, generator=gen, dtype=DT) * 2 + 0.01, 1); m = torch.randint(0, K, (B_, L), generator=gen)
    mask = torch.ones(B_, L, dtype=torch.bool); runs = []
    for arm in ('online_trace', 'online_kp'):
        p = init_params(K, d, a.modes, a.n_exp, a.n_ln, [0.0] * a.n_ln, DT, torch.Generator().manual_seed(1))
        a.arm = arm; add_feedback(p, arm, 0, equal=True); opt = make_opt(p, a)
        run_batch(p, t, m, mask, a, 1.0, 'online_trace', opt=opt, tr=Traces(B_, a.modes, d, K, DT))
        runs.append(p)
    moved = max((runs[0][k] - init_params(K, d, a.modes, a.n_exp, a.n_ln, [0.0] * a.n_ln, DT, torch.Generator().manual_seed(1))[k]).abs().max().item() for k in runs[0])
    err = max((runs[0][k] - runs[1][k]).abs().max().item() for k in runs[0])
    err = max(err, max((runs[1][nm] - runs[1]['_fb_' + nm]).abs().max().item() for nm in FB))
    print(f'KP contract: weights moved {moved:.2e}; max difference to the exact learner and between W and B after {L - 1} updates: {err:.2e}',
          'PASS' if err < 1e-12 and moved > 1e-4 else 'FAIL')
    return err if moved > 1e-4 else 1.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag'); ap.add_argument('--dataset', default='taxi')
    ap.add_argument('--arm', choices=['bptt', 'online_trace', 'online_local', 'online_kp', 'online_fa'], default='online_trace')
    ap.add_argument('--d', type=int, default=32); ap.add_argument('--modes', type=int, default=16)
    ap.add_argument('--n-exp', type=int, default=4); ap.add_argument('--n-ln', type=int, default=8)
    ap.add_argument('--epochs', type=int, default=20); ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--batch', type=int, default=16, help='sequences per group (bptt: per update; online: parallel streams)')
    ap.add_argument('--seed', type=int, default=0); ap.add_argument('--contract', action='store_true')
    ap.add_argument('--wd', type=float, default=0.1, help='decoupled decay on the readout/head matrices (and their feedback)')
    ap.add_argument('--kp-contract', action='store_true')
    a = ap.parse_args(); torch.set_num_threads(1)
    if a.kp_contract:
        err = kp_contract(a); od = ROOT / 'experiments/results/credit'; od.mkdir(parents=True, exist_ok=True)
        (od / f'{a.tag}.json').write_text(json.dumps(dict(status='completed' if err < 1e-12 else 'failed', tag=a.tag,
            contract='Kolen-Pollack feedback initialized at B = W stays equal to W and reproduces the exact-transpose learner',
            max_difference=err, source_sha256={str(Path(__file__).resolve().relative_to(ROOT)): hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}), indent=1) + '\n')
        return
    if a.contract:
        err = contract(a); od = ROOT / 'experiments/results/credit'; od.mkdir(parents=True, exist_ok=True)
        (od / f'{a.tag}.json').write_text(json.dumps(dict(status='completed' if err < 1e-9 else 'failed', tag=a.tag,
            contract='trace gradient == BPTT gradient, fixed weights, float64', max_relative_difference=err,
            source_sha256={str(Path(__file__).resolve().relative_to(ROOT)): hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}), indent=1) + '\n')
        return
    t0 = time.time(); torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed)
    tr_s, dev_s = load_split(a.dataset, 'train'), load_split(a.dataset, 'dev')
    a.K = int(max(max(s[1]) for s in tr_s + dev_s)) + 1
    gaps = np.concatenate([np.diff(s[0]) for s in tr_s]); pos = gaps[gaps > 0]; scale = float(np.median(pos))
    log_q = np.log(np.quantile(pos / scale, np.linspace(0.1, 0.9, a.n_ln)))
    p = init_params(a.K, a.d, a.modes, a.n_exp, a.n_ln, log_q, DT, torch.Generator().manual_seed(a.seed))
    add_feedback(p, a.arm, a.seed)
    opt = make_opt(p, a)
    tr = Traces(a.batch, a.modes, a.d, a.K, DT) if a.arm in ('online_trace', 'online_kp', 'online_fa') else None
    hist = []; best = -1e18; updates = 0
    for ep in range(a.epochs):
        e0 = time.time()
        for t, m, mask in batches(tr_s, a.batch, True, rng):
            t = t.to(DT)
            if a.arm == 'bptt':
                opt.zero_grad(); _, _, c, tot = run_batch(p, t, m, mask, a, scale, 'bptt'); (tot / max(c, 1)).backward(); opt.step(); updates += 1
            else:
                if tr is not None and len(m) != a.batch:
                    tr_b = Traces(len(m), a.modes, a.d, a.K, DT)
                else:
                    tr_b = tr
                run_batch(p, t, m, mask, a, scale, 'online_local' if a.arm == 'online_local' else 'online_trace', opt=opt, tr=tr_b); updates += m.shape[1] - 1
        tll, mll, ll = evaluate(p, dev_s, a, scale)
        hist.append(dict(epoch=ep, dev_ll=ll, dev_time_ll=tll, dev_mark_ll=mll, updates=updates, epoch_s=time.time() - e0,
                         feedback_cosine=alignment(p)))
        print(json.dumps(hist[-1]), flush=True); best = max(best, ll)
    res = dict(status='completed', battle='ENABLER (online learning, theory note 160 prediction 4)', tag=a.tag, args=vars(a),
               scale=scale, best_dev_ll=best, final_dev_ll=hist[-1]['dev_ll'], history=hist, wall_s=time.time() - t0,
               parameters=sum(v.numel() for k, v in p.items() if not k.startswith('_fb_')),
               feedback_parameters=sum(v.numel() for k, v in p.items() if k.startswith('_fb_')),
               source_sha256={str(Path(f).resolve().relative_to(ROOT)): hashlib.sha256(Path(f).read_bytes()).hexdigest()
                              for f in (__file__, ROOT / 'experiments/tpp/race_tpp_v8.py')})
    od = ROOT / 'experiments/results/credit'; od.mkdir(parents=True, exist_ok=True)
    (od / f'{a.tag}.json').write_text(json.dumps(res, indent=1) + '\n')
    print('RESULT', json.dumps({k: res[k] for k in ('best_dev_ll', 'final_dev_ll', 'wall_s')}), flush=True)


if __name__ == '__main__':
    main()
