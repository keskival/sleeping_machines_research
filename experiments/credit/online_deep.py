#!/usr/bin/env python3
"""Theory note 160 sect. 12: exact per-event learning through a DEEP (two-layer) temporal memory, without BPTT.

One diagonal memory layer has diagonal traces (each mode depends only on its own parameters). Stacking breaks that:
layer-1 parameters reach the layer-2 state through every past layer-2 input. Here every memory-path parameter P (event
embedding, gap encoder, both layers' write/gate/decay/frequency, the layer-1 readout) carries forward traces for both
layers, S_l,t = dz_l,t/dP, updated per event:
    S_1,t = a_1,t S_1,t-1 + local_1,t
    du2_t/dP = du_t/dP + R_1 S_1,t + d(R_1 zcat_1 + Rb_1)/dR_1,Rb_1
    S_2,t = a_2,t S_2,t-1 + (own-parameter local_2,t) + J_2,t du2_t/dP      (J_2 = db_2/du_2)
When event t+1 arrives, its loss is differentiated locally w.r.t. (z_2,t, u2_t) and the readout/head; the memory-path
gradient is lambda_2 . S_2,t + mu . du2_t/dP. Nothing is stored; no backward sweep. Work per event and stream is
O(n_2 |P|) for the trace update (charged and reported) -- the price of exactness across depth.

Arms: bptt (reference: full-sequence backpropagation through both layers), online_deep (exact traces above),
online_trunc (layer-local: S_2 keeps no history w.r.t. lower-layer parameters, so credit reaches them only through the
current input, as in e-prop), online_local (no traces at all). Contract (--contract): fixed weights, online_deep
gradient == BPTT. Taxi DEV log-likelihood per event, EasyTPP split. Run under run_safe.
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
DT = torch.float64


def mem_shapes(K, d, n):
    s = [('E', (K, d)), ('Gw', (d, 2)), ('Gb', (d,))]
    for l in (1, 2):
        s += [(f'Wr{l}', (n, d)), (f'br{l}', (n,)), (f'Wi{l}', (n, d)), (f'bi{l}', (n,)), (f'V{l}', (n, d)), (f'bv{l}', (n,)),
              (f'lr{l}', (n,)), (f'fr{l}', (n,))]
        if l == 1:
            s += [('R1', (d, 2 * n)), ('Rb1', (d,))]
    return s


def init_params(K, d, n, ne, nl, log_q, gen):
    r = lambda *s, sc=1.0: torch.randn(*s, generator=gen, dtype=DT) * sc
    p = {}
    for name, sh in mem_shapes(K, d, n):
        if name == 'E':
            p[name] = r(*sh, sc=0.3)
        elif name == 'Gw':
            p[name] = r(*sh, sc=0.3)
        elif name[0] in 'WV':
            p[name] = r(*sh, sc=d ** -0.5)
        elif name.startswith('lr'):
            p[name] = torch.logspace(-2, 1, n, dtype=DT).log()
        elif name.startswith('fr'):
            p[name] = r(*sh, sc=0.5)
        elif name == 'R1':
            p[name] = r(*sh, sc=(2 * n) ** -0.5)
        else:
            p[name] = torch.zeros(*sh, dtype=DT)
    M = ne + nl
    p.update(R2=r(d, 2 * n, sc=(2 * n) ** -0.5), Rb2=torch.zeros(d, dtype=DT), P=r(d, d, sc=d ** -0.5), Pb=torch.zeros(d, dtype=DT),
             H=r(ne + 3 * nl, d, sc=0.01), Hb=torch.zeros(ne + 3 * nl, dtype=DT), Mk=r(M * K, d, sc=0.01), Mb=torch.zeros(M * K, dtype=DT))
    p['Hb'][ne:ne + nl] = torch.as_tensor(log_q, dtype=DT); p['Hb'][ne + nl:ne + 2 * nl] = -0.5
    for v in p.values():
        v.requires_grad_(True)
    return p


class Layout:
    def __init__(self, K, d, n):
        self.off, o = {}, 0
        for name, sh in mem_shapes(K, d, n):
            sz = int(np.prod(sh)); self.off[name] = (o, o + sz, sh); o += sz
        self.P = o

    def view(self, T, name):
        a, b, sh = self.off[name]
        return T[..., a:b].reshape(*T.shape[:-1], *sh)


def head_ll(p, h, tau, mk, ne, nl, K):
    hid = F.gelu(h @ p['P'].T + p['Pb']); o = hid @ p['H'].T + p['Hb']
    lrate = o[:, :ne]; mu, lsig, lfire = o[:, ne:ne + nl], o[:, ne + nl:ne + 2 * nl], o[:, ne + 2 * nl:]
    sig = F.softplus(lsig) + 0.05; lt = torch.log(tau)[:, None]; zz = (lt - mu) / sig
    lp = F.logsigmoid(lfire); log_S = torch.log1p(-torch.exp(lp + torch.special.log_ndtr(zz)).clamp(max=1 - 1e-12))
    log_h = torch.cat([lrate, lp - 0.5 * zz ** 2 - 0.5 * LOG2PI - torch.log(sig) - lt - log_S], -1)
    H = torch.cat([lrate.exp() * tau[:, None], -log_S], -1).sum(-1)
    lpk = F.log_softmax((hid @ p['Mk'].T + p['Mb']).view(len(h), ne + nl, K), -1)
    lam = torch.logsumexp(log_h, -1)
    lam_k = torch.logsumexp(log_h + lpk.gather(-1, mk[:, None, None].expand(-1, ne + nl, 1)).squeeze(-1), -1)
    return lam - H, lam_k - lam


def layer(p, l, u, dt, zr, zi):
    wr = u @ p[f'Wr{l}'].T + p[f'br{l}']; wi = u @ p[f'Wi{l}'].T + p[f'bi{l}']; g = torch.sigmoid(u @ p[f'V{l}'].T + p[f'bv{l}'])
    r = p[f'lr{l}'].exp(); e = torch.exp(-r * dt[:, None]); ang = p[f'fr{l}'] * dt[:, None]
    ar, ai = e * torch.cos(ang), e * torch.sin(ang)
    azr, azi = ar * zr - ai * zi, ar * zi + ai * zr
    return azr + wr * g, azi + wi * g, dict(wr=wr, wi=wi, g=g, r=r, ar=ar, ai=ai, azr=azr, azi=azi)


def forward_step(p, m, dt, z1r, z1i, z2r, z2i):
    f = torch.stack([torch.log1p(dt), (dt == 0).to(DT)], -1)
    u = p['E'][m] + f @ p['Gw'].T + p['Gb']
    z1r, z1i, c1 = layer(p, 1, u, dt, z1r, z1i)
    u2 = u + torch.cat([z1r, z1i], -1) @ p['R1'].T + p['Rb1']
    z2r, z2i, c2 = layer(p, 2, u2, dt, z2r, z2i)
    return u, f, z1r, z1i, u2, z2r, z2i, c1, c2


class DeepTraces:
    def __init__(self, B, n, d, K, lay, trunc):
        self.L, self.n, self.d, self.K, self.trunc = lay, n, d, K, trunc
        z = lambda: torch.zeros(B, n, lay.P, dtype=DT)
        self.S1r, self.S1i, self.S2r, self.S2i = z(), z(), z(), z()
        a2 = lay.off['Wr2'][0]; self.lower = slice(0, a2)                      # parameters below layer 2

    def _own(self, p_, Lr, Li, l, u, c, dt):
        """layer l's own-parameter local derivatives, written into dense [B, n, P] (diagonal in the mode index)."""
        n = self.n; idx = torch.arange(n); gp = c['g'] * (1 - c['g'])
        def dg(T, name, val):
            v = self.L.view(T, name)
            v[:, idx, idx] = val
        dg(Lr, f'Wr{l}', c['g'][..., None] * u[:, None]); dg(Lr, f'br{l}', c['g'])
        dg(Li, f'Wi{l}', c['g'][..., None] * u[:, None]); dg(Li, f'bi{l}', c['g'])
        dg(Lr, f'V{l}', (c['wr'] * gp)[..., None] * u[:, None]); dg(Li, f'V{l}', (c['wi'] * gp)[..., None] * u[:, None])
        dg(Lr, f'bv{l}', c['wr'] * gp); dg(Li, f'bv{l}', c['wi'] * gp)
        k = -c['r'] * dt[:, None]
        dg(Lr, f'lr{l}', k * c['azr']); dg(Li, f'lr{l}', k * c['azi'])
        dg(Lr, f'fr{l}', -dt[:, None] * c['azi']); dg(Li, f'fr{l}', dt[:, None] * c['azr'])
        Jr = c['g'][..., None] * p_[f'Wr{l}'] + (c['wr'] * gp)[..., None] * p_[f'V{l}']   # db/du [B, n, d]
        Ji = c['g'][..., None] * p_[f'Wi{l}'] + (c['wi'] * gp)[..., None] * p_[f'V{l}']
        return Jr, Ji

    @torch.no_grad()
    def step(self, p, m, f, u, z1r, z1i, u2, c1, c2, dt):
        B = len(m); d = self.d
        Du = torch.zeros(B, d, self.L.P, dtype=DT); bix = torch.arange(B); j = torch.arange(d)
        vE = self.L.view(Du, 'E'); vE[bix[:, None], j[None, :], m[:, None], j[None, :]] = 1.0
        vG = self.L.view(Du, 'Gw'); vG[:, j, j, :] = f[:, None, :]
        vGb = self.L.view(Du, 'Gb'); vGb[:, j, j] = 1.0
        # layer 1
        L1r = torch.zeros(B, self.n, self.L.P, dtype=DT); L1i = torch.zeros_like(L1r)
        J1r, J1i = self._own(p, L1r, L1i, 1, u, c1, dt)
        L1r += J1r @ Du; L1i += J1i @ Du
        ar, ai = c1['ar'][..., None], c1['ai'][..., None]
        self.S1r, self.S1i = ar * self.S1r - ai * self.S1i + L1r, ar * self.S1i + ai * self.S1r + L1i
        # input of layer 2
        n = self.n; R1 = p['R1']
        Du2 = Du + R1[:, :n] @ self.S1r + R1[:, n:] @ self.S1i
        vR = self.L.view(Du2, 'R1'); vR[:, j, j, :] = torch.cat([z1r, z1i], -1)[:, None, :]
        vRb = self.L.view(Du2, 'Rb1'); vRb[:, j, j] = 1.0
        # layer 2
        L2r = torch.zeros(B, self.n, self.L.P, dtype=DT); L2i = torch.zeros_like(L2r)
        J2r, J2i = self._own(p, L2r, L2i, 2, u2, c2, dt)
        L2r += J2r @ Du2; L2i += J2i @ Du2
        ar, ai = c2['ar'][..., None], c2['ai'][..., None]
        S2r, S2i = ar * self.S2r - ai * self.S2i, ar * self.S2i + ai * self.S2r
        if self.trunc:                                    # layer-local: no history w.r.t. lower-layer parameters
            S2r[..., self.lower] = 0; S2i[..., self.lower] = 0
        self.S2r, self.S2i = S2r + L2r, S2i + L2i
        self.Du2 = Du2

    def grad(self, l2r, l2i, mu, w):
        g = ((l2r * w[:, None])[..., None] * self.S2r + (l2i * w[:, None])[..., None] * self.S2i).sum((0, 1))
        return g + ((mu * w[:, None])[..., None] * self.Du2).sum((0, 1))


def run_batch(p, t, m, mask, a, scale, mode, opt=None, lay=None):
    B, Lq = m.shape; n = a.modes
    dt = torch.diff(t, dim=1, prepend=t[:, :1]).clamp_min(0) / scale
    z1r = torch.zeros(B, n, dtype=DT); z1i = z1r.clone(); z2r = z1r.clone(); z2i = z1r.clone()
    online = mode.startswith('online'); tr = DeepTraces(B, n, a.d, a.K, lay, mode == 'online_trunc') if mode in ('online_deep', 'online_trunc') else None
    tl_sum = ml_sum = 0.0; cnt = 0; tot = torch.zeros((), dtype=DT); prev = None
    mem_names = [nm for nm, _ in mem_shapes(a.K, a.d, n)]
    for i in range(Lq):
        if i > 0:
            valid = mask[:, i]; tau = dt[:, i].clamp_min(1e-9); nv = int(valid.sum())
            if online:
                u2l = prev['u2'].detach().requires_grad_(True); zr = prev['z2r'].detach().requires_grad_(True); zi = prev['z2i'].detach().requires_grad_(True)
                h = u2l + torch.cat([zr, zi], -1) @ p['R2'].T + p['Rb2']
                tl, ml = head_ll(p, h, tau, m[:, i], a.n_exp, a.n_ln, a.K)
                if nv:
                    opt.zero_grad(); (-((tl + ml) * valid).sum()).backward()
                    if tr is not None:
                        gflat = tr.grad(zr.grad, zi.grad, u2l.grad, valid.to(DT))
                        for nm in mem_names:
                            g = lay.view(gflat, nm)
                            p[nm].grad = g.clone() if p[nm].grad is None else p[nm].grad + g
                    for v in p.values():
                        if v.grad is not None:
                            v.grad /= nv
                    opt.step()
            else:
                h = prev['u2'] + torch.cat([prev['z2r'], prev['z2i']], -1) @ p['R2'].T + p['Rb2']
                tl, ml = head_ll(p, h, tau, m[:, i], a.n_exp, a.n_ln, a.K)
                if mode == 'bptt':
                    tot = tot - ((tl + ml) * valid).sum()
            with torch.no_grad():
                tl_sum += float(((tl - math.log(scale)) * valid).sum()); ml_sum += float((ml * valid).sum()); cnt += int(valid.sum())
        ctx = torch.no_grad() if (online or mode == 'eval') else torch.enable_grad()
        with ctx:
            u, f, z1r, z1i, u2, z2r, z2i, c1, c2 = forward_step(p, m[:, i], dt[:, i], z1r, z1i, z2r, z2i)
            if tr is not None:
                tr.step(p, m[:, i], f, u, z1r, z1i, u2, c1, c2, dt[:, i])
        prev = dict(u2=u2, z2r=z2r, z2i=z2i)
    return tl_sum, ml_sum, cnt, tot


def evaluate(p, seqs, a, scale):
    T = M = N = 0
    with torch.no_grad():
        for t, m, mask in batches(seqs, 64, False, None):
            tl, ml, c, _ = run_batch(p, t.to(DT), m, mask, a, scale, 'eval')
            T += tl; M += ml; N += c
    return T / N, M / N, (T + M) / N


def contract(a):
    gen = torch.Generator().manual_seed(0); K = 5; a.K = K; a.d = 8; a.modes = 4
    p = init_params(K, a.d, a.modes, a.n_exp, a.n_ln, [0.0] * a.n_ln, gen)
    with torch.no_grad():
        for v in p.values():
            v.add_(torch.randn(v.shape, generator=gen, dtype=DT) * 0.1)
    B, Lq = 3, 10
    t = torch.cumsum(torch.rand(B, Lq, generator=gen, dtype=DT) * 2 + 0.01, 1); m = torch.randint(0, K, (B, Lq), generator=gen)
    mask = torch.ones(B, Lq, dtype=torch.bool); mask[0, 7:] = False
    for v in p.values():
        v.grad = None
    _, _, _, tot = run_batch(p, t, m, mask, a, 1.0, 'bptt'); tot.backward()
    ref = {k: v.grad.clone() for k, v in p.items()}
    lay = Layout(K, a.d, a.modes); out = {}
    for mode in ('online_deep', 'online_trunc'):
        acc = {k: torch.zeros_like(v) for k, v in p.items()}; nvs = [int(mask[:, i].sum()) for i in range(Lq)]; it = {'i': 0}

        class Opt:
            def zero_grad(s):
                for v in p.values():
                    v.grad = None

            def step(s):
                it['i'] += 1
                for k, v in p.items():
                    if v.grad is not None:
                        acc[k] += v.grad * nvs[it['i']]
        run_batch(p, t, m, mask, a, 1.0, mode, opt=Opt(), lay=lay)
        rel = {k: ((acc[k] - ref[k]).abs().max() / ref[k].abs().max().clamp_min(1e-30)).item() for k in p}
        out[mode] = dict(max_rel=max(rel.values()), worst=max(rel, key=rel.get), lower_layer_max_rel=max(rel[k] for k in ('E', 'Wr1', 'V1', 'lr1', 'R1')))
    ok = out['online_deep']['max_rel'] < 1e-9 and out['online_trunc']['lower_layer_max_rel'] > 1e-6
    print('contract:', json.dumps(out), 'PASS' if ok else 'FAIL')
    return ok, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', required=True); ap.add_argument('--dataset', default='taxi')
    ap.add_argument('--arm', choices=['bptt', 'online_deep', 'online_trunc', 'online_local'], default='online_deep')
    ap.add_argument('--d', type=int, default=32); ap.add_argument('--modes', type=int, default=16)
    ap.add_argument('--n-exp', type=int, default=4); ap.add_argument('--n-ln', type=int, default=8)
    ap.add_argument('--epochs', type=int, default=20); ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--batch', type=int, default=16); ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--contract', action='store_true')
    a = ap.parse_args(); torch.set_num_threads(1)
    od = ROOT / 'experiments/results/credit'; od.mkdir(parents=True, exist_ok=True)
    sha = {str(Path(f).resolve().relative_to(ROOT)): hashlib.sha256(Path(f).read_bytes()).hexdigest()
           for f in (__file__, ROOT / 'experiments/tpp/race_tpp_v8.py')}
    if a.contract:
        ok, out = contract(a)
        (od / f'{a.tag}.json').write_text(json.dumps(dict(status='completed' if ok else 'failed', tag=a.tag, contract=out,
            statement='fixed weights: online_deep gradient == BPTT; online_trunc differs on lower-layer parameters', source_sha256=sha), indent=1) + '\n')
        return
    t0 = time.time(); torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed)
    tr_s, dev_s = load_split(a.dataset, 'train'), load_split(a.dataset, 'dev')
    a.K = int(max(max(s[1]) for s in tr_s + dev_s)) + 1
    gaps = np.concatenate([np.diff(s[0]) for s in tr_s]); pos = gaps[gaps > 0]; scale = float(np.median(pos))
    log_q = np.log(np.quantile(pos / scale, np.linspace(0.1, 0.9, a.n_ln)))
    p = init_params(a.K, a.d, a.modes, a.n_exp, a.n_ln, log_q, torch.Generator().manual_seed(a.seed))
    lay = Layout(a.K, a.d, a.modes); opt = torch.optim.Adam(p.values(), lr=a.lr)
    hist = []; best = -1e18; updates = 0
    for ep in range(a.epochs):
        e0 = time.time()
        for t, m, mask in batches(tr_s, a.batch, True, rng):
            t = t.to(DT)
            if a.arm == 'bptt':
                opt.zero_grad(); _, _, c, tot = run_batch(p, t, m, mask, a, scale, 'bptt'); (tot / max(c, 1)).backward(); opt.step(); updates += 1
            else:
                run_batch(p, t, m, mask, a, scale, a.arm, opt=opt, lay=lay); updates += m.shape[1] - 1
        tll, mll, ll = evaluate(p, dev_s, a, scale)
        hist.append(dict(epoch=ep, dev_ll=ll, dev_time_ll=tll, dev_mark_ll=mll, updates=updates, epoch_s=time.time() - e0))
        print(json.dumps(hist[-1]), flush=True); best = max(best, ll)
    res = dict(status='completed', battle='ENABLER (deep online learning, theory note 160 sect. 12)', tag=a.tag, args=vars(a),
               scale=scale, best_dev_ll=best, final_dev_ll=hist[-1]['dev_ll'], history=hist, wall_s=time.time() - t0,
               parameters=sum(v.numel() for v in p.values()), memory_path_parameters=lay.P,
               trace_floats_per_stream=4 * a.modes * lay.P, source_sha256=sha)
    (od / f'{a.tag}.json').write_text(json.dumps(res, indent=1) + '\n')
    print('RESULT', json.dumps({k: res[k] for k in ('best_dev_ll', 'final_dev_ll', 'wall_s')}), flush=True)


if __name__ == '__main__':
    main()
