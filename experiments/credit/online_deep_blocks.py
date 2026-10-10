#!/usr/bin/env python3
"""B1 exact online3 credit with structural-zero blocks removed; forward model unchanged.

Layer 1 cannot depend on R1 or layer-2 parameters. Layer-2 own parameters
affect only their own mode; its cross-layer prefix remains fully dense.
No sensitivity is truncated or approximated. See ONLINE_DEEP_BLOCKS.md.
"""
import argparse
import hashlib
import json
import resource
import statistics
import time
from pathlib import Path
from types import SimpleNamespace

import torch
import online_deep as base

DT = base.DT
SOURCES = ['experiments/credit/online_deep_blocks.py', 'experiments/credit/online_deep.py',
           'experiments/tpp/race_tpp_v8.py']


class BlockTraces(base.DeepTraces):
    def __init__(self, B, n, d, K, lay, trunc):
        self.L, self.n, self.d, self.K, self.trunc = lay, n, d, K, trunc
        self.p1 = lay.off['R1'][0]
        self.lower_p = lay.off['Wr2'][0]
        self.q = 3*d + 5
        self.S1r = torch.zeros(B, n, self.p1, dtype=DT)
        self.S1i = torch.zeros_like(self.S1r)
        self.S2r = torch.zeros(B, n, self.lower_p, dtype=DT)
        self.S2i = torch.zeros_like(self.S2r)
        self.O2r = torch.zeros(B, n, self.q, dtype=DT)
        self.O2i = torch.zeros_like(self.O2r)

    def input_derivatives(self, T, J, m, f):
        # J @ du/dtheta: only E[m], Gw and Gb are nonzero.
        ix = torch.arange(len(m))
        self.L.view(T, 'E')[ix, :, m, :] += J
        self.L.view(T, 'Gw')[:] += J[..., None] * f[:, None, None, :]
        self.L.view(T, 'Gb')[:] += J

    @torch.no_grad()
    def step(self, p, m, f, u, z1r, z1i, u2, c1, c2, dt):
        B, d, n = len(m), self.d, self.n
        L1r = torch.zeros_like(self.S1r); L1i = torch.zeros_like(self.S1i)
        J1r, J1i = self._own(p, L1r, L1i, 1, u, c1, dt)
        self.input_derivatives(L1r, J1r, m, f)
        self.input_derivatives(L1i, J1i, m, f)
        ar, ai = c1['ar'][..., None], c1['ai'][..., None]
        self.S1r, self.S1i = (ar*self.S1r-ai*self.S1i+L1r,
                              ar*self.S1i+ai*self.S1r+L1i)
        Du2 = torch.zeros(B, d, self.lower_p, dtype=DT)
        Du2[..., :self.p1] = p['R1'][:, :n] @ self.S1r + p['R1'][:, n:] @ self.S1i
        eye = torch.eye(d, dtype=DT).expand(B, -1, -1)
        self.input_derivatives(Du2, eye, m, f)
        j = torch.arange(d)
        self.L.view(Du2, 'R1')[:, j, j, :] = torch.cat([z1r, z1i], -1)[:, None, :]
        self.L.view(Du2, 'Rb1')[:, j, j] = 1.
        self.Du2 = Du2
        gp = c2['g']*(1-c2['g'])
        J2r = c2['g'][..., None]*p['Wr2'] + (c2['wr']*gp)[..., None]*p['V2']
        J2i = c2['g'][..., None]*p['Wi2'] + (c2['wi']*gp)[..., None]*p['V2']
        ar, ai = c2['ar'][..., None], c2['ai'][..., None]
        r, i = ar*self.S2r-ai*self.S2i, ar*self.S2i+ai*self.S2r
        if self.trunc:
            r.zero_(); i.zero_()
        self.S2r, self.S2i = r + J2r@Du2, i + J2i@Du2
        zeros = torch.zeros_like(u2[:, None, :].expand(-1, n, -1))
        zg = torch.zeros_like(gp)
        g = c2['g']; ur = u2[:, None, :]
        kr = -c2['r']*dt[:, None]
        localr = torch.cat([g[..., None]*ur, g[..., None], zeros, zg[..., None],
                            (c2['wr']*gp)[..., None]*ur, (c2['wr']*gp)[..., None],
                            (kr*c2['azr'])[..., None], (-dt[:, None]*c2['azi'])[..., None]], -1)
        locali = torch.cat([zeros, zg[..., None], g[..., None]*ur, g[..., None],
                            (c2['wi']*gp)[..., None]*ur, (c2['wi']*gp)[..., None],
                            (kr*c2['azi'])[..., None], (dt[:, None]*c2['azr'])[..., None]], -1)
        self.O2r, self.O2i = (ar*self.O2r-ai*self.O2i+localr,
                              ar*self.O2i+ai*self.O2r+locali)

    def grad(self, l2r, l2i, mu, w):
        lr, li = l2r*w[:, None], l2i*w[:, None]
        g = torch.zeros(self.L.P, dtype=DT)
        g[:self.lower_p] = (lr[..., None]*self.S2r + li[..., None]*self.S2i).sum((0, 1))
        g[:self.lower_p] += ((mu*w[:, None])[..., None]*self.Du2).sum((0, 1))
        own = (lr[..., None]*self.O2r + li[..., None]*self.O2i).sum(0)
        offset = 0
        for name in ('Wr2', 'br2', 'Wi2', 'bi2', 'V2', 'bv2', 'lr2', 'fr2'):
            a, b, shape = self.L.off[name]
            width = (b-a)//self.n
            g[a:b] = own[:, offset:offset+width].reshape(-1)
            offset += width
        return g

    def stored_floats(self):
        return sum(v.numel() for v in (self.S1r, self.S1i, self.S2r, self.S2i, self.O2r, self.O2i))


def measure_shape(B, d, n, K, length, seed):
    a = SimpleNamespace(d=d, modes=n, K=K, n_exp=4, n_ln=8)
    gen = torch.Generator().manual_seed(seed)
    p = base.init_params(K, d, n, 4, 8, [0.]*8, gen)
    lay = base.Layout(K, d, n)
    dense = base.DeepTraces(B, n, d, K, lay, False)
    blocks = BlockTraces(B, n, d, K, lay, False)
    t = torch.rand(B, length, generator=gen, dtype=DT).add(.01).cumsum(1)
    m = torch.randint(K, (B, length), generator=gen)
    mask = torch.ones(B, length, dtype=torch.bool)
    mask[0, length//2:] = False
    initial = {k: v.detach().clone() for k, v in p.items()}
    outputs = {}; timings = {}
    original = base.DeepTraces
    try:
        for label, trace_type in [('dense', original), ('blocks', BlockTraces)]:
            q = {k: v.clone().requires_grad_() for k, v in initial.items()}
            opt = torch.optim.Adam(q.values(), lr=.001)
            base.DeepTraces = trace_type
            timings[label] = []
            for repeat in range(4):
                start = time.perf_counter()
                sums = base.run_batch(q, t, m, mask, a, 1., 'online_deep', opt, lay)
                if repeat: timings[label].append(time.perf_counter()-start)
            outputs[label] = (q, opt, sums[:3])
    finally:
        base.DeepTraces = original
    q0, opt0, sums0 = outputs['dense']; q1, opt1, sums1 = outputs['blocks']
    parameter_error = max(float((q0[k]-q1[k]).abs().max().detach()) for k in q0)
    state_error = max(float((opt0.state[q0[k]][field]-opt1.state[q1[k]][field]).abs().max())
                      for k in q0 for field in opt0.state[q0[k]])
    loss_error = max(abs(x-y) for x, y in zip(sums0, sums1))
    if max(parameter_error, state_error, loss_error) > 1e-9:
        raise AssertionError((parameter_error, state_error, loss_error))
    med = {k: statistics.median(v) for k, v in timings.items()}
    return dict(batch=B, d=d, modes=n, K=K, length=length,
                parameter_error=parameter_error, optimizer_state_error=state_error,
                accumulated_likelihood_error=loss_error, median_batch_s=med,
                whole_step_speedup=med['dense']/med['blocks'],
                dense_trace_floats=4*B*n*lay.P, block_trace_floats=blocks.stored_floats(),
                measured_scope='Four complete online batches; three timed after warmup; forward, credit, backward, Adam included')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tag', required=True)
    parser.add_argument('--stage', choices=['contract', 'measure'], required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    out = base.ROOT/'experiments/results/credit'/f'{args.tag}.json'
    assert not out.exists()
    start = time.monotonic()
    if args.stage == 'contract':
        # Existing every-parameter BPTT contract, with compressed traces substituted.
        original = base.DeepTraces
        try:
            base.DeepTraces = BlockTraces
            ok, fixed = base.contract(SimpleNamespace(n_exp=4, n_ln=8))
            assert ok
        finally:
            base.DeepTraces = original
        rows = [measure_shape(3, 8, 4, 5, 12, 193)]
    else:
        fixed = None
        rows = [measure_shape(B, d, n, 10, 24, 193) for B, d, n in ((1, 16, 8), (16, 32, 16), (4, 64, 32))]
    result = dict(status='completed', tag=args.tag, battle='B1/R1 exact deep-credit execution',
                  stage=args.stage, fixed_weight_contract=fixed, measurements=rows,
                  wall_s=time.monotonic()-start, max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                  source_sha256={s:hashlib.sha256((base.ROOT/s).read_bytes()).hexdigest() for s in SOURCES},
                  scope='Same two-layer online3 model and learning rule; structural zeros removed; no TEST or quality claim. '
                        'Cross-layer dense prefix remains; changing-weight drift is unchanged.')
    with out.open('x') as stream: stream.write(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
