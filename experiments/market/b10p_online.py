"""B10-P phase 2: online price forecasting on a continuous stream with delayed label revelation.

Every arm issues, for each trade i, the next-trade tape law and 10/60/300 s return distributions from state built only
from trades <= i; the issued distributions are scored immediately against the realized tick cells. After every 32 trades
(current time = latest revealed trade), each arm takes one AdamW step on (a) the revealed tape terms of the block and
(b) the return likelihood of every earlier forecast whose horizon t_i + H has passed, using the head inputs stored when
that forecast was issued. Numerical state persists across updates (Transformer: rolling re-encoded history).
"""
import argparse, hashlib, json, math, resource, sys, time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from scipy.special import stdtr
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT/'experiments/market'))
from b10p_data import day_arrays, HORIZONS, NFEAT
from b10p_models import PriceRace, log_cell_mixture, COMPONENTS
from b10_online_stream import detach, revealed_terms

sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
SOURCES = ['experiments/market/b10p_online.py', 'experiments/market/b10p_models.py', 'experiments/market/b10p_data.py', 'experiments/market/b10_online_stream.py',
           'experiments/market/b10_matched_models.py', 'experiments/market/b10_depth_tpp_v1.py', 'experiments/market/b10_stable_clock_head.py', 'experiments/tpp/race_tpp_v5.py']
CHUNK = 32; CONTEXT = 2048; LR = 1e-4


def forecast(model, t, m, f, state):
    """b10_online_stream.forecast with the causal price-feature input; also returns the return-head inputs."""
    st = {} if state is None else state; core = model.m
    previous = st.get('last_time', t[:, :1]); dt = torch.diff(torch.cat((previous, t), 1), dim=1)/core.scale
    x = core.embed(m) + model.feat(f)
    if model.family == 'native':
        x = x + core.gap(torch.stack((dt.log1p(), (dt > 0).to(dt.dtype)), -1)); new = []
        for depth, layer in enumerate(core.layers):
            z = st.get('layers', [None]*len(core.layers))[depth]
            if z is None: z = x.new_zeros(1, 2*layer.n)
            writes = layer.write(x); gates = layer.gate(x).sigmoid(); decay = torch.exp(-layer.log_rate.exp()*dt[:, :, None]); angle = layer.freq*dt[:, :, None]
            cr = decay*angle.cos(); ci = decay*angle.sin(); out = []
            for i in range(x.shape[1]):
                r, j = z.chunk(2, -1); w = writes[:, i]*gates[:, i].repeat(1, 2)
                z = torch.cat((cr[:, i]*r - ci[:, i]*j + w[:, :layer.n], ci[:, i]*r + cr[:, i]*j + w[:, layer.n:]), -1); out.append(z)
            values = torch.stack(out, 1); new.append(z); x = layer.norm1(x + layer.drop(layer.read(values))); x = layer.norm2(x + layer.drop(layer.mlp(x)))
        st = {**st, 'layers': new}
    elif model.family == 'lstm':
        x = x + core.gap(dt.log1p()[:, :, None]); new = []
        for depth, (a, b) in enumerate(zip(model.encoder.input, model.encoder.hidden)):
            h, c = st.get('lstm', [(None, None)]*2)[depth]
            if h is None: h = x.new_zeros(1, 32); c = torch.zeros_like(h)
            projected = a(x); out = []
            for i in range(x.shape[1]):
                ig, fg, cg, og = (projected[:, i] + b(h)).chunk(4, -1); c = fg.sigmoid()*c + ig.sigmoid()*cg.tanh(); h = og.sigmoid()*c.tanh(); out.append(h)
            x = torch.stack(out, 1); new.append((h, c))
        st = {**st, 'lstm': new}
    else:
        origin = st.get('origin', t[:, :1]); count = st.get('count', 0); positions = torch.arange(count, count + t.shape[1], dtype=t.dtype)[None]
        ht = torch.cat((st.get('history_t', t[:, :0]), t), 1); hm = torch.cat((st.get('history_m', m[:, :0]), m), 1); hd = torch.cat((st.get('history_dt', dt[:, :0]), dt), 1)
        hp = torch.cat((st.get('history_position', positions[:, :0]), positions), 1); hf = torch.cat((st.get('history_f', f[:, :0]), f), 1)
        x = core.embed(hm) + model.feat(hf) + core.gap(hd.log1p()[:, :, None]); frequency = torch.exp(torch.arange(8, dtype=t.dtype)*(-math.log(10000)/8))
        def pe(z):
            a = z[:, :, None]*frequency; return torch.stack((a.sin(), a.cos()), -1).flatten(-2)
        x = x + torch.cat((pe((ht - origin)/core.scale), pe(hp)), -1); x = model.encoder(x)[:, -t.shape[1]:]
        st = {**st, 'origin': origin, 'count': count + t.shape[1], 'history_t': ht[:, -CONTEXT:], 'history_m': hm[:, -CONTEXT:], 'history_dt': hd[:, -CONTEXT:],
              'history_position': hp[:, -CONTEXT:], 'history_f': hf[:, -CONTEXT:]}
    slots = st.get('slots', x.new_zeros(1, core.K, core.marks.dv)); values = F.softplus(core.marks.value(x)); decay = torch.exp(-core.marks.log_rate.exp()*dt[:, :, None])
    onehot = F.one_hot(m, core.K).to(x.dtype); out = []
    for i in range(t.shape[1]):
        slots = slots*decay[:, i, None, :] + onehot[:, i, :, None]*values[:, i, None, :]; out.append(slots)
    slot_seq = torch.stack(out, 1)
    return core.clocks(x, slot_seq), torch.cat([x, slot_seq.flatten(-2)], -1), {**st, 'slots': slots, 'last_time': t[:, -1:]}


def price_ll(model, head_in, lo, hi):
    raw = model.price(head_in).view(*head_in.shape[:-1], len(HORIZONS), COMPONENTS, 3)
    return log_cell_mixture(lo, hi, raw[..., 0], raw[..., 1], raw[..., 2])


class LinearT(torch.nn.Module):
    """Classical reference: Student-t with location and log-scale linear in the seven causal inputs (online-adapted)."""
    def __init__(self, params):
        super().__init__()
        self.loc = torch.nn.Parameter(torch.tensor([p['loc'] for p in params])); self.scale = torch.nn.Parameter(torch.tensor([p['scale'] for p in params]))
        self.lognu = torch.nn.Parameter(torch.log(torch.tensor([p['nu'] for p in params]) - 2.0))
    def params(self, f):
        X = torch.cat([torch.ones(*f.shape[:-1], 1), f], -1); return X @ self.loc.T, (X @ self.scale.T).exp(), self.lognu.exp() + 2.0
    def train_ll(self, f, lo, hi):
        mu, s, nu = self.params(f); return torch.distributions.StudentT(nu, mu, s).log_prob(0.5*(lo + hi)) + torch.log(hi - lo)
    @torch.no_grad()
    def cell_ll(self, f, lo, hi):
        mu, s, nu = (x.numpy() for x in self.params(f)); a = (lo.numpy() - mu)/s; b = (hi.numpy() - mu)/s; up = a > 0
        p = np.where(up, stdtr(nu, -a) - stdtr(nu, -b), stdtr(nu, b) - stdtr(nu, a)); return torch.tensor(np.log(np.clip(p, 1e-300, None)))


def stream(day, moments, start, count):
    a = day_arrays(day); s = slice(start, start + count)
    f = (a['feats'][s] - np.array(moments['mean']))/np.array(moments['std'])
    return dict(t=torch.tensor(a['t'][s])[None], m=torch.tensor(a['mark'][s])[None], f=torch.tensor(f)[None], lo=torch.tensor(a['lo'][s])[None], hi=torch.tensor(a['hi'][s])[None])


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--tag', required=True); ap.add_argument('--arm', choices=['native', 'lstm', 'transformer', 'linear_t'], required=True)
    ap.add_argument('--fit', help='phase-1 B10-P result json (neural arms)'); ap.add_argument('--references', default='experiments/results/market/aws_b10p_references_20261015_v2.json')
    ap.add_argument('--day', default='2026-08-25'); ap.add_argument('--warmup-day', default='2026-08-24'); ap.add_argument('--targets', type=int, default=65536)
    a = ap.parse_args(); torch.set_num_threads(1); torch.set_default_dtype(torch.float64); began = time.monotonic(); inputs = {}; parity = {}
    out = ROOT/f'experiments/results/market/{a.tag}.json'; assert not out.exists()
    if a.arm == 'linear_t':
        ref = json.loads((ROOT/a.references).read_text()); inputs[a.references] = sha(ROOT/a.references); moments = ref['feature_moments']
        model = LinearT([ref['references']['linear_features'][f'{int(h)}s']['params'] for h in HORIZONS])
    else:
        fit = json.loads((ROOT/a.fit).read_text()); assert fit['status'] == 'completed' and not fit['smoke'] and fit['family'] == a.arm; inputs[a.fit] = sha(ROOT/a.fit)
        saved = torch.load(ROOT/fit['final_weights'], weights_only=False); inputs[fit['final_weights']] = sha(ROOT/fit['final_weights']); moments = saved['moments']
        torch.manual_seed(0); model = PriceRace(torch.ones(10), a.arm, saved['scales']); model.load_state_dict(saved['model']); model.eval()
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    if a.arm != 'linear_t':                                         # gate: streamed forecasts equal the phase-1 window law
        g = stream(a.warmup_day, moments, 0, 65)
        with torch.no_grad():
            tl, ml, pl, _ = model.all_terms(g['t'], g['m'], g['f'], torch.nan_to_num(g['lo'], nan=0.), torch.nan_to_num(g['hi'], nan=1.))
            p1, h1, s1 = forecast(model, g['t'][:, :32], g['m'][:, :32], g['f'][:, :32], None); p2, h2, _ = forecast(model, g['t'][:, 32:64], g['m'][:, 32:64], g['f'][:, 32:64], s1)
            tape = torch.cat([revealed_terms(model, p1, g['t'][:, :32], g['t'][:, 1:33], g['m'][:, 1:33]), revealed_terms(model, p2, g['t'][:, 32:64], g['t'][:, 33:65], g['m'][:, 33:65])], 1)
            ret = price_ll(model, torch.cat([h1, h2], 1), torch.nan_to_num(g['lo'][:, :64], nan=0.), torch.nan_to_num(g['hi'][:, :64], nan=1.))
        gate_error = max(float((tape - (tl + ml)[:, :64]).abs().max()), float((ret - pl[:, :64]).abs().max())); assert gate_error < 1e-8, gate_error
        parity['stream_parity_max_error'] = gate_error
    warm = stream(a.warmup_day, moments, 0, 2048 + 1); dev = stream(a.day, moments, 0, a.targets + 128 + 1)
    width = NFEAT if a.arm == 'linear_t' else model.price[0].in_features
    def empty(): return dict(t=torch.zeros(0), x=torch.zeros(0, width), lo=torch.zeros(0, len(HORIZONS)), hi=torch.zeros(0, len(HORIZONS)))
    state = None; pending = empty()                                 # issued forecasts awaiting horizon maturity
    tape_sum = 0.; tape_n = 0; ret_sum = np.zeros(len(HORIZONS)); ret_n = np.zeros(len(HORIZONS)); curve = []; updates = 0; started = None

    def step(s, lo_i, cursor, scored):
        nonlocal state, tape_sum, tape_n, updates
        t, m, f = s['t'][:, cursor:cursor+CHUNK], s['m'][:, cursor:cursor+CHUNK], s['f'][:, cursor:cursor+CHUNK]
        lo, hi = s['lo'][:, cursor:cursor+CHUNK], s['hi'][:, cursor:cursor+CHUNK]; now = float(s['t'][0, cursor+CHUNK])
        opt.zero_grad(set_to_none=True); loss = torch.zeros(())
        if a.arm == 'linear_t':
            issued = f; tape = None
            if scored:
                ll = model.cell_ll(f, lo, hi)
        else:
            params, head_in, state = forecast(model, t, m, f, state)
            tape = revealed_terms(model, params, t, s['t'][:, cursor+1:cursor+CHUNK+1], s['m'][:, cursor+1:cursor+CHUNK+1]); loss = loss - tape.mean()
            issued = head_in.detach()
            if scored:
                with torch.no_grad(): ll = price_ll(model, issued, torch.nan_to_num(lo, nan=0.), torch.nan_to_num(hi, nan=1.))
        if scored:
            valid = torch.isfinite(lo); ret_sum[:] -= torch.where(valid, ll, torch.zeros_like(ll)).sum((0, 1)).numpy(); ret_n[:] += valid.sum((0, 1)).numpy()
            if tape is not None: tape_sum -= float(tape.detach().sum()); tape_n += tape.numel()
        P = pending
        P['t'] = torch.cat([P['t'], t[0]]); P['x'] = torch.cat([P['x'], issued[0]]); P['lo'] = torch.cat([P['lo'], lo[0]]); P['hi'] = torch.cat([P['hi'], hi[0]])
        # Labels whose horizon has passed by the latest revealed trade become learnable, each exactly once.
        mask = (P['t'][:, None] + torch.tensor(HORIZONS) <= now) & torch.isfinite(P['lo'])
        rows = mask.any(1)
        if rows.any():
            X, L, Hh, M = P['x'][rows], P['lo'][rows], P['hi'][rows], mask[rows]
            if a.arm == 'linear_t': ll2 = model.train_ll(X, torch.nan_to_num(L, nan=0.), torch.nan_to_num(Hh, nan=1.))
            else: ll2 = price_ll(model, X, torch.nan_to_num(L, nan=0.), torch.nan_to_num(Hh, nan=1.))
            loss = loss - torch.where(M, ll2, torch.zeros_like(ll2)).sum()/M.sum()
            P['lo'] = torch.where(mask, torch.full_like(P['lo'], float('nan')), P['lo'])
        alive = torch.isfinite(P['lo']).any(1)
        for k in P: P[k] = P[k][alive]
        if loss.requires_grad:
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); updates += 1
        if a.arm != 'linear_t': state = detach(state)

    for cursor in range(0, 2048, CHUNK): step(warm, None, cursor, False)
    pending.update(empty())
    if a.arm != 'linear_t':                                         # context events: state only
        with torch.no_grad(): _, _, state = forecast(model, dev['t'][:, :128], dev['m'][:, :128], dev['f'][:, :128], state)
        state = detach(state)
    started = time.monotonic()
    for k, cursor in enumerate(range(128, 128 + a.targets, CHUNK)):
        step(dev, None, cursor, True)
        if (k + 1) % 128 == 0:
            curve.append(dict(targets=(k + 1)*CHUNK, tape_nll=tape_sum/max(tape_n, 1), return_nll=dict(zip([f'{int(h)}s' for h in HORIZONS], (ret_sum/ret_n).tolist())))); print(json.dumps(curve[-1]), flush=True)
    per = (ret_sum/ret_n).tolist()
    record = dict(status='completed', battle='B10', task='B10-P price online', arm=a.arm, day=a.day, targets=a.targets, tape_nll=(tape_sum/tape_n if tape_n else None),
                  return_nll=dict(zip([f'{int(h)}s' for h in HORIZONS], per)), return_nll_sum=float(sum(per)), return_targets=ret_n.astype(int).tolist(), curve=curve,
                  online_updates=updates, online_wall_s=time.monotonic() - started, process_wall_s=time.monotonic() - began, max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                  source_sha256={n: sha(ROOT/n) for n in SOURCES}, input_sha256=inputs, gates=parity, september_opened=False,
                  scope=f'Continuous {a.day} stream after a 2048-trade {a.warmup_day} warmup and 128 context trades; every arm forecasts with persistent state, is scored at issue, and adapts after every 32 trades on revealed tape terms and on matured horizon labels only. Same inputs/heads as phase 1; seed 0.')
    out.write_text(json.dumps(record, indent=2) + '\n')


if __name__ == '__main__':
    main()
