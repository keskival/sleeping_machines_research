"""R1/B1 reciprocal credit and update gate on the unchanged native keyed model.

Correctness instrument: full/local actor gradients are BOTH computed. Audit
sampling therefore saves no teacher work here. No sparse-runtime or depth win.
One-step future derivatives are exact for this declared update; no claim about
unbounded meta-credit. Persistent block credit state and coordinate optimizer
state are distinct and time-evolve lazily when visited.
"""
import argparse
import hashlib
import json
import math
import resource
import socket
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.func import functional_call

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'experiments/tpp'))
from recall_tpp_v4 import KeyedRaceTPP, make_split

DT = torch.float64


class NativeLoss(nn.Module):
    def __init__(self, depth):
        super().__init__()
        self.model = KeyedRaceTPP(8, 8, 3, depth, 2, 2, 2, 0., 1., [-1., 0.],
                                 dk=4, local=False, prev_msg=True, qk_norm=False)

    def forward(self, t, m):
        tl, ml, n, _ = self.model.loglik(t, m, torch.ones_like(m, dtype=torch.bool))
        return -(tl + ml) / n


def native_pair(depth, seed):
    torch.manual_seed(seed)
    full = NativeLoss(depth).to(DT)
    local = NativeLoss(depth).to(DT)
    local.model.local = True
    local.load_state_dict(full.state_dict())
    return full, local


def batch(seed, size=2, pairs=3):
    t, m, _, _ = make_split(size, pairs, 4, 4, np.random.default_rng(seed))
    return t.to(DT), m


def group(name):
    parts = name.split('.')
    return '.'.join(parts[:3]) if parts[1] == 'layers' else '.'.join(parts[:2])


def zeros_like_parameters(p):
    return {k: torch.zeros_like(v) for k, v in p.items()}


class Credit(nn.Module):
    """Shared coordinate predictor; two persistent numbers per addressed block.

Forward packet: causal h mean/std, sequence elapsed time. It is available at
learning time after the support stream. Generated outcomes are never inserted
in pre-outcome inference features. Credit is distinct from cause attribution.
"""
    def __init__(self):
        super().__init__()
        self.log_rate = nn.Parameter(torch.tensor([-2., -4.], dtype=DT))
        self.predictor = nn.Sequential(nn.Linear(8, 8), nn.Tanh(), nn.Linear(8, 1)).to(DT)
        self.transition = nn.Sequential(nn.Linear(5, 8), nn.Tanh(), nn.Linear(8, 2)).to(DT)
        with torch.no_grad():
            self.predictor[-1].weight.mul_(.03)
            self.predictor[-1].bias.zero_()

    def transport(self, state, elapsed):
        if elapsed < 0:
            raise ValueError('Elapsed time cannot be negative')
        return state * torch.exp(-self.log_rate.exp() * elapsed)

    def predict(self, parameters, local, packet, state, elapsed):
        result = {}
        for name, p in parameters.items():
            c = self.transport(state.get(group(name), torch.zeros(2, dtype=DT)), elapsed)
            g = local[name].detach()
            scale = g.square().mean().sqrt().detach() + .01
            features = torch.stack((p.detach().tanh(), g / scale,
                                    torch.full_like(p, float(packet[0])),
                                    torch.full_like(p, float(packet[1])),
                                    torch.full_like(p, math.log1p(float(packet[2]))),
                                    c[0].expand_as(p), c[1].expand_as(p),
                                    torch.ones_like(p)), -1)
            result[name] = scale * self.predictor(features).squeeze(-1)
        return result

    def observe(self, state, residual, packet, elapsed):
        result = {}
        grouped = {}
        for name, r in residual.items():
            grouped.setdefault(group(name), []).append(r.detach().flatten())
        for name, values in grouped.items():
            r = torch.cat(values)
            c = self.transport(state.get(name, torch.zeros(2, dtype=DT)), elapsed)
            evidence = torch.stack((r.mean().tanh(), r.square().mean().sqrt().tanh(),
                                    torch.as_tensor(packet[0], dtype=DT).tanh()))
            result[name] = c + .1 * self.transition(torch.cat((c, evidence)))
        return result


class Update(nn.Module):
    """Learned coordinate gain, time-decayed first/second moment state."""
    def __init__(self):
        super().__init__()
        self.log_rate = nn.Parameter(torch.tensor([-2., -3.], dtype=DT))
        self.net = nn.Sequential(nn.Linear(6, 8), nn.Tanh(), nn.Linear(8, 1)).to(DT)
        with torch.no_grad():
            self.net[-1].weight.mul_(.03)
            self.net[-1].bias.fill_(-1.)

    def step(self, parameters, gradients, state, elapsed):
        if elapsed < 0:
            raise ValueError('Elapsed time cannot be negative')
        updated, nextstate = {}, {}
        retention = torch.exp(-self.log_rate.exp() * elapsed)
        for name, p in parameters.items():
            g = gradients[name]
            m, v = state.get(name, (torch.zeros_like(g), torch.zeros_like(g)))
            m = retention[0] * m + (1 - retention[0]) * g
            v = retention[1] * v + (1 - retention[1]) * g.square()
            scale = (v + 1e-8).sqrt() + .1
            features = torch.stack((g / scale, m / scale, torch.log1p(v),
                                    p.tanh(), torch.full_like(p, math.log1p(elapsed)),
                                    torch.ones_like(p)), -1)
            gain = .0003 + .0057 * self.net(features).squeeze(-1).sigmoid()
            updated[name] = p - gain * g / scale
            nextstate[name] = (m, v)
        return updated, nextstate


def actor_evidence(full, local, parameters, data, create_graph=False):
    t, m = data
    lf = functional_call(full, parameters, (t, m))
    ll = functional_call(local, parameters, (t, m))
    if not torch.allclose(lf, ll, atol=1e-12, rtol=1e-12):
        raise AssertionError('Credit substitution changed native forward predictions')
    def grad(l):
        raw = torch.autograd.grad(l, tuple(parameters.values()), create_graph=create_graph,
                                  allow_unused=True)
        return {k: torch.zeros_like(v) if g is None else g for (k, v), g in zip(parameters.items(), raw)}
    gf, gl = grad(lf), grad(ll)
    # This driver deliberately uses detached teacher derivatives for a ONE-step
    # meta-update from fixed actor parameters. Multi-step exact meta-credit is
    # not implemented. Forward features are detached from the auxiliary learner.
    packet = (float(full.model.producer_summary[0]), float(full.model.producer_summary[1]),
              float((t[:, -1] - t[:, 0]).mean())) if hasattr(full.model, 'producer_summary') else (
                  float(full.model.embed(m).detach().mean()),
                  float(full.model.embed(m).detach().std()), float((t[:, -1] - t[:, 0]).mean()))
    return float(lf.detach()), gf, gl, packet


def corrected(local, prediction, target, bits, probabilities):
    result = {}
    for name, g in local.items():
        prob = probabilities[group(name)]
        if not 0 < prob <= 1:
            raise ValueError('Audit support must be positive')
        residual = target[name] - g - prediction[name]
        result[name] = g + prediction[name] + bits[group(name)] / prob * residual
    return result


def flat(values):
    return torch.cat([x.reshape(-1) for x in values.values()])


def parameters(model):
    return {k: v.detach().clone().requires_grad_(True) for k, v in model.named_parameters()}


def finite_difference(module, objective, coordinate, epsilon=1e-5):
    ps = list(module.parameters())
    p, idx = coordinate
    parameter = ps[p]
    old = float(parameter.flatten()[idx])
    with torch.no_grad():
        parameter.flatten()[idx] = old + epsilon
    plus = float(objective().detach())
    with torch.no_grad():
        parameter.flatten()[idx] = old - epsilon
    minus = float(objective().detach())
    with torch.no_grad():
        parameter.flatten()[idx] = old
    return (plus - minus) / (2 * epsilon)


def contracts():
    rows = []
    for depth in (2, 4, 8):
        full, local = native_pair(depth, 164)
        p = parameters(full)
        _, past_full, past_local, past_packet = actor_evidence(full, local, p, batch(163))
        _, gf, gl, packet = actor_evidence(full, local, p, batch(164))
        past_residual = {k: past_full[k] - past_local[k] for k in p}
        credit = Credit()
        state = credit.observe({}, past_residual, past_packet, .4)
        pred = credit.predict(p, gl, packet, state, .6)
        groups = list(dict.fromkeys(group(k) for k in p))
        probs = {k: .25 + .5 * i / max(1, len(groups) - 1) for i, k in enumerate(groups)}
        # Enumerate each block's independent inclusion analytically; joint mean
        # and covariance factorize, avoiding exponential growth at depth eight.
        means, variances = {}, {}
        for name in p:
            b = group(name); prob = probs[b]
            absent = gl[name] + pred[name]
            present = absent + (gf[name] - absent) / prob
            means[name] = (1 - prob) * absent + prob * present
            actual = (1 - prob) * (absent - gf[name]).square() + prob * (present - gf[name]).square()
            declared = (1 / prob - 1) * (gf[name] - absent).square()
            variances[name] = float((actual - declared).abs().max())
        error = float((flat(means) - flat(gf)).abs().max())
        assert error < 1e-10 and max(variances.values()) < 1e-10
        # All audits recover full credit and therefore the same learned step.
        ones = {k: 1 for k in groups}; unit = {k: 1. for k in groups}
        exact = corrected(gl, pred, gf, ones, unit)
        update = Update()
        pa, _ = update.step(p, exact, {}, 1.)
        pb, _ = update.step(p, gf, {}, 1.)
        step_error = float((flat(pa) - flat(pb)).abs().max())
        assert step_error < 1e-12
        c = torch.tensor([.3, -.7], dtype=DT)
        transport_error = float((credit.transport(credit.transport(c, .3), .7) - credit.transport(c, 1.)).abs().max())
        assert transport_error < 1e-12
        # Fixed audit mask; exact one-update dependence on psi/omega, including
        # stored credit activations from the earlier observe transition.
        bits = {k: int(i % 2 == 0) for i, k in enumerate(groups)}
        future = batch(165)
        def objective():
            cs = credit.observe({}, past_residual, past_packet, .4)
            prediction = credit.predict(p, gl, packet, cs, .6)
            g = corrected(gl, prediction, gf, bits, probs)
            changed, _ = update.step(p, g, {}, 1.)
            return functional_call(full, changed, future)
        obj = objective()
        allpsi = list(credit.parameters()); allomega = list(update.parameters())
        derivatives = torch.autograd.grad(obj, allpsi + allomega)
        meta = {}
        for label, module, derivs in [('credit', credit, derivatives[:len(allpsi)]),
                                      ('optimizer', update, derivatives[len(allpsi):])]:
            # Strongest derivative is a non-vacuous witness.
            pi = max(range(len(derivs)), key=lambda j: float(derivs[j].abs().max()))
            ci = int(derivs[pi].abs().flatten().argmax())
            analytical = float(derivs[pi].flatten()[ci])
            numerical = finite_difference(module, objective, (pi, ci))
            assert abs(analytical) > 1e-9
            assert math.isclose(analytical, numerical, rel_tol=2e-4, abs_tol=2e-8), (label, analytical, numerical)
            meta[label] = dict(analytical=analytical, finite_difference=numerical, absolute_error=abs(analytical - numerical))
        rows.append(dict(depth=depth, parameters=sum(v.numel() for v in p.values()),
                         audit_blocks=len(groups), mean_error=error, variance_identity_error=max(variances.values()),
                         full_audit_update_error=step_error, temporal_semigroup_error=transport_error,
                         one_step_future_meta_derivatives=meta,
                         persistent_credit_state_elements=2*len(groups), optimizer_state_elements=2*sum(v.numel() for v in p.values())))
    return rows


def smoke(steps):
    full, local = native_pair(2, 164)
    p = parameters(full)
    credit, update = Credit(), Update()
    opt = torch.optim.Adam(list(credit.parameters()) + list(update.parameters()), lr=.001)
    history = []
    for step in range(steps):
        _, past_full, past_local, past_packet = actor_evidence(full, local, p, batch(500 + step))
        _, gf, gl, packet = actor_evidence(full, local, p, batch(1000 + step))
        residual = {k: gf[k] - gl[k] for k in p}
        # An earlier observed support event trains the persistent credit state;
        # adaptation is assessed on an independent subsequent DEV batch.
        c = credit.observe({}, {k: past_full[k] - past_local[k] for k in p}, past_packet, .5)
        prediction = credit.predict(p, gl, packet, c, .5)
        groups = list(dict.fromkeys(group(k) for k in p))
        probs = {k: .5 for k in groups}
        gen = torch.Generator().manual_seed(2000 + step)
        bits = {k: int(torch.rand((), generator=gen) < .5) for k in groups}
        g = corrected(gl, prediction, gf, bits, probs)
        changed, _ = update.step(p, g, {}, 1.)
        future = batch(3000 + step)
        future_loss = functional_call(full, changed, future)
        calibration = sum(bits[group(k)] / probs[group(k)] *
                          (prediction[k] - residual[k]).square().sum() for k in p) / sum(v.numel() for v in p.values())
        objective = future_loss + calibration
        opt.zero_grad(); objective.backward()
        psi_norm = math.sqrt(sum(float(x.grad.square().sum()) for x in credit.parameters() if x.grad is not None))
        omega_norm = math.sqrt(sum(float(x.grad.square().sum()) for x in update.parameters() if x.grad is not None))
        assert math.isfinite(float(objective)) and psi_norm > 0 and omega_norm > 0
        opt.step()
        history.append(dict(step=step, future_nll=float(future_loss.detach()), calibration_mse=float(calibration.detach()),
                            credit_parameter_gradient_norm=psi_norm, optimizer_parameter_gradient_norm=omega_norm,
                            selected_audit_blocks=sum(bits.values()), total_audit_blocks=len(groups)))
        print(json.dumps(history[-1]), flush=True)
    return dict(history=history, actor_support_and_future_sequences=steps*6,
                teacher_full_and_local_gradient_calls=steps*4, independent_future_loss_calls=steps,
                actual_sparse_teacher_savings=False,
                scope='Fixed initialized native actor, one adapted step per independent task, shared credit/update parameters learn; no full actor fit or quality claim')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tag', required=True)
    parser.add_argument('--mode', choices=('contract', 'smoke'), required=True)
    parser.add_argument('--steps', type=int, default=3)
    args = parser.parse_args()
    out = ROOT / 'experiments/results/credit' / (args.tag + '.json')
    if out.exists():
        raise FileExistsError(out)
    torch.set_num_threads(1); torch.set_default_dtype(DT); torch.manual_seed(164)
    start = time.monotonic()
    result = contracts() if args.mode == 'contract' else smoke(args.steps)
    paths = ['experiments/credit/reciprocal_native_v1.py', 'experiments/tpp/recall_tpp_v4.py', 'experiments/tpp/race_tpp_v5.py']
    record = dict(status='completed', tag=args.tag, battle='R1/B1', args=vars(args),
                  decision='Whether persistent reciprocal credit and learned optimizer admit a bounded native DEV pilot',
                  training=args.mode == 'smoke', metrics=result, wall_s=time.monotonic()-start,
                  peak_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, host=socket.gethostname(),
                  source_sha256={s: hashlib.sha256((ROOT/s).read_bytes()).hexdigest() for s in paths},
                  scope='Native keyed temporal model, small synthetic development sequences; dense exact teachers charged; no TEST or efficiency claim')
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix('.tmp'); tmp.write_text(json.dumps(record, indent=2)+'\n'); tmp.replace(out)
    print('RESULT', json.dumps(record), flush=True)


if __name__ == '__main__':
    main()
