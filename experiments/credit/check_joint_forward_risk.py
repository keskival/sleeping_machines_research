"""B1 joint forward-only risk labels, shared producer cross terms and depth."""
import argparse
import hashlib
import json
from pathlib import Path
import resource
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'experiments/tpp'))
from check_deep_temporal_packet import DeepTimedActor, native
from native_fused_temporal_packet import NativeDeepFusedTemporalPacket
from joint_forward_residual_risk import joint_residual_observations
from predictable_audit_allocation_v2 import ResidualRiskLearner


def probes(dimension):
    h = torch.ones(1, 1)
    while len(h) < dimension:
        h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
    return h[:, :dimension]


class SharedProducerActor(DeepTimedActor):
    def produce(self, t, m):
        dt = torch.diff(t, dim=1, prepend=t[:, :1]).clamp_min(0) / self.scale
        age = (t - t[:, :1]) / self.scale
        f = torch.stack([torch.log1p(dt), (dt > 0).to(t.dtype), torch.log1p(age)], -1)
        x = self.embed(m) + self.gap(f)
        log_tempo = self.tempo(f).squeeze(-1) + .1 * x.mean(-1)
        elapsed = dt * torch.nn.functional.softplus(log_tempo)
        return x, elapsed, dt, f, log_tempo

    def encode(self, t, m, mask):
        x, elapsed, physical_dt, _, _ = self.produce(t, m)
        self.packet = NativeDeepFusedTemporalPacket.capture(
            self.layers, x, elapsed, context='taxi', version=0, connected=True)
        h = self.packet.output
        h.retain_grad()
        return h, self.marks(h, m, physical_dt)


def producer_directions(model, m, dt, features, log_tempo, directions):
    shapes = [model.embed.weight.shape, model.gap.weight.shape, model.gap.bias.shape,
              model.tempo.weight.shape, model.tempo.bias.shape]
    sizes = [int(np.prod(shape)) for shape in shapes]
    pieces = [piece.reshape(len(directions), *shape)
              for piece, shape in zip(directions.split(sizes, dim=1), shapes)]
    embedding, weight, bias, tempo_weight, tempo_bias = pieces
    x = embedding[:, m] + torch.einsum('blf,qdf->qbld', features, weight) + bias[:, None, None, :]
    log_tempo_direction = (torch.einsum('blf,qdf->qbld', features, tempo_weight).squeeze(-1) +
                           tempo_bias[:, None, :] + .1 * x.mean(-1))
    elapsed = dt * torch.sigmoid(log_tempo) * log_tempo_direction
    return x, elapsed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tag', required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(944)
    start = time.monotonic()
    train = native.load_split('taxi', 'train')
    seq = next(s for s in train if len(s[1]) >= 4)
    K = 1 + max(int(m.max()) for _, m in train)
    positive = np.diff(seq[0][:4]); positive = positive[positive > 0]
    scale = float(np.median(positive))
    qs = np.log(np.quantile(positive, [.1, .5, .9])).tolist()
    t, m = torch.tensor(seq[0][:4])[None], torch.tensor(seq[1][:4])[None]
    mask = torch.ones_like(m, dtype=torch.bool)
    rows = []
    for depth in [2, 4, 8]:
        model = SharedProducerActor(K, 8, 4, depth, 2, 3, 2, 0., scale, qs).eval()
        tl, ml, count, _ = model.loglik(t, m, mask)
        loss = -(tl + ml) / count
        loss.backward()
        c = model.packet.output.grad.detach()
        # Deliberately imperfect fields; labels train risk without reverse targets.
        predicted_x, predicted_t = .5 * c, torch.zeros_like(t)
        joint_probes = probes(c.numel() + t.numel())
        v = joint_probes[:, :c.numel()].reshape(-1, *c.shape)
        w = joint_probes[:, c.numel():].reshape(-1, *t.shape)
        x, elapsed, dt, features, log_tempo = model.produce(t, m)
        params = [model.embed.weight, model.gap.weight, model.gap.bias,
                  model.tempo.weight, model.tempo.bias]
        P = sum(p.numel() for p in params)
        parameter_probes = probes(P)
        pv, pw = producer_directions(model, m, dt, features, log_tempo, parameter_probes)
        tick = time.monotonic()
        response = model.packet.apply_with_time(v, w, context='taxi', version=0)
        _, labels = joint_residual_observations(c, predicted_x, predicted_t, v, w, response)
        response_p = model.packet.apply_with_time(pv, pw, context='taxi', version=0)
        _, producer_labels = joint_residual_observations(c, predicted_x, predicted_t, pv, pw, response_p)
        label_s = time.monotonic() - tick
        # Reverse validation and its work are separate from forward label generation.
        tick = time.monotonic()
        exact_x, exact_t = model.packet.transpose_with_time(c[None], context='taxi', version=0)
        rx, rt = exact_x[0] - predicted_x, exact_t[0] - predicted_t
        cut_risk = float(rx.square().sum() + rt.square().sum())
        gx = torch.autograd.grad((x * rx).sum(), params, allow_unused=True, retain_graph=True)
        gt = torch.autograd.grad((elapsed * rt).sum(), params, allow_unused=True)
        gx = [torch.zeros_like(p) if g is None else g for p, g in zip(params, gx)]
        gt = [torch.zeros_like(p) if g is None else g for p, g in zip(params, gt)]
        producer_risk = sum(float((a + b).square().sum()) for a, b in zip(gx, gt))
        separated = sum(float(a.square().sum() + b.square().sum()) for a, b in zip(gx, gt))
        cross_term = 2 * sum(float((a * b).sum()) for a, b in zip(gx, gt))
        validation_s = time.monotonic() - tick
        cut_error = abs(float(labels.mean()) - cut_risk)
        producer_error = abs(float(producer_labels.mean()) - producer_risk)
        assert max(cut_error, producer_error, abs(producer_risk - separated - cross_term)) < 1e-10
        risk = ResidualRiskLearner(2)
        issued = risk.issue([1., depth / 8], 1.)
        gradient = np.mean([risk.gradient(issued, audited=True, residual_squared=float(z))
                            for z in producer_labels], axis=0)
        expected_gradient = risk.gradient(issued, audited=True, residual_squared=producer_risk)
        risk_gradient_error = float(np.max(np.abs(gradient - expected_gradient)))
        assert risk_gradient_error < 1e-10
        rows.append(dict(depth=depth, loss=float(loss.detach()), joint_cut_risk=cut_risk,
                         joint_cut_risk_error=cut_error, producer_risk=producer_risk,
                         producer_risk_error=producer_error, separate_output_risk=separated,
                         producer_cross_term=cross_term, risk_training_gradient_error=risk_gradient_error,
                         cut_probes=len(v), producer_probes=len(pv), producer_parameters=P,
                         forward_label_s=label_s, reverse_validation_s=validation_s))
    assert any(abs(row['producer_cross_term']) > 1e-8 for row in rows)
    sources = ['experiments/credit/check_joint_forward_risk.py',
               'experiments/credit/joint_forward_residual_risk.py',
               'experiments/credit/check_deep_temporal_packet.py',
               'experiments/credit/native_fused_temporal_packet.py',
               'experiments/credit/native_deep_temporal_packet.py',
               'experiments/credit/native_temporal_teaching_packet.py',
               'experiments/credit/native_training_temporal_packet.py',
               'experiments/credit/native_training_linearization.py',
               'experiments/credit/native_cached_linearization.py',
               'experiments/credit/native_cached_jvp.py',
               'experiments/credit/predictable_audit_allocation_v2.py',
               'experiments/tpp/race_tpp_v19.py']
    result = dict(status='completed', tag=args.tag, battle='B1 joint forward-observed risk', rows=rows,
                  source_sha256={s: hashlib.sha256((ROOT / s).read_bytes()).hexdigest() for s in sources},
                  data_sha256=hashlib.sha256((ROOT / 'data/easytpp/taxi/train.json').read_bytes()).hexdigest(),
                  wall_s=time.monotonic() - start, max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                  scope='Exact observation-label/conditional risk-gradient contract, depths2/4/8, one real4-event Taxi TRAIN prefix. Diagnostic positive internal tempo depends on shared content parameters; physical mark/race clocks unchanged. Exhaustive orthogonal labels use forward responses, separate reverse validation paid. No learned calibration, sparse-audit utility, task-quality, active architecture substitution or full-fit efficiency claim.')
    with (ROOT / 'experiments/results/credit' / f'{args.tag}.json').open('x') as output:
        json.dump(result, output, indent=2)
    print(json.dumps(result))


if __name__ == '__main__':
    main()
