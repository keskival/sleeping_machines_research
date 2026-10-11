"""B1 forward-taught neural credit gradient vs exact cut/producer objectives."""
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
from check_joint_forward_risk import SharedProducerActor, producer_directions, probes, native
from joint_forward_credit import joint_credit_teaching_loss


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tag', required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(945)
    start = time.monotonic()
    train = native.load_split('taxi', 'train')
    seq = next(s for s in train if len(s[1]) >= 4)
    K = 1 + max(int(m.max()) for _, m in train)
    pos = np.diff(seq[0][:4]); pos = pos[pos > 0]
    scale = float(np.median(pos)); qs = np.log(np.quantile(pos, [.1, .5, .9])).tolist()
    t, m = torch.tensor(seq[0][:4])[None], torch.tensor(seq[1][:4])[None]
    mask = torch.ones_like(m, dtype=torch.bool)
    rows = []
    for depth in [2, 4, 8]:
        model = SharedProducerActor(K, 8, 4, depth, 2, 3, 2, 0., scale, qs).eval()
        tl, ml, count, _ = model.loglik(t, m, mask)
        loss = -(tl + ml) / count
        loss.backward()
        c = model.packet.output.grad.detach()
        x, elapsed, dt, features, log_tempo = model.produce(t, m)
        context = torch.cat([x.flatten(), dt.flatten(), c.flatten()]).detach()
        dimension = x.numel() + dt.numel()
        critic = torch.nn.Sequential(torch.nn.Linear(len(context), 16), torch.nn.Tanh(),
                                     torch.nn.Linear(16, dimension))
        params = list(critic.parameters())
        directions = probes(dimension)
        v = directions[:, :x.numel()].reshape(-1, *x.shape)
        w = directions[:, x.numel():].reshape(-1, *dt.shape)
        producer_params = [model.embed.weight, model.gap.weight, model.gap.bias,
                           model.tempo.weight, model.tempo.bias]
        P = sum(p.numel() for p in producer_params)
        pv, pw = producer_directions(model, m, dt, features, log_tempo, probes(P))
        av, aw = producer_directions(model, m, dt, features, log_tempo, torch.eye(P))
        A = torch.cat([av.flatten(1), aw.flatten(1)], 1).detach()
        tick = time.monotonic()
        response = model.packet.apply_with_time(v, w, context='taxi', version=0)
        producer_response = model.packet.apply_with_time(pv, pw, context='taxi', version=0)
        label_s = time.monotonic() - tick
        # Exact backward maps are validation only and never create teaching targets.
        tick = time.monotonic()
        gx, gt = model.packet.transpose_with_time(c[None], context='taxi', version=0)
        exact = torch.cat([gx.flatten(), gt.flatten()]).detach()
        reverse_s = time.monotonic() - tick
        predicted = critic(context)
        px, pt = predicted[:x.numel()].reshape_as(x), predicted[x.numel():].reshape_as(dt)
        teaching = joint_credit_teaching_loss(px, pt, c, v, w, response)
        reference = .5 * (predicted - exact).square().sum()
        teaching_grad = torch.autograd.grad(teaching, params, retain_graph=True)
        exact_grad = torch.autograd.grad(reference, params, retain_graph=True)
        cut_gradient_error = max(float((a-b).abs().max()) for a,b in zip(teaching_grad, exact_grad))
        projected = joint_credit_teaching_loss(px, pt, c, pv, pw, producer_response)
        projected_reference = .5 * (A @ (predicted - exact)).square().sum()
        projected_grad = torch.autograd.grad(projected, params, retain_graph=True)
        projected_exact_grad = torch.autograd.grad(projected_reference, params)
        producer_gradient_error = max(float((a-b).abs().max())
                                      for a,b in zip(projected_grad, projected_exact_grad))
        assert max(cut_gradient_error, producer_gradient_error) < 1e-10
        assert abs(float(teaching.detach() - reference.detach())) < 1e-10
        assert abs(float(projected.detach() - projected_reference.detach())) < 1e-10
        # Verify finite local optimizer use on this issued observation; not held-out utility.
        opt = torch.optim.SGD(critic.parameters(), lr=.001)
        initial = float(reference.detach())
        for _ in range(4):
            opt.zero_grad()
            p = critic(context)
            fit_loss = joint_credit_teaching_loss(p[:x.numel()].reshape_as(x),
                       p[x.numel():].reshape_as(dt), c, v, w, response)
            fit_loss.backward(); opt.step()
        final = float(.5 * (critic(context).detach() - exact).square().sum())
        assert final < initial and all(torch.isfinite(p).all() for p in params)
        rows.append(dict(depth=depth, estimator_parameters=sum(p.numel() for p in params),
                         cut_probe_count=len(v), producer_probe_count=len(pv),
                         producer_parameters=P, cut_gradient_error=cut_gradient_error,
                         producer_gradient_error=producer_gradient_error,
                         initial_in_sample_credit_error=initial, final_in_sample_credit_error=final,
                         forward_label_s=label_s, reverse_validation_s=reverse_s))
    sources = ['experiments/credit/check_joint_credit_training.py',
               'experiments/credit/joint_forward_credit.py',
               'experiments/credit/check_joint_forward_risk.py',
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
    result = dict(status='completed', tag=args.tag, battle='B1 joint forward-taught credit', rows=rows,
                  source_sha256={s: hashlib.sha256((ROOT/s).read_bytes()).hexdigest() for s in sources},
                  data_sha256=hashlib.sha256((ROOT/'data/easytpp/taxi/train.json').read_bytes()).hexdigest(),
                  wall_s=time.monotonic()-start, max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                  scope='One four-event Taxi TRAIN prefix per depth2/4/8, frozen native dense-write model, diagnostic shared content/time producer. Neural estimator gradient matches exact cut and shared-producer regression objectives using exhaustive forward probe labels; four in-sample SGD steps only. Reverse teacher is validation only. No acquired held-out credit structure, future task utility, hard-route, active model substitution or whole-fitting efficiency claim.')
    with (ROOT/'experiments/results/credit'/f'{args.tag}.json').open('x') as output:
        json.dump(result, output, indent=2)
    print(json.dumps(result))


if __name__ == '__main__':
    main()
