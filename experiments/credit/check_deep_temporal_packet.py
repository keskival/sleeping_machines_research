"""B1 shared-clock depth contract, including real actor producer/Adam parity."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import resource
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'experiments/tpp'))
import race_tpp_v19 as native
from native_deep_temporal_packet import NativeDeepTemporalPacket


class DeepTimedActor(native.RaceTPP):
    def __init__(self, *args):
        super().__init__(*args)
        self.tempo = torch.nn.Linear(3, 1)
        with torch.no_grad():
            self.tempo.weight.zero_()
            self.tempo.bias.fill_(math.log(math.expm1(1.)))

    def encode(self, t, m, mask):
        dt = torch.diff(t, dim=1, prepend=t[:, :1]).clamp_min(0) / self.scale
        age = (t - t[:, :1]) / self.scale
        features = torch.stack([torch.log1p(dt), (dt > 0).to(t.dtype), torch.log1p(age)], -1)
        content = self.embed(m) + self.gap(features)
        elapsed = dt * torch.nn.functional.softplus(self.tempo(features).squeeze(-1))
        if getattr(self, 'packet_mode', False):
            self.cut_input, self.cut_elapsed = content, elapsed
            self.packet = NativeDeepTemporalPacket.capture(
                self.layers, content.detach(), elapsed.detach(), context='taxi',
                version=self.credit_version, connected=True)
            h = self.packet.output
            h.retain_grad()
            self.cut_output = h
        else:
            h = content
            for layer in self.layers:
                h = layer(h, elapsed)
        return h, self.marks(h, m, dt)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tag', required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(943)
    start = time.monotonic()
    train = native.load_split('taxi', 'train')
    seq = next(s for s in train if len(s[1]) >= 32)
    K = 1 + max(int(m.max()) for _, m in train)
    gaps = np.diff(seq[0][:32])
    positive = gaps[gaps > 0]
    scale = float(np.median(positive))
    qs = np.log(np.quantile(positive, [.1, .5, .9])).tolist()
    rows, actor_rows = [], []
    for depth in [2, 4, 8]:
        for length in [4, 32]:
            layers = torch.nn.ModuleList([native.TemporalMemoryLayer(8, 4, 0.).eval()
                                          for _ in range(depth)])
            content = torch.randn(1, length, 8)
            times = torch.tensor(seq[0][:length])[None]
            dt = torch.diff(times, dim=1, prepend=times[:, :1]) / scale
            def forward(x, elapsed):
                for layer in layers:
                    x = layer(x, elapsed)
                return x
            packet = NativeDeepTemporalPacket.capture(layers, content, dt,
                                                      context='taxi', version=0)
            v = torch.randn(4, *content.shape)
            w = torch.randn(4, *dt.shape)
            c = torch.randn(4, *content.shape)
            response = packet.apply_with_time(v, w, context='taxi', version=0)
            expected = torch.func.vmap(lambda x, y: torch.func.jvp(
                forward, (content, dt), (x, y))[1])(v, w)
            x, elapsed = content.clone().requires_grad_(), dt.clone().requires_grad_()
            exact = torch.autograd.grad(forward(x, elapsed), (x, elapsed), c,
                                        is_grads_batched=True)
            actual = packet.transpose_with_time(c, context='taxi', version=0)
            jvp_error = float((response - expected).abs().max())
            vjp_error = max(float((a - b).abs().max()) for a, b in zip(actual, exact))
            reciprocity = float(((response * c).flatten(1).sum(-1) -
                                 (actual[0] * v).flatten(1).sum(-1) -
                                 (actual[1] * w).flatten(1).sum(-1)).abs().max())
            assert max(jvp_error, vjp_error, reciprocity) < 1e-10
            # A suffix cannot alter any prefix primal or derivative.
            prefix = length // 2
            altered_content, altered_dt = content.clone(), dt.clone()
            altered_content[:, prefix:] += 2
            altered_dt[:, prefix:] += 1
            changed = NativeDeepTemporalPacket.capture(layers, altered_content, altered_dt,
                                                       context='taxi', version=0)
            causal = max(float((packet.output[:, :prefix] - changed.output[:, :prefix]).abs().max()),
                         float((response[:, :, :prefix] - changed.apply_with_time(
                             v, w, context='taxi', version=0)[:, :, :prefix]).abs().max()))
            assert causal == 0
            with torch.no_grad():
                layers[depth // 2].freq.add_(.001)
            for method, values in [(packet.apply_with_time, (v, w)),
                                   (packet.transpose_with_time, (c,))]:
                try:
                    method(*values, context='taxi', version=0)
                except ValueError:
                    pass
                else:
                    raise AssertionError('stale intermediate clock accepted')
            rows.append(dict(depth=depth, events=length, jvp_error=jvp_error,
                             vjp_error=vjp_error, reciprocity_error=reciprocity,
                             causal_suffix_error=causal, cached_floats=packet.cached_floats()))

        ctor = (K, 8, 4, depth, 2, 3, 2, 0., scale, qs)
        ref, actor = DeepTimedActor(*ctor).eval(), DeepTimedActor(*ctor).eval()
        actor.load_state_dict(ref.state_dict())
        actor.packet_mode = True
        ref_opt, opt = torch.optim.Adam(ref.parameters(), lr=.001), torch.optim.Adam(actor.parameters(), lr=.001)
        t, m = torch.tensor(seq[0][:8])[None], torch.tensor(seq[1][:8])[None]
        mask = torch.ones_like(m, dtype=torch.bool)
        for version in range(4):
            actor.credit_version = version
            ref_opt.zero_grad(); opt.zero_grad()
            tl, ml, n, _ = ref.loglik(t, m, mask)
            reference_loss = -(tl + ml) / n
            reference_loss.backward()
            tl, ml, n, _ = actor.loglik(t, m, mask)
            loss = -(tl + ml) / n
            loss.backward()
            gx, gt = actor.packet.transpose_with_time(actor.cut_output.grad[None],
                                                     context='taxi', version=version)
            torch.autograd.backward([actor.cut_input, actor.cut_elapsed], [gx[0], gt[0]])
            gradient_error = max(float((p.grad - q.grad).abs().max())
                                 for p, q in zip(ref.parameters(), actor.parameters()))
            tempo_gradient = float(actor.tempo.weight.grad.abs().sum() + actor.tempo.bias.grad.abs().sum())
            assert gradient_error < 1e-10 and tempo_gradient > 0
            assert abs(float(loss - reference_loss)) < 1e-10
            ref_opt.step(); opt.step()
            parameter_error = max(float((p - q).detach().abs().max())
                                  for p, q in zip(ref.parameters(), actor.parameters()))
            adam_error = max(float((ref_opt.state[p][k] - opt.state[q][k]).abs().max())
                             for p, q in zip(ref.parameters(), actor.parameters())
                             for k in ['exp_avg', 'exp_avg_sq'])
            assert max(parameter_error, adam_error) < 1e-10
            actor_rows.append(dict(depth=depth, version=version, loss=float(loss.detach()),
                                   gradient_error=gradient_error, parameter_error=parameter_error,
                                   adam_error=adam_error, tempo_gradient_norm=tempo_gradient))
    sources = ['experiments/credit/check_deep_temporal_packet.py',
               'experiments/credit/native_deep_temporal_packet.py',
               'experiments/credit/native_temporal_teaching_packet.py',
               'experiments/credit/native_training_temporal_packet.py',
               'experiments/credit/native_training_linearization.py',
               'experiments/credit/native_cached_linearization.py',
               'experiments/credit/native_cached_jvp.py', 'experiments/tpp/race_tpp_v19.py']
    result = dict(status='completed', tag=args.tag, battle='B1 deep joint temporal credit',
                  rows=rows, actor_rows=actor_rows,
                  source_sha256={s: hashlib.sha256((ROOT / s).read_bytes()).hexdigest() for s in sources},
                  data_sha256=hashlib.sha256((ROOT / 'data/easytpp/taxi/train.json').read_bytes()).hexdigest(),
                  wall_s=time.monotonic() - start, max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                  scope='Native dense-write execution diagnostic; depths2/4/8, shared scaled elapsed time, real Taxi TRAIN gaps. Exact content/time transport, intermediate-clock invalidation, causal suffix and all-parameter Adam parity. No learned-credit, task-quality, hard-route, full-fit or efficiency claim; active architecture unchanged.')
    with (ROOT / 'experiments/results/credit' / f'{args.tag}.json').open('x') as output:
        json.dump(result, output, indent=2)
    print(json.dumps(result))


if __name__ == '__main__':
    main()
