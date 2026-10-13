"""B1/B10: admit compact delayed transport only after native adjoint parity."""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'experiments/tpp'))
from sleeping_machines.learning.temporal_credit_transport import transport_cotangent
from race_tpp_v19 import TemporalMemoryLayer


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--tag', required=True)
    args = parser.parse_args(); started = time.monotonic()
    torch.set_num_threads(1); torch.set_default_dtype(torch.float64); torch.manual_seed(19123)
    rows = []
    for horizon in (2, 8, 32, 128):
        layer = TemporalMemoryLayer(32, 16, 0.).eval()
        x = torch.randn(1, horizon, 32, requires_grad=True)
        gaps = torch.rand(1, horizon) * .3; gaps[:, 0] = 0
        output = layer(x, gaps)
        demand = torch.randn(32)
        exact = torch.autograd.grad(output[0, -1] @ demand, x)[0][0, 0]
        def write(u):
            w = layer.write(u); g = torch.sigmoid(layer.gate(u))
            return torch.cat((w[:16] * g, w[16:] * g))
        state = torch.zeros(32)
        for k in range(horizon):
            # Forward uses the transpose of the same adjoint rotation.
            state = transport_cotangent(state, layer.log_rate, -layer.freq, gaps[0, k]) + write(x[0, k])
        def read(s):
            h = layer.norm1(x[0, -1] + layer.read(s))
            return layer.norm2(h + layer.mlp(h))
        endpoint = state.detach().requires_grad_()
        cotangent = torch.autograd.grad(read(endpoint) @ demand, endpoint)[0]
        pulled = transport_cotangent(cotangent, layer.log_rate, layer.freq, gaps[0, 1:].sum())
        first = write(x[0, 0])
        compact = torch.autograd.grad(first, x, grad_outputs=pulled.detach())[0][0, 0]
        sequential = cotangent
        for k in range(horizon - 1, 0, -1):
            sequential = transport_cotangent(sequential, layer.log_rate, layer.freq, gaps[0, k])
        error = float((compact - exact).abs().max())
        composition = float((pulled - sequential).abs().max())
        assert max(error, composition) < 1e-11
        rows.append(dict(horizon=horizon, input_credit_error=error,
                         collapsed_vs_sequential_error=composition))
    result = dict(status='completed', battle='B1/B10', rows=rows,
                  state_packet_floats=32, dense_transition_floats=1024,
                  rotation_contraction_flops=96, dense_contraction_flops=2048,
                  scope='Fixed-version single native temporal layer; endpoint derivatives remain exact and paid. Direct state-path transport only, not full stacked actor/learner credit or measured total-work reduction. Clock construction nonlinear work and endpoint VJPs excluded from contraction counts.',
                  wall_s=time.monotonic()-started,
                  source_sha256={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in (
                      'experiments/credit/check_temporal_credit_transport.py',
                      'sleeping_machines/learning/temporal_credit_transport.py', 'experiments/tpp/race_tpp_v19.py')})
    with (ROOT/f'experiments/results/credit/{args.tag}.json').open('x') as f:
        json.dump(result, f, indent=2); f.write('\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
