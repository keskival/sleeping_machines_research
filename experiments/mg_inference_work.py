"""Per-step inference work of NeuroBench Mackey-Glass native members (operator tracer; shapes only, untrained weights).

Counts one autonomous step (after a short warm-up) of the winner-only stepper (sampled / argmax: one stream; mixK: K
streams) and of the exact-expected-reception stepper, in the unit/special convention of race_language_screen.capture.
Multiply-accumulates are reported as arithmetic FLOPs / 2 for comparison with NeuroBench's effective-MAC column (which
counts hooked layer MACs; conventions differ and are labelled).
"""
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'experiments'))
from race_language_screen import capture  # noqa: E402
from sleeping_machines.addressed_event_heads import AddressedEventHeads  # noqa: E402
from sleeping_machines.expected_reception import ExpectedStepper  # noqa: E402
from sleeping_machines.fast_native_core import fast_class  # noqa: E402
from sleeping_machines.sparse_inference import SparseStepper  # noqa: E402


def per_step(stepper, taps):
    x = torch.zeros(1, taps)
    for t in range(5):
        stepper.step(torch.tensor([float(t)]), x)
    rec = capture(lambda: stepper.step(torch.tensor([5.]), x))
    return rec['arithmetic_flops'], rec['special_function_evaluations']


if __name__ == '__main__':
    out = ROOT / sys.argv[1]
    rows = []
    for payload, pool in ((16, 2), (16, 4), (32, 2)):
        torch.manual_seed(0)
        m = fast_class(AddressedEventHeads)(sources=1, content_dim=8, classes=1, payload=payload, depth=2, heads=2, pool=pool)
        a_s, s_s = per_step(SparseStepper(m, 1, 7), 8)
        a_e, s_e = per_step(ExpectedStepper(m, 1), 8)
        rows.append(dict(payload=payload, depth=2, heads=2, pool=pool, parameters=sum(p.numel() for p in m.parameters()),
                         winner_stream_flops=a_s, winner_stream_special=s_s, winner_stream_macs=a_s / 2,
                         mix8_macs=8 * a_s / 2, expected_flops=a_e, expected_special=s_e, expected_macs=a_e / 2))
        print(json.dumps(rows[-1]), flush=True)
    out.write_text(json.dumps(dict(rows=rows, scope=__doc__.strip()), indent=2) + '\n')
