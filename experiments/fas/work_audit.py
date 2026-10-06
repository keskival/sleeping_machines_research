"""Operation audit for FAS race-readout models: race_language_screen.RaceAudit plus counting formulas for the
operators the competing-risks readout uses (battle B3 work accounting; same conventions: arithmetic FLOPs, special
function evaluations and comparisons counted separately).

- logsumexp over m inputs per output: max (m − 1 comparisons), m subtractions, m exponentials, m − 1 additions,
  1 logarithm and 1 addition per output.
- log_sigmoid(x) = min(x, 0) − log1p(exp(−|x|)): 2 special functions, 2 arithmetic and 1 comparison per element. The
  backward pass is 3 arithmetic per element.
- log_ndtr(x) = log Φ(x): an erfc and a log (2 special functions) plus 3 arithmetic per element.
- logaddexp(a, b) = max(a, b) + log1p(exp(−|a − b|)): 2 special functions, 2 arithmetic and 2 comparisons per output.
- rsub: 1 arithmetic per element.
"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'experiments')); sys.path.insert(0, str(ROOT))
from race_language_screen import RaceAudit  # noqa: E402
from sleeping_machines.operation_audit import floating_size, tensors  # noqa: E402


class FasAudit(RaceAudit):
    def formula(self, func, args, kwargs, out):
        base = str(func).split('.')[1].rstrip('_')
        x = tensors(args)[0] if tensors(args) else None
        inp = floating_size(x) if x is not None else 0
        if base == 'logsumexp':
            n = floating_size(out)
            return 2 * inp + n, inp + n, max(inp - n, 0), 'logsumexp (max, shift, exp, sum, log)'
        if base == 'logaddexp':
            n = floating_size(out)
            return 2 * n, 2 * n, 2 * n, 'logaddexp: max(a, b) + log1p(exp(-|a - b|))'
        if base == 'rsub':
            return floating_size(out), 0, 0, 'elementwise arithmetic'
        if base == 'log_sigmoid_forward':
            return 2 * inp, 2 * inp, inp, 'log-sigmoid: min, exp, log1p'
        if base == 'log_sigmoid_backward':
            return 3 * inp, 0, 0, 'log-sigmoid derivative'
        if base == 'special_log_ndtr':
            return 3 * inp, 2 * inp, 0, 'log normal CDF: erfc + log'
        return super().formula(func, args, kwargs, out)


def capture(action):
    with FasAudit() as audit:
        action()
    result = audit.result()
    if not result['formula_coverage_complete']:
        raise ValueError(result['unsupported_floating_operators'])
    return result
