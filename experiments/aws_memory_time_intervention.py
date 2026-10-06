"""Diagnostic selected-unit temporal interventions; no parameter mutation.

Importing this module performs no numerical work. Execute numerical contracts
and selected-checkpoint scoring only through the guarded host queue.
"""
from contextlib import contextmanager
from functools import wraps
from inspect import signature

MODES = ('intact', 'no_decay', 'no_rotation', 'zero_age')


def unit_intervention(original, mode, zeros_like):
    """Intercept only the unit's temporal coefficients or seen mask.

    Race clocks, transport, payload, cached keys and learned parameters remain
    supplied unchanged. Subsequent state and routes may change causally.
    """
    if mode not in MODES:
        raise ValueError('Unknown temporal intervention: ' + mode)
    contract = signature(original)
    required = {'rate', 'frequency', 'seen_d', 'm', 'arr_d'}
    if not required.issubset(contract.parameters):
        raise ValueError('Unsupported selected-unit signature')

    @wraps(original)
    def intercepted(*args, **kwargs):
        if mode == 'intact':
            return original(*args, **kwargs)
        bound = contract.bind(*args, **kwargs)
        field = {'no_decay': 'rate', 'no_rotation': 'frequency',
                 'zero_age': 'seen_d'}[mode]
        bound.arguments[field] = zeros_like(bound.arguments[field])
        return original(*bound.args, **bound.kwargs)

    return intercepted


@contextmanager
def installed_unit_intervention(kernel, mode, zeros_like):
    """Process-local diagnostic hook, restored even when scoring raises.

    The caller supplies the imported sparse_counterfactual_layer module.
    Use sequentially in a one-thread guarded job, never during a live fit.
    """
    original = kernel._unit
    intercepted = unit_intervention(original, mode, zeros_like)
    kernel._unit = intercepted
    try:
        yield
    finally:
        kernel._unit = original
