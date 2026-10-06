"""Positive selected-unit time scale; race and transport clocks stay intact."""
from contextlib import contextmanager
from functools import wraps
from inspect import signature
import math


def scaled_unit(original, scale):
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError('Memory time scale must be finite and positive')
    contract = signature(original)
    if not {'rate', 'frequency'}.issubset(contract.parameters):
        raise ValueError('Unsupported selected-unit signature')

    @wraps(original)
    def intercepted(*args, **kwargs):
        if scale == 1:
            return original(*args, **kwargs)
        bound = contract.bind(*args, **kwargs)
        for field in ('rate', 'frequency'):
            bound.arguments[field] = bound.arguments[field] * scale
        return original(*bound.args, **bound.kwargs)

    return intercepted


@contextmanager
def installed_time_scale(kernel, scale):
    original = kernel._unit
    kernel._unit = scaled_unit(original, scale)
    try:
        yield
    finally:
        kernel._unit = original
