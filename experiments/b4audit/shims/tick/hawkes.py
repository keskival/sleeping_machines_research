def _unavailable(*a, **k):
    raise RuntimeError('tick shim: Hawkes simulation is not available in the B4 audit environment')
SimuHawkes = SimuHawkesSumExpKernels = SimuHawkesMulti = SimuHawkesExpKernels = HawkesKernelExp = _unavailable
