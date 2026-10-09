"""Allow-list unpickling for TGB negative-sample files (downloaded data): only the numpy array-reconstruction globals
found by pickletools in the official files are permitted; anything else raises. `patch_tgb()` routes py-tgb's load_pkl
through it."""
import pickle
import numpy as np

_ALLOWED = {('numpy', 'dtype'): np.dtype,
            ('numpy.core.numeric', '_frombuffer'): None, ('numpy._core.numeric', '_frombuffer'): None,
            ('numpy.core.multiarray', 'scalar'): None, ('numpy._core.multiarray', 'scalar'): None}


class _Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if (module, name) not in _ALLOWED:
            raise pickle.UnpicklingError(f'blocked global {module}.{name}')
        if name == 'dtype':
            return np.dtype
        mod = module.replace('numpy.core.', 'numpy._core.') if hasattr(np, '_core') else module
        import importlib
        return getattr(importlib.import_module(mod), name)


def load_pkl(path):
    with open(path, 'rb') as fh:
        return _Unpickler(fh).load()


def patch_tgb():
    import tgb.utils.utils as u
    u.load_pkl = load_pkl
    import tgb.linkproppred.negative_sampler as ns
    if hasattr(ns, 'load_pkl'):
        ns.load_pkl = load_pkl
