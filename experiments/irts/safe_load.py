#!/usr/bin/env python3
"""Load Raindrop .npy files (pickled NumPy object arrays) through an allow-list unpickler.

Only NumPy array reconstruction and builtin container/scalar types are permitted; any other global referenced by the
pickle raises instead of executing. Members are read straight from the release zip, never extracted to disk.
"""
import io, pickle, zipfile
import numpy as np

ALLOWED = {
    ('numpy.core.multiarray', '_reconstruct'), ('numpy._core.multiarray', '_reconstruct'),
    ('numpy.core.multiarray', 'scalar'), ('numpy._core.multiarray', 'scalar'),
    ('numpy', 'ndarray'), ('numpy', 'dtype'),
    ('builtins', 'dict'), ('builtins', 'list'), ('builtins', 'tuple'), ('builtins', 'set'),
    ('builtins', 'int'), ('builtins', 'float'), ('builtins', 'str'), ('builtins', 'bytes'), ('builtins', 'bool'),
    ('collections', 'OrderedDict'), ('_codecs', 'encode'),
}


class AllowListUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if (module, name) in ALLOWED:
            return super().find_class(module, name)
        raise pickle.UnpicklingError(f'blocked global {module}.{name}')


def load_npy(zip_path, member):
    with zipfile.ZipFile(zip_path) as z, z.open(member) as f:
        raw = f.read()
    buf = io.BytesIO(raw)
    version = np.lib.format.read_magic(buf)
    read = np.lib.format.read_array_header_1_0 if version == (1, 0) else np.lib.format.read_array_header_2_0
    shape, fortran, dtype = read(buf)
    if dtype.hasobject:
        return AllowListUnpickler(buf).load()
    return np.frombuffer(buf.read(), dtype=dtype).reshape(shape, order='F' if fortran else 'C')
