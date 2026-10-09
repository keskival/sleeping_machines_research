"""Epoch-boundary recovery; no changes to model, batches or optimizer updates."""
import os
import random
from pathlib import Path

import numpy as np
import torch


def save_state(path, *, model, optimizer, ema, metadata, **progress):
    state = dict(metadata=metadata, model=model.state_dict(), optimizer=optimizer.state_dict(),
                 ema=None if ema is None else ema.state_dict(), python_rng=random.getstate(),
                 numpy_rng=np.random.get_state(), torch_rng=torch.get_rng_state(), **progress)
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    with tmp.open('wb') as stream:
        torch.save(state, stream)
        stream.flush()
        os.fsync(stream.fileno())
    tmp.replace(path)


def load_state(path, *, model, optimizer, ema, metadata):
    state = torch.load(path, map_location='cpu', weights_only=False)
    if state['metadata'] != metadata:
        raise ValueError('Recovery source, data or protocol differs; use a new run tag')
    if (ema is None) != (state['ema'] is None):
        raise ValueError('Recovery EMA configuration differs')
    model.load_state_dict(state['model'])
    optimizer.load_state_dict(state['optimizer'])
    if ema is not None:
        ema.load_state_dict(state['ema'])
    random.setstate(state['python_rng'])
    np.random.set_state(state['numpy_rng'])
    torch.set_rng_state(state['torch_rng'])
    return state
