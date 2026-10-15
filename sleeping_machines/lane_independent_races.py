"""Versioned eager native episodes with independent per-lane race tapes.

Preserves each lane's exponential-race and sampled-loser laws. Changes coupling
between lanes, not the declared one-lane gradient policy. Existing frozen kernels
and jobs remain untouched. This is a candidate learning backend, not a quality win.
"""
import inspect
from types import FunctionType

import torch

from . import sparse_counterfactual_episodes as episodes
from . import multidepth_counterfactual_episodes as multidepth
from . import sparse_counterfactual_layer as layer_module
from .ordered_token_episodes import ContentFunctions


def _noise_for_lanes(value, lanes, heads, receivers):
    if not isinstance(value, torch.Tensor):
        raise TypeError('Race noise must be a tensor')
    if value.shape == (heads, receivers):
        return value.unsqueeze(0)
    if value.shape == (lanes, heads, receivers):
        return value
    raise ValueError('Race noise must have [head,receiver] or [lane,head,receiver] shape')


def _make_layer():
    source = inspect.getsource(layer_module.sparse_counterfactual_layer)
    for anchor in ('noise[None] / s64.exp()', 'alt_noise[None] / pi.detach()'):
        if source.count(anchor) != 1:
            raise RuntimeError('Native race layer source changed: ' + anchor)
    source = source.replace('noise[None] / s64.exp()', '_noise_for_lanes(noise,n,H,U) / s64.exp()')
    source = source.replace('alt_noise[None] / pi.detach()', '_noise_for_lanes(alt_noise,n,H,U) / pi.detach()')
    namespace = dict(layer_module.sparse_counterfactual_layer.__globals__)
    namespace['_noise_for_lanes'] = _noise_for_lanes
    exec(compile(source, '<lane-independent native race layer>', 'exec'), namespace)
    return namespace['sparse_counterfactual_layer']


sparse_counterfactual_layer = _make_layer()


def _make_episode(original):
    source = inspect.getsource(original)
    edits = {
        'torch.stack([torch.empty(U, dtype=torch.float64).exponential_(generator=generator) for _ in range(H)])':
        'torch.stack([torch.empty(n,U, dtype=torch.float64).exponential_(generator=generator) for _ in range(H)],dim=1)',
        'torch.empty(H,U,dtype=torch.float64).exponential_(generator=alternative_generator)':
        'torch.empty(n,H,U,dtype=torch.float64).exponential_(generator=alternative_generator)',
    }
    for anchor, replacement in edits.items():
        if source.count(anchor) != 1:
            raise RuntimeError('Native episode source changed: ' + anchor)
        source = source.replace(anchor, replacement)
    namespace = dict(original.__globals__)
    namespace['sparse_counterfactual_layer'] = sparse_counterfactual_layer
    exec(compile(source, '<lane-independent native episode>', 'exec'), namespace)
    return namespace['token_features']


_episode = _make_episode(episodes.token_features)
_all_depth_episode = _make_episode(multidepth.token_features)


def _run_episode(episode, model, tokens, state=None, generator=None, *, token_content=None, **kwargs):
    """Native eager CPU episode; checkpoint both explicit generators as usual.

    Chunk partitioning preserves the tape for fixed lane shape/order. Changing
    lane count/order does not promise the same random realization. Deterministic
    mode retains the original API's RNG-consumption convention. No global patch.
    """
    if kwargs.get('compiled', False):
        raise ValueError('Independent-lane backend is eager CPU only')
    if token_content is not None:
        if len(token_content) != tokens.shape[1]:
            raise ValueError('One content tensor per observed token required')
        functions = ContentFunctions(tokens, token_content)
        namespace = dict(episode.__globals__)
        namespace['F'] = functions
        episode = FunctionType(episode.__code__, namespace, episode.__name__, episode.__defaults__)
    result = episode(model, tokens, state, generator, **kwargs)
    if token_content is not None and functions.index != tokens.shape[1]:
        raise RuntimeError('Observed content lookups missing')
    return result


def token_features(model, tokens, state=None, generator=None, **kwargs):
    return _run_episode(_episode, model, tokens, state, generator, **kwargs)


def all_depth_token_features(model, tokens, state=None, generator=None, **kwargs):
    return _run_episode(_all_depth_episode, model, tokens, state, generator, **kwargs)


def independent_forward(original):
    """Return a private forward function for supported native language models.

    Use in a new, explicitly versioned model subclass; never mutate a running
    model or module global. Existing replay/force/suppress paths use the same
    adapted forward. Protected parent/private-branch streams require a separate
    adapter and are deliberately rejected. Checkpoints must record this policy.
    """
    helper = original.__globals__.get('token_features')
    if 'token_features' not in original.__code__.co_names or helper is None:
        raise ValueError('Forward does not expose a native episode backend')
    name = helper.__module__
    if name in ('sleeping_machines.sparse_counterfactual_episodes',
                'sleeping_machines.ordered_token_episodes'):
        replacement = token_features
    elif name in ('sleeping_machines.multidepth_counterfactual_episodes',
                  'sleeping_machines.multidepth_ordered_episodes'):
        replacement = all_depth_token_features
    else:
        raise ValueError('Unsupported forward episode policy: ' + name)
    namespace = dict(original.__globals__)
    namespace['token_features'] = replacement
    result = FunctionType(original.__code__, namespace, original.__name__,
                          original.__defaults__, original.__closure__)
    result.__kwdefaults__ = original.__kwdefaults__
    return result
