"""Ordered-content wrapper (as multidepth_ordered_episodes) around the residual race stack; compiled steps allowed."""
from types import FunctionType
from .multidepth_ordered_episodes import ContentFunctions, ordered_content
from .residual_race_episodes import token_features as original


def token_features(*args, token_content, **kwargs):
    tokens = args[1]
    if len(token_content) != tokens.shape[1]: raise ValueError('One content tensor per observed token required')
    functions = ContentFunctions(tokens, token_content)
    namespace = dict(original.__globals__); namespace['F'] = functions
    episode = FunctionType(original.__code__, namespace, original.__name__, original.__defaults__, original.__closure__)
    episode.__kwdefaults__ = original.__kwdefaults__
    result = episode(*args, **kwargs)
    if functions.index != tokens.shape[1]: raise ValueError('Observed content lookups missing')
    return result
