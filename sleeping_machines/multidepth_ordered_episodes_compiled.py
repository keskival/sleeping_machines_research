"""Ordered-content token episodes with optional compiled layer steps.

Same interception as multidepth_ordered_episodes (explicit per-token content values
replace F.embedding inside the eager episode loop). Compilation wraps only the
per-layer sparse counterfactual step, which performs no content lookup, so the
interception and its event-order checks are unchanged. The original module stays
byte-identical because completed results bind its hash.
"""
from types import FunctionType
from .multidepth_ordered_episodes import ContentFunctions, original, ordered_content


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
