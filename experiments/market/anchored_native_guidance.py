"""Future-taught native credit augments, rather than removes, exact factual BP.

The projected-minus-own direction is an update proposal, not an unbiased
replacement gradient. Actual subsequent Adam forecast utility trains emit/hold.
"""
import inspect

def anchored_stream(original):
    source=inspect.getsource(original)
    anchor='''        if acquisition:
            exact_end, grads, exact_scores = window(model, copy.deepcopy(old), t, m, k, group, 'bptt')
            assert max(abs(a-b) for a, b in zip(scores, exact_scores)) < 1e-12
        else:
            grads = projected if q else own'''
    replacement='''        # Retain ALL within-window factual credit. The optional proposal is
        # an extra direction selected by actual forward-loss consequences.
        exact_end, factual, exact_scores = window(model, copy.deepcopy(old), t, m, k, group, 'bptt')
        assert max(abs(a-b) for a, b in zip(scores, exact_scores)) < 1e-12
        if projected is not None:
            projected = [base + remote - local for base, remote, local in zip(factual, projected, own)]
        own = factual
        grads = factual if acquisition else (projected if q else factual)'''
    if source.count(anchor)!=1:raise ValueError('Native guidance source contract changed')
    namespace=dict(original.__globals__);exec(compile(source.replace(anchor,replacement),'<factual-anchored native guidance>','exec'),namespace);return namespace[original.__name__]
