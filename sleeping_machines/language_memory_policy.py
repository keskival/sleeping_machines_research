"""Explicit persistent token-memory policy; EOS remains an observed token.

Rebind a private forward namespace so factual and counterfactual worlds share
one policy without editing a frozen driver or mutating its module globals.
"""
from types import FunctionType

def persistent_forward(original):
    helper=original.__globals__.get('token_features')
    if helper is None or 'token_features' not in original.__code__.co_names:raise ValueError('Native token episode interface required')
    def features(*args,**kwargs):
        kwargs['eos']=None
        return helper(*args,**kwargs)
    namespace=dict(original.__globals__);namespace['token_features']=features
    result=FunctionType(original.__code__,namespace,original.__name__,original.__defaults__,original.__closure__);result.__kwdefaults__=original.__kwdefaults__;return result
