"""Keep the prior total clock scale stable as addressed capacity grows.

Relative race preferences, RNG draws, sparse writes and the existing temporal
credit rule are retained. The scalar scale changes computational waiting time.
"""
import math
from types import FunctionType
from .sparse_counterfactual_episodes import token_features as original_features
from .sparse_counterfactual_layer import sparse_counterfactual_layer as original_layer


def calibrated_layer(*args, reference_pool=4, **kwargs):
    if not math.isfinite(reference_pool) or reference_pool <= 0:
        raise ValueError('Positive finite reference pool required')
    memory=args[2] if len(args)>2 else kwargs['m']
    scale=memory.shape[2]/reference_pool
    if scale==1:return original_layer(*args,**kwargs)
    if len(args)>7:
        args=list(args);args[7]=args[7]*scale
    else:
        kwargs=dict(kwargs);kwargs['noise']=kwargs['noise']*scale
    return original_layer(*args,**kwargs)


def token_features(*args, clock_reference_pool=4, **kwargs):
    if kwargs.get('compiled',False) or (len(args)>9 and args[9]):
        raise ValueError('Capacity-clock adapter currently uses the eager CPU path')
    def layer(*values,**options):
        return calibrated_layer(*values,reference_pool=clock_reference_pool,**options)
    namespace=dict(original_features.__globals__)
    namespace['sparse_counterfactual_layer']=layer
    function=FunctionType(original_features.__code__,namespace,original_features.__name__,
        original_features.__defaults__,original_features.__closure__)
    function.__kwdefaults__=original_features.__kwdefaults__
    return function(*args,**kwargs)
