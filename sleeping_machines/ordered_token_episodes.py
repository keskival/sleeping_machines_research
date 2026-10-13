"""Reuse the unchanged episode function with explicitly supplied token values."""
from types import FunctionType
from torch.nn import functional as F
from .sparse_counterfactual_episodes import token_features as original
from .ordered_token_lookup import ordered_content


class ContentFunctions:
    def __init__(self,tokens,values):self.tokens,self.values,self.index=tokens,values,0
    def __getattr__(self,name):return getattr(F,name)
    def embedding(self,indices,weight):
        k=self.index;self.index+=1
        if not indices.equal(self.tokens[:,k]):raise ValueError('Content lookup event order changed')
        return self.values[k]


def token_features(*args,token_content,**kwargs):
    if kwargs.get('compiled',False):raise ValueError('Ordered content prototype requires eager CPU execution')
    tokens=args[1]
    if len(token_content)!=tokens.shape[1]:raise ValueError('One content tensor per observed token required')
    functions=ContentFunctions(tokens,token_content)
    namespace=dict(original.__globals__);namespace['F']=functions
    episode=FunctionType(original.__code__,namespace,original.__name__,original.__defaults__,original.__closure__)
    episode.__kwdefaults__=original.__kwdefaults__
    result=episode(*args,**kwargs)
    if functions.index!=tokens.shape[1]:raise ValueError('Observed content lookups missing')
    return result
