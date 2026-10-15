"""Use the actual actor's clock-head law in isolated streaming compositions.

Old episode helpers explicitly called RaceTPP.clocks. A repaired log-scale head
must not receive the old positive sigma as its log_sigma argument. Rebind only
an isolated function namespace, leaving source and actor parameters untouched.
"""
from types import FunctionType
from experiments.market import episode_state
class ActorOwnedHead:
    @staticmethod
    def clocks(model,*args):return model.clocks(*args)
def consumer():
    fn=episode_state.consume;namespace=dict(fn.__globals__);namespace['RaceTPP']=ActorOwnedHead
    result=FunctionType(fn.__code__,namespace,fn.__name__,fn.__defaults__,fn.__closure__);result.__kwdefaults__=fn.__kwdefaults__;return result
