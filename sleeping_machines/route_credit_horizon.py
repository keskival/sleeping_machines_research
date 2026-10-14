"""Bounded forward consequence horizons across detached factual windows."""
from .depth_credit_allocator import DepthCreditAllocator


def credit_endpoint(token, length, window, windows=1):
    if not (0 <= token < length and window >= 1 and windows >= 1):
        raise ValueError('Valid causal site and positive window counts required')
    return min(length, (token//window + windows)*window)


class HorizonDepthCreditAllocator(DepthCreditAllocator):
    """Use the actual suffix length in the TRAIN allocation proxy."""
    def __init__(self, *args, consequence_windows=1, **kwargs):
        super().__init__(*args, **kwargs)
        if consequence_windows < 1:
            raise ValueError('Positive consequence horizon required')
        self.consequence_windows=consequence_windows

    def observe(self, probabilities, alternative, proposal_probability, difference):
        if self.last is None:
            raise RuntimeError('Outcome arrived without an issued audit')
        token,length=self.last['token'],self.last['length']
        old=credit_endpoint(token,length,self.window)
        new=credit_endpoint(token,length,self.window,self.consequence_windows)
        # Base observe multiplies by the old suffix fraction. Correct its proxy
        # only; the actor receives the original extended-horizon loss label.
        super().observe(probabilities,alternative,proposal_probability,
                        difference*((new-token)/(old-token)))

    def packet(self):
        result=super().packet()
        if self.consequence_windows!=1:
            result['consequence_windows']=self.consequence_windows
            result['scope']+=' Proxy uses actual extended suffix weight; raw consequence labels reach across factual detach boundaries within the optimizer chunk.'
        return result
