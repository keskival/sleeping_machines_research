"""B1/R1 predictable audit allocation and observation-anchored risk fitting.
Independent reusable primitive; not wired into active actor/critic sources.
"""
from dataclasses import dataclass
import math
import numpy as np


def allocate(probabilities, risks, costs, budget, *, floor=0.02):
    pi,v,c=[np.asarray(x,dtype=float) for x in (probabilities,risks,costs)]
    if pi.ndim!=1 or not (pi.shape==v.shape==c.shape):raise ValueError('shape mismatch')
    if not all(np.isfinite(x).all() for x in (pi,v,c)) or np.any(pi<=0) or np.any(v<0) or np.any(c<=0):raise ValueError('invalid allocation inputs')
    if abs(float(pi.sum())-1)>1e-10 or not 0<floor<=1:raise ValueError('invalid probabilities/floor')
    minimum=float(pi@c)*floor;maximum=float(pi@c)
    if not math.isfinite(budget) or budget<minimum-1e-12 or budget>maximum+1e-12:raise ValueError('infeasible budget')
    if budget<=minimum:return np.full_like(pi,floor)
    if budget>=maximum:return np.ones_like(pi)
    def rates(scale):return np.clip(scale*np.sqrt(v/c),floor,1)
    # Zero estimated risk can stay at its support floor; unused budget is allowed.
    hi=1.
    while float(pi@(c*rates(hi)))<budget:
        if np.all((v==0)|(rates(hi)==1)):return rates(hi)
        hi*=2
    lo=0.
    for _ in range(100):
        mid=(lo+hi)/2
        if float(pi@(c*rates(mid)))>budget:hi=mid
        else:lo=mid
    return rates(lo)


@dataclass(frozen=True)
class IssuedRiskPacket:
    features: tuple
    prediction: float
    propensity: float
    predictor_version: int


class ResidualRiskLearner:
    """Softplus linear conditional second moment, trained only by audited residuals.

    Issued features/predictions bind the gradient. Delayed consumption applies
    that old-version gradient; it is not asserted to be the current gradient.
    """
    def __init__(self,dimension):
        self.weights=np.zeros(dimension,dtype=float);self.version=0
    def issue(self,features,propensity):
        f=np.asarray(features,dtype=float)
        if f.shape!=self.weights.shape or not np.isfinite(f).all() or not 0<propensity<=1:raise ValueError('invalid issued inputs')
        z=float(self.weights@f);prediction=float(np.logaddexp(0,z))
        return IssuedRiskPacket(tuple(f),prediction,float(propensity),self.version)
    def gradient(self,packet,*,audited,residual_squared=None):
        f=np.asarray(packet.features,dtype=float)
        if f.shape!=self.weights.shape:raise ValueError('dimension mismatch')
        if not audited:
            if residual_squared is not None:raise ValueError('unaudited packet cannot reveal residual')
            return np.zeros_like(f)
        if residual_squared is None or not math.isfinite(residual_squared) or residual_squared<0:raise ValueError('invalid observed residual')
        # sigmoid(z)=1-exp(-softplus(z)); packet carries the issued derivative.
        slope=-math.expm1(-packet.prediction)
        return ((packet.prediction-residual_squared)*slope/packet.propensity)*f
    def consume(self,packet,*,audited,residual_squared=None,step=0.01):
        if not math.isfinite(step) or step<=0:raise ValueError('invalid step')
        g=self.gradient(packet,audited=audited,residual_squared=residual_squared)
        updated=self.weights-step*g
        if not np.isfinite(updated).all():raise ValueError('nonfinite risk update')
        self.weights=updated;self.version+=1
        return g
