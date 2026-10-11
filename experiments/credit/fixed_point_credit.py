"""Matrix-free issued-version cotangent solver; explicit residual/error budget.
Requires a valid caller-supplied Euclidean contraction bound. No learned teacher.
"""
from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class CreditSolution:
    vector: np.ndarray
    context: str
    version: int
    operator_calls: int
    residual_norm: float
    error_bound: float
    converged: bool

def solve_credit(source,transport,*,context,version,contraction,tolerance=1e-9,max_calls=128,initial=None):
    b=np.asarray(source,dtype=float)
    if b.ndim!=1 or not np.isfinite(b).all() or not 0<=contraction<1 or tolerance<=0 or not np.isfinite(tolerance) or max_calls<1:raise ValueError('invalid credit solve')
    # A warm candidate is explicitly revalidated under the CURRENT operator.
    x=np.zeros_like(b) if initial is None else np.array(initial,dtype=float,copy=True)
    if x.shape!=b.shape or not np.isfinite(x).all():raise ValueError('invalid initial credit')
    for calls in range(1,max_calls+1):
        k=np.asarray(transport(x,context=context,version=version),dtype=float)
        if k.shape!=b.shape or not np.isfinite(k).all():raise ValueError('invalid transport response')
        r=b+k-x;norm=float(np.linalg.norm(r));bound=norm/(1-contraction)
        if bound<=tolerance or calls==max_calls:
            x.setflags(write=False)
            return CreditSolution(x,context,version,calls,norm,bound,bound<=tolerance)
        x=x+r
