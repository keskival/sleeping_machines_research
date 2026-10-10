"""One-route full-support prediction plus residual audit contribution.
Caller owns proposal support, lazy discovery, route semantics and exact replay.
"""
import math

def estimate(route_probability,audit_probability,prediction,exact=None):
    for p in [route_probability,audit_probability]:
        if not math.isfinite(p) or not 0<p<=1:raise ValueError('strictly positive propensity required')
    if not prediction or any(not math.isfinite(v) for v in prediction):raise ValueError('finite nonempty prediction')
    if exact is not None and (len(exact)!=len(prediction) or any(not math.isfinite(v) for v in exact)):raise ValueError('exact contribution shape/values')
    if exact is None:return [h/route_probability for h in prediction]
    return [(h+(g-h)/audit_probability)/route_probability for h,g in zip(prediction,exact)]
