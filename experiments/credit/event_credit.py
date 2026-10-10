"""Independent residual audits; probabilities are chosen before draws.
Allocation risk estimates may be inaccurate: full support preserves expectation.
Vector contributions are already addressed; candidate discovery is caller-owned.
"""
import math


def probabilities(risks, costs, budget, floor=0.01):
    if not risks or len(risks)!=len(costs): raise ValueError("shape")
    if not 0 < floor <= 1 or not all(math.isfinite(x) and x>=0 for x in risks): raise ValueError("risk/floor")
    if not all(math.isfinite(c) and c>0 for c in costs): raise ValueError("cost")
    total=sum(costs)
    if not math.isfinite(budget) or budget < floor*total: raise ValueError("infeasible budget")
    if budget>=total:return [1.0]*len(costs)
    if not any(risks):return [floor]*len(costs)
    def at(mu): return [max(floor,min(1., math.sqrt(v/c/mu))) for v,c in zip(risks,costs)]
    lo,hi=0.,max(v/c for v,c in zip(risks,costs))/floor**2
    for _ in range(100):
        mid=(lo+hi)/2; p=at(mid)
        if sum(c*q for c,q in zip(costs,p))>budget:lo=mid
        else:hi=mid
    return at(hi)


def estimate(predictions, audited, inclusion):
    # audited maps only selected indices to exact vector contributions.
    if not predictions or len(predictions)!=len(inclusion):raise ValueError("shape")
    width=len(predictions[0])
    if not width or any(len(v)!=width for v in predictions):raise ValueError("vectors")
    if any(not math.isfinite(p) or not 0<p<=1 for p in inclusion):raise ValueError("probability")
    if any(not isinstance(i,int) or not 0<=i<len(predictions) or len(v)!=width for i,v in audited.items()):raise ValueError("audit")
    if any(not math.isfinite(x) for v in list(predictions)+list(audited.values()) for x in v):raise ValueError("nonfinite")
    result=[sum(v[k] for v in predictions) for k in range(width)]
    for i,actual in audited.items():
        for k in range(width):result[k]+=(actual[k]-predictions[i][k])/inclusion[i]
    return result
