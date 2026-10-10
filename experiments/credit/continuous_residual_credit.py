"""R1 continuous producer-credit identity, finite independent-audit checks."""
import itertools
from math import prod


def estimate(baseline, corrections, probabilities, selected):
    if len(corrections)!=len(probabilities) or len(selected)!=len(probabilities):
        raise ValueError('One propensity and selection per producer site required')
    if any(not 0<p<=1 for p in probabilities):
        raise ValueError('Positive supported audit probabilities required')
    result=list(baseline)
    for correction,p,bit in zip(corrections,probabilities,selected):
        if bit:
            result=[g+c/p for g,c in zip(result,correction)]
    return result


def enumerate_audits(baseline, corrections, probabilities, target):
    mean=[0.]*len(target);variance=0.
    for bits in itertools.product((0,1),repeat=len(probabilities)):
        weight=prod(p if bit else 1-p for p,bit in zip(probabilities,bits))
        value=estimate(baseline,corrections,probabilities,bits)
        mean=[m+weight*g for m,g in zip(mean,value)]
        variance+=weight*sum((g-t)**2 for g,t in zip(value,target))
    formula=sum((1/p-1)*sum(g*g for g in correction) for correction,p in zip(corrections,probabilities))
    return dict(maximum_mean_error=max(abs(g-t) for g,t in zip(mean,target)),
                variance_trace=variance,declared_variance_trace=formula,
                expected_audits=sum(probabilities),audit_draws=2**len(probabilities))
