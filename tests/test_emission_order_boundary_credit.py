"""Deterministic integral witnesses; no fitted model or Monte Carlo claim."""
import math
import numpy as np


def integrate_exp(function,rate):
    x,w=np.polynomial.laguerre.laggauss(64)
    return float(np.dot(w,function(x/rate)))


def test_exact_order_gradient_exists_when_interior_derivative_is_zero():
    rate,other=1.7,.8
    expected=rate*other/(rate+other)**2
    # Conditional boundary correction averaged over the competing timer.
    boundary=integrate_exp(lambda s:rate*s*np.exp(-rate*s),other)
    # Same derivative via the exact joint hazard score, integrating the
    # probability that the other timer exceeds this one.
    score=integrate_exp(lambda t:(1-rate*t)*np.exp(-other*t),rate)
    h=1e-5
    prob=lambda u:math.exp(u)/(math.exp(u)+other)
    finite=(prob(math.log(rate)+h)-prob(math.log(rate)-h))/(2*h)
    assert abs(boundary-expected)<1e-12
    assert abs(score-expected)<1e-12
    assert abs(finite-expected)<1e-10
    assert expected>0  # piecewise constant loss has zero interior derivative


def test_continuous_interior_plus_jump_matches_exact_risk_derivative():
    rate,other,c=2.1,.7,.3
    # L(t,s)=c*t + 1[t<s]. E[L]=c/rate + rate/(rate+other).
    interior=-c/rate
    boundary=integrate_exp(lambda s:rate*s*np.exp(-rate*s),other)
    exact=-c/rate+rate*other/(rate+other)**2
    score=integrate_exp(lambda t:(1-rate*t)*(c*t+np.exp(-other*t)),rate)
    assert abs(interior+boundary-exact)<1e-12
    assert abs(score-exact)<1e-12
