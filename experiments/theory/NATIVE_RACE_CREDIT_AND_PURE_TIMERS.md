# Native race credit: coupled value/time learning and pure-timer variance

15 October2026. Formal derivation, quadrature and numerical contracts; no fit
or changed forward architecture. The current native learning job stays unchanged.

## Keep the native clock and value teacher together

Given prefix-fixed rates lambda_i=exp(s_i), independent exponential clocks give
winner J with p_i=lambda_i/Lambda and time T~Exp(Lambda), independent of J.
The native bounded delay is delta(T)=.001+.010*T/(1+T). This is standard
exponential-race algebra. Values v_i below are fixed conditional on the prefix.

For loss L(v,delta)=v*a(delta)+b(delta), the actual TemporalRoute backward has
both a conserved counterfactual value term and the winner's interior time term.
After averaging J at fixed T, its score-i component is

lambda_i*T*(v_i-vbar)*a(delta)
 - p_i*T*delta'(T)*[v_i*a'(delta)+b'(delta)],

where vbar=sum_i p_i v_i. Integration by parts under Exp(Lambda) gives
Lambda E[T a(delta(T))] = E[a(delta(T))] + E[T delta'(T) a'(delta(T))].
Combining the two native terms gives exactly the derivative of
vbar E[a(delta(T))] + E[b(delta(T))]. The bounded delay's derivative matters.
Thus replacing the winner's timing gradient alone by a smoother formula while
leaving the value teacher unchanged can break their compensation.

`test_native_clock_credit_integral.py` integrates the actual native backward,
not a reimplementation, over64-point Gauss-Laguerre quadrature and enumerates
winners. It agrees with the risk derivative to1e-11 for the affine-value case
with nonlinear bounded-time dependence. A quadratic-value case exhibits the
remaining nonlinear surrogate gap. This preserves a positive property of the
native mechanism without pretending it is globally exact.

## An exact hazard-score audit covers nonlinear/discontinuous consequences

The joint density is p(J=j,T=t)=lambda_j exp(-Lambda*t). Therefore
partial_s_i log p = 1[J=i]-lambda_i*T.
A full downstream loss, with the prefix fixed, has an unbiased local hazard-score
estimator [1[J=i]-lambda_i*T]*(L-b), where b is prefix-predictable. This needs no
pathwise derivative through changed queue ordering. If all alternative suffix
losses L_j(T) are available at the same sampled T, averaging J yields
p_i*[L_i(T)-Lambda*T*sum_j p_j L_j(T)].

The quadrature contract verifies this joint estimator for the nonlinear case
where the native local surrogate differs. It is an audit identity, not an
implemented replacement for full-model learning. Variance, predictable baselines,
shared upstream parameters and interacting stochastic nodes need an explicit
stochastic-computation-graph treatment. A naive add-on to existing native
backpropagation can double count timing/score paths. Whole-event replay also
needs correct endogenous-stream and publication semantics.

## Pure emission timers permit a lean exact improvement

Our indexed object's arm/fire operation currently uses the winning time but
not the winning clock identity to choose emitted content: all winners materialize
the same published memory at delta(T). Conditional on prefix and T, its delay
adjoint c(T) is independent of J. The native timing gradient is
-c(T)*delta'(T)*T*1[J=i].
Its conditional expectation over J is
-c(T)*delta'(T)*T*p_i.
This is Rao-Blackwellization: same expectation, no conditional winner variance.
The remaining variance from T/prefix/future computation persists. It is invalid
if the winner selects different content, a different state write or other
winner-dependent consequences. Do not use it for native receipt/value races.

`pure_race_timer.py` implements this one operation. It samples exactly the same
native exponentials and returns bitwise-identical time/winner under the same
seed. Only backward changes. Contracts verify native forward parity, conditional
mean equality, positive winner variance removed, and finite timing learning for
all candidate rates. No inference arithmetic is added to the forward.

A separate frozen-model diagnostic, conditional on a completed native fit, can
measure gradient variance at its terminal pure readout timer without refitting
or changing any receipt race. Integrating it into the next native fit is justified
only with its pure-timer scope preserved. No prediction-quality or hardware
latency/energy gain is inferred from a variance identity.
