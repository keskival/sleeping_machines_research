# Emission timing has an interior derivative and an order boundary

15 October 2026. Formal audit selected by generated child-memory emissions.
Not a replacement teacher and not authorization for another fit.

## The concrete missing learning signal

The coalesced construction arms multiple child memories before draining their
outputs. Arrival order can change a parent's nonlinear state. Native transported
values and pure-timer interior gradients differentiate computation at the
observed order; they do not differentiate a discrete order change. Averaging
pure clock winners reduces variance of the interior estimator but cannot create
this missing signal. This is a specific learning problem, not a reason to remove
native timing or make all child updates synchronous.

Let a child timer have raw waiting time T~Exp(lambda). Its bounded publication
delay is delta(T)=.001+.010 T/(1+T), strictly increasing. Fix the other emitted
packet's raw waiting time s. When both use this same delay transform, ordering
changes at T=s. Let L(t) be the suffix loss with all other randomness fixed,
piecewise differentiable, and let Delta_j=L(b_j+)-L(b_j-) at finite boundaries.
Assume integrable derivatives, finite jumps and no extra moving boundaries.

For u=log(lambda), the exact rate derivative is

d E[L(T)]/du = E[-T partial_t L(T)] - sum_j b_j f_T(b_j) Delta_j,

plus any direct parameter derivative of the suffix with times held fixed.
The first term is the ordinary interior path derivative. The second accounts
for probability mass crossing each order boundary. This follows by integrating
partial_u f=(1-lambda*t)f against L and integrating by parts separately between
jumps. It is not an additional correction to apply blindly to an existing
coupled value/time teacher. Its assumptions fix all other randomness and
thresholds independently of this local log rate.

For a pure first-arrival loss L=1[T<s], the interior term is zero while the
boundary term is lambda*s*exp(-lambda*s)>0. For independent second timer
S~Exp(mu), integrating gives lambda*mu/(lambda+mu)^2, the derivative of
P(T<S)=lambda/(lambda+mu). Thus an order-sensitive task can have exactly zero
interior timing credit despite a nonzero exact learning signal.

The hazard score and piecewise integration-by-parts identities are standard probability tools. The architectural question is where to attach them to native memory emissions without duplicating existing credit.

## Exact audit and candidate engineering route

The same derivative also equals E[(1-lambda*T)(L-b)] for a baseline b independent
of this timer draw. This hazard-score estimator covers arbitrary suffix changes
but may have large variance. For fixed competing threshold s, replaying the two
orderings with coupled downstream randomness gives the conditional boundary
contribution lambda*s*exp(-lambda*s)*(L_first-L_second), when each ordering's
local loss is constant in T. With continuous time-dependent values, retain the
interior term and evaluate the limiting suffix losses at the boundary instead.

For a pure timer with aggregate lambda=sum_k exp(score_k), multiply the local
log-lambda derivative by pi_k=exp(score_k)/lambda to reach each candidate score.
This differs from the categorical identity chosen inside a value-routing race.
Those races keep their coupled native teacher; the new audit does not authorize
adding a second likelihood term that double-counts their timing/value paths.

Actual queue boundaries include multiple siblings, stale versions, internal
races and time-dependent views. An exact local audit must hold a causal prefix,
identify the changed timer(s), specify a coupling for downstream randomness,
retain all direct derivatives, and charge suffix replays. A nearest-neighbour
swap heuristic is not automatically an unbiased global teacher. First compare
interior and hazard-score gradients against an exact small integrated parent
fixture; only then design a variance/cost-controlled native teacher. The current
composition fit stays source-bound and unchanged.
