# Economical causal credit must track learning memory, not only event decay

R1/B1,11 October2026. The current integrated route learner scores eight future
supervised targets. This is a declared finite-window objective. A temporal key
can become silent while the optimizer changes it caused continue to affect
predictions; key decay alone cannot certify a short credit horizon.

## A two-channel calculation

Let u be a route-induced addressed-state difference and v the difference in a
learning state. Around one issued trajectory consider the linear propagation

    u_(j+1) = rho u_j,
    v_(j+1) = a v_j + b u_j,
    d_j = c_u u_j + c_v v_j.

Start u_0=delta and v_0=0. Then u_j=rho^j delta, and

    v_j = b delta (a^j-rho^j)/(a-rho),       a != rho,
    v_j = b delta j a^(j-1),               a = rho.

With a=1 and |rho|<1, the addressed difference tends to zero but the learning
difference tends to b delta/(1-rho). Thus sparse silent state need not mean
zero downstream learning influence. This is an exact linear model, not a
claim that Adam or the current native model has a=1. The complete nonlinear
state must include parameters, moments, addressed values, clocks and any
controller memory required for the future transition.

## A conditional tail bound

After H observed steps, suppose two branch states differ by delta_H in a chosen
norm. Assume common future observations and a fixed continuation policy whose
state transition is alpha-Lipschitz and whose per-step loss is L-Lipschitz.
For stochastic state-dependent future routing use a coupling/transition bound
in Wasserstein distance instead; forcing the same actions is not a bound for
policies that choose different future actions. If discount gamma satisfies
0<=gamma<=1 and gamma*alpha<1, the omitted discounted contrast is bounded by

    |sum_(j>=H) gamma^j d_j| <= gamma^H L delta_H/(1-gamma alpha).

Proof: future state distances are at most alpha^k delta_H; apply the loss bound
and sum the geometric series. A finite N-step horizon uses the finite geometric
sum, so it also applies when gamma alpha>=1. With gamma=1 and alpha=1 the
finite tail bound is L delta_H(N-H); there is no infinite summable certificate.
Changing optimizer metrics/state versions requires the corresponding transition
factors. A small measured residual does not supply a uniform alpha guarantee.

For a fixed issued policy score s_A, discarding the tail changes its conditional
route-gradient contribution by at most

    E[||s_A|| gamma^H L delta_H(A)/(1-gamma alpha) | issued history].

This describes one issued route's downstream objective. It does not turn a
local eight-target controller update into the total gradient through every
later actor update. A full sequential score-function gradient contains the
later decision scores as well; online changing actor versions need explicit
semantics and recorded activations.

## Consequence for asynchronous resource allocation

Track two influence channels: decaying physical/message effect and transported
learning/optimizer effect. Allocate delayed credit work according to the
score-weighted tail uncertainty and expected task benefit per paid operation.
When a certified bound falls below an allocated error tolerance, retirement is
justified for that objective. Without a certificate, sparse delayed residual
observations with recorded positive selection probability can correct the tail;
the estimator must charge the continuation, normalization, variance and stored
snapshot work. Silence by itself is not a retirement rule.

The practical next diagnostic is branch-state discrepancy beyond eight targets,
with direct-only and independently learning continuations, on development data.
Measure which channel persists before paying for uniformly longer windows.
The current fixed-window result remains evidence about its stated objective;
no theorem here supplies a whole-model scaling or Adam contraction claim.
