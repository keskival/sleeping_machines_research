# Closed-loop learning of intervention, acquisition and credit

User-directed analytical development,10 October2026. B1/R1 shared enabler;
no new queue, fit, benchmark claim or machine-checked theorem. Proofs below
are algebraic under explicit assumptions. Extends
[learned exploration utility](LEARNED_EXPLORATION_UTILITY.md),
[unseen-route credit](UNSEEN_ROUTE_CREDIT.md),
[economical event credit](ECONOMICAL_EVENT_CREDIT.md) and
[coupled stability](166_coupled_credit_convergence_and_storage.md).

## 1. One augmented decision process includes its own learning

Let X_t=(S_t,theta_t,psi_t,O_t,Z_t,D_t,v_t,t) contain forward persistent state,
actor and estimator weights, optimizer state, connected credit state, retained
counterfactual packets, versions and time. Include enough environmental
history/belief to make the transition Markov; otherwise use the full causal
history as X. This is a mathematical state, not a requirement to materialize
all its coordinates in every message.

An action U_t=(A_t,E_t,M_t) specifies a forward intervention, counterfactual
acquisition, and learning/update decision. E=0 buys no new counterfactual.
The transition X_{t+1}=T(X_t,U_t,xi_t) includes the actual updates of theta,
psi,O,Z,D. Exogenous randomness xi has a declared law. Set

    r_t = - loss_t - lambda work_t,
    J(pi) = E_pi[sum_{t>=0} gamma^t r_t],  0<gamma<1.

Assume bounded reward for infinite-horizon results; a finite horizon uses
terminal value instead. Count work for proposals, descriptions, counterfactuals,
predictor/critic fitting, optimizer, transport, storage and reads. Unbounded
real losses require integrability/drift hypotheses, not the sup-norm claims below.
A resource-constrained formulation maximizes expected prediction return subject
to expected work <=B; lambda is a dual price. Strong duality is not automatic
for arbitrary neural policy classes. A stochastic dual update is
lambda <- [lambda + beta(work-B)]_+, with normalized budgets matching the return.

This objective already includes improvement of future learning: no separate
"trainer of the trainer's trainer" is present in the definition. Improvements
that change subsequent policies are represented by the transition, not erased
by a frozen-forward continuation.

## 2. Value and acquisition close through Bellman equations

For a fixed joint policy pi, define

    V^pi(x)=E_pi[sum gamma^k r_{t+k}|X_t=x],
    Q^pi(x,u)=E[r_t+gamma V^pi(X_{t+1})|x,do(u)].

Then V^pi(x)=E_{u~pi}[Q^pi(x,u)]. Its Bellman operator B_pi is a gamma
contraction in sup norm: |B_pi V-B_pi W| <= gamma ||V-W||_infinity. Thus
there is a unique bounded value, and iterating a function is enough;
there is no need for a new value model at each recursion depth.
The optimality operator max_u E[r+gamma V(next)] is also a contraction
when the supremum is well-defined and bounded. Computing that maximizer can
be expensive; a learned policy approximates it with a separately measurable
optimization gap. Contraction does not prove neural fitting or discovery succeeds.

For a probe p returning packet Z_p, the general acquisition advantage is

    A_acq(x,p) = Q^pi(x,E=p)-Q^pi(x,E=0).

Both branches include their own subsequent learning and actions. Their costs,
latencies and effects on state belong to their transitions. A direct intervention
uses the corresponding Q difference. A learned acquisition function predicts
A_acq, and a gate/proposal/budget policy learns to act on it.

A transparent decision-only specialization: after a probe, choose intervention a
with conditional utility m_a(H,Z)=E[U_a|H,Z]. With the same feasible actions,
no physical disruption, correct conditional beliefs and cost c_p,

    VOI(H,p) = E_Z[max_a m_a(H,Z)] - max_a E_Z[m_a(H,Z)] - c_p.

The first two terms have nonnegative difference because max E <= E max.
This proves that information has nonnegative gross optimal decision value in
this specialization. It does not prove arbitrary probing helps a bounded,
miscalibrated learner; compute/time may also alter the environment.

**Example.** Hidden Z is +/-1 with equal probabilities. Intervention a in
{-1,+1} pays aZ. Without information, optimal expected payoff is0. A binary
probe predicts Z correctly with probability p>=1/2, so choosing its sign pays
2p-1 in expectation. Its net acquisition value is2p-1-c. A learned gate can
learn p from causal packets and acquire only when this advantage is positive.
Current intervention utility alone cannot reveal that acquisition value.

## 3. Exact delayed supervision without paired futures at every event

For a policy parameter phi with initial-state law independent of phi, assume
all phi-dependent transition choices are represented by known differentiable
stochastic action densities pi_phi. Reward/transition conditional on x,u have
no additional explicit phi dependence. Then

    grad_phi J = E[sum_t gamma^t grad_phi log pi_phi(U_t|X_t) G_t],
    G_t = sum_{k>=t} gamma^(k-t) r_k.

Proof: differentiate trajectory density, then remove rewards preceding each
action because its conditional score has mean zero. The return includes all
later learning effects, so no differentiation through the update stack is
needed for this estimator. Its variance may be large. Deterministic
phi-dependent writes, initialization or updates not represented by those
stochastic choices require additional pathwise derivatives; silently dropping
these terms changes the objective. Randomizing update actions gives one
possible finite-level closure, not a guarantee of inexpensive optimization.

For a causally fixed, action-independent baseline b_t(X_t), replace G_t by
G_t-b_t without changing the mean. Existing later factual losses supply G;
no exact no-probe twin trajectory is required. A learned no-probe value is a
baseline, not an observed individual counterfactual.

A learned Q can replace G in the policy update only approximately unless it
is conditionally correct or residual correction is retained. If
|Qhat-Q|<=epsilon uniformly, the absolute bias of its score update is bounded
by epsilon E||score|| for the single decision. A greedy action from Qhat has
regret <=2epsilon: compare its estimated maximizer with the true maximizer
and use two error terms. These bounds need actual uniform bounds; calibration
MSE does not imply them.

## 4. Counterfactual feedstock and supported acquisition

A packet is (causal prefix, route descriptor, action, alternative state/message,
observed local/future response, versions, discovery propensity, work). Reuse
one packet for route value, future update utility, acquisition value and credit
prediction. Its generation is charged once; additional fitting/reads are charged.
Immediate loss differences are features/partial labels for long-horizon value,
not interchangeable with complete future-advantage labels.

For a single-decision observational dataset with behavior q(u|x)>0 and
correctly logged propensity, let Y be a realized declared continuation return
with E[Y|x,u]=Q(x,u). For target pi and fixed prediction m(x,u),

    Vhat = sum_u pi(u|x)m(x,u)
           + pi(U|x)/q(U|x) * (Y-m(x,U)),  U~q,

has conditional mean sum_u pi(u|x)Q(x,u), regardless of m. Proof: the residual
expectation cancels the baseline's prediction error term by term. Exact
normalization/summation is charged; sample that first term independently from
pi when dense enumeration is expensive. A packet's continuation must match
the declared target continuation. Changed sequential policies need sequential
importance correction or another justified estimator; a local route propensity
alone does not correct all future actions. Positivity, finite moments and
causally frozen predictions/propensities are essential.

## 5. A finite family of predictors can estimate its own learning credit

Index a finite set of trainable modules b=1,...,m: forward actor, intervention
policy, acquisition policy, value/response models and credit models. Let g_b
be a declared exact gradient/contribution target for module b's loss. That
loss must be explicit: task return, audited response regression, TD
semi-gradient regression, etc.; these are not automatically gradients of one
common scalar objective. At a causal/versioned boundary H choose a prediction
h_b and audit probability a_b>0. With conditionally independent B_b~Bernoulli(a_b),

    ghat_b = h_b + B_b/a_b * (g_b-h_b).

Conditionally on H,g_b, E[ghat_b]=g_b and
E||ghat_b-g_b||^2=(1/a_b-1)||g_b-h_b||^2. Predictions may come from the same
finite shared credit system used for the actor. Thus credit-model gradients
can themselves be predicted, with occasional exact derivatives of their
specified finite local learning losses as anchors. No separate learned
supervisor is mathematically required for each b.

The proof does not require h_b to be independent of previous history, other
modules or old labels. It does require no fitting on the present audit followed
by recomputation of its supposedly pre-audit prediction. Feedback available
before the audit may condition H. Nonlinear clipping/Adam of ghat need not have
the same mean as the exact update; state that objective boundary explicitly.

If g_b is itself computed from an approximate bootstrapped label, the correction
is exact for that declared training target, not for an unknown true infinite
horizon gradient. Bellman/adjoint closure addresses that target error separately.
A rare exact long replay remains potentially expensive; probability alone does
not make its worst-case work disappear. Local anchored training losses and
observed-return policy gradients offer closure without routinely constructing
that long exact meta-gradient.

## 6. Cotangent closure: reusing the current forward/backward graph

For smooth state transitions x'=T(x,u,xi), known fixed noise law and differentiable
pi(u|x) with state-independent support, derivative interchange yields

    lambda(x) = E_u,xi[r_x + gamma T_x^T lambda(x')
                         + Q(x,u) grad_x log pi(u|x)],
    lambda(x)=grad_x V^pi(x).

This follows by differentiating V=E_pi[r+gamma V(T)]. The final score term is
required when the future action distribution depends on state. Random
transitions with state-dependent densities need their score terms too.
For hard routing this is a distributional expected-return derivative;
it is not a proof of the derivative through a fixed-noise discontinuous race.

Once the Q term is provided, lambda satisfies an affine fixed-point equation
lambda=b+K lambda. If sup||T_x||<=kappa and gamma*kappa<1, K is a contraction,

    lambda = sum_{n>=0} K^n b.

Finite-horizon backward recurrence is exact with its declared terminal
cotangent without this contraction assumption. A connected trained estimator
amortizes the conditional continuation cotangent. Existing local VJPs supply
T_x^T times a predicted/revealed cotangent. Higher-order gradients, alternative
paths and optimizer-state transport are additional operations, not automatic
outputs of a first-order scan backend.

If ||Qhat-Q||<=epsilon_Q, ||score_x||<=s and the learned cotangent has fixed-point
residual ||lambdahat-(bhat+K lambdahat)||<=delta, then

    ||lambdahat-lambda|| <= (delta+s*epsilon_Q)/(1-gamma*kappa).

Proof: subtract both equations, bound the inhomogeneous Q error by s*epsilon_Q,
and move the contraction term to the left. Slow/noncontractive modes can
require longer retained credit or explicit audits; a learned function does
not establish this inequality without checking its premises.

## 7. Producer-weighted teaching and local intervention utility

Let A be the H-measurable local producer map from a cotangent g to its
parameter update contribution. The useful calibration loss is

    E[||A(g-h(H))||^2].

For square-integrable g, h*=E[g|H] minimizes it, up to A's nullspace. Orthogonal
conditional expectation proves the risk decomposition. If A is not known
at H, the optimum solves E[A^T A|H]h=E[A^T A g|H] instead; plain conditional
mean need not be optimal. Audited squared-error fitting uses B/a to preserve
the population objective under adaptive supported sampling.

If a probe changes a local update by d, a later smooth loss F has

    F(theta+d)-F(theta) = grad F(theta)^T d + R,
    |R| <= L||d||^2/2

when F is L-smooth along that segment. Existing matching-version cotangents
may supply this dot product. Intervening learner/state evolution changes F
and d's transport; include it in the augmented derivative or label the local
approximation. Reused factual backward information is therefore useful but
not automatically an exact intervention return.

For mean-preserving unbiased gradients with locally fixed preconditioner P,
the expected first-order effect is unchanged. Under a quadratic/local Hessian
approximation, variance contributes

    eta^2/2 * tr(H P Cov(ghat) P^T).

Residual-risk minimization is a useful surrogate; indefinite curvature,
nonlinear optimizers and distribution drift make actual future advantage a
separate measurement. Predicting credit accurately in producer-irrelevant
directions can leave task utility unchanged.

## 8. Reciprocal forward/credit closure: no third learned pass

The user's intended closure is stronger than merely limiting a trainer stack:
forward events teach the backward process, and backward events teach the forward
process. Both are trainable parts of the same persistent event substrate. The
primitive observations/local algebra below are operations within those events,
not a separately trained teacher or another compulsory temporal pass.

Let a forward transition y=F_theta(x) have local Jacobian J=D_x F_theta.
A separate backward map predicts B lambda, where lambda is an output cotangent
and B should approximate J^T. A forward tangent probe v yields z=Jv. Reciprocity
requires the scalar identity

    v^T B lambda = lambda^T z.

The right side is supplied by the forward transition, so define

    delta = v^T B lambda - lambda^T z,
    L_recip = delta^2/2,
    B <- B - beta delta v lambda^T.

This update is a local outer product of the probe, received cotangent and scalar
residual. It is the exact derivative of this explicit local teaching loss;
there is no third backward network needed to teach B. Its packets can travel
with the same addressed forward/credit events. Computing a forward tangent
or finite-difference probe is still paid work. Finite differences teach a
secant rather than the exact Jacobian unless their approximation error is bounded.

**Directional identification theorem.** At fixed J, draw v and lambda
independently, with E[vv^T]=I and E[lambda lambda^T]=I and finite fourth moments.
Then

    E[L_recip] = ||B-J^T||_F^2/2,
    E[delta v lambda^T] = B-J^T.

Proof: write D=B-J^T, delta=v^T D lambda; use the two independent covariance
identities in the quadratic and outer product expectations. Hence local forward
probes identify the backward transport without forming the full Jacobian. A
rank-deficient probe covariance identifies only its observed subspace; supported
discovery is necessary for omitted directions, just as for omitted routes.
This supervises local transport, not the unknown future cotangent itself.

**Exact coordinate-probe contraction.** For B in R^(n by m), choose i,j uniformly
and independently, v=sqrt(n)e_i, lambda=sqrt(m)e_j. The update changes D_ij by
factor 1-beta*n*m, leaving other entries fixed. Therefore

    E[||D_next||_F^2 | D]
      = (1-2 beta+beta^2 n m)||D||_F^2.

It contracts for 0<beta<2/(n m). This is an explicit reciprocal learning law
with a finite teaching circuit and a checkable step-size condition. It is not
an efficient dense-matrix implementation claim: parameter storage, probe
computation, discovered coordinates and model capacity are still charged.
Context-dependent nonlinear B_psi(x,lambda) can fit the same scalar loss;
its local parameter derivative is delta * D_psi(v^T B_psi(x,lambda)). If that
module needs nonlocal temporal parameter credit, the existing backward process
may supply it with supported corrections. It does not justify another hierarchy
of backward models or assert that all its derivatives are the outer product above.

**Moving-target reciprocal update.** For a linear forward W with backward
B in the opposite orientation, suppose the endpoints reuse the same local
forward update U and both apply decay:

    W_next=(1-alpha)W+U,
    B_next=(1-alpha)B+U^T.

Then B_next-W_next^T=(1-alpha)(B-W^T), exactly for every realized U, even when U
was produced using B. For 0<alpha<2 the alignment discrepancy contracts. This
is the mirrored-update construction already in note160 section11 (known
Kolen-Pollack-type principle); the identity is not a new general learning
convergence theorem. Exact replay of the update, including the applicable
optimizer/decay semantics, is required. Nonlinear state Jacobians need the
probe/tracking construction above or another justified reciprocal law.

For future cotangents, the observed output loss derivative supplies the boundary
message, and learned backward transport supplies a continuation estimate.
Forward tangent probes/counterfactual packets teach transport and response;
later factual losses teach predicted utility; reciprocal credit teaches actor,
intervention and acquisition parameters. The adjoint fixed point and finite
coupling condition bound this recurrent loop under their stated hypotheses.
They do not turn local Jacobian alignment into global model optimality.

Thus closure is architectural: two mutually teaching directions, multiple
shared estimator functions if useful, and primitive local observations/updates
inside the same event graph. A policy/value/credit function is not a new pass
merely because it has a different prediction target. Count the actual extra
operations, retained records and events rather than the names of estimators.

## 9. Coupled stability closes a finite graph, not an infinite stack

Let e_t>=0 collect value error, credit error, acquisition/intervention decision
error and tracking error of moving targets. Suppose a declared local training
regime admits the componentwise comparison

    E[e_{t+1}|history] <= A e_t + epsilon,

where A is a fixed nonnegative finite matrix including all cross-module gains,
and epsilon bounds noise/approximation/drift forcing. If rho(A)<1,

    E[e_t] <= A^t E[e_0] + sum_{k=0}^{t-1} A^k epsilon,
    limsup E[e_t] <= (I-A)^(-1) epsilon.

Proof: induction; the nonnegative matrix geometric series converges. Equivalently,
choose w>0,q<1 with A w<=q w and use the weighted max norm. For epsilon=0,
errors contract to zero. This generalizes note166's two-module inequality.
It is a sufficient condition under measured/established comparison constants,
not a theorem that arbitrary mutually trained neural estimators converge.
Discounted Bellman closure, contractive credit transport and sufficiently fast
tracking are ways to obtain some of these diagonal/coupling bounds.

An exactly self-consistent but ungrounded predictor can be wrong: training
h2 to copy h1 and h1 to copy h2 admits every common constant. Observed task
rewards, supported intervention outcomes and exact local derivatives remove
that free-floating supervision. The closure has a finite set of learned
functions plus a finite set of primitive anchors. It does not require every
operation, including the primitive observation and local derivative, to have
a learned teacher.

## 10. Economics: where another trained estimator earns its place

For independent residual audits r_b=g_b-h_b, auditing costs c_b and allowed
variance V, minimize sum c_b a_b subject to
sum (1/a_b-1)||r_b||^2<=V. The interior optimum is

    a_b = min(1, sqrt(mu) ||r_b||/sqrt(c_b)),

with mu chosen for the constraint (and any required support floor). Differentiate
the Lagrangian to obtain c_b-mu||r_b||^2/a_b^2=0. Residuals are not known before
auditing; learn their predictable conditional squared norms. For predictable
allocation minimizing expected conditional variance, replace ||r_b||^2 by
E[||r_b||^2|H]. Correlated/slate selections require covariance terms.

A finite ledger is

    C_total = C_forward + C_selected_local_derivatives + C_policy_proposal
              + C_predictor_read_and_fit + C_transport_and_storage
              + sum_b a_b C_exact_audit,b + C_utility_checks.

The same paid audit may label several estimators if their actual targets are
available from it. This avoids repeated generation, not repeated target
computation for free. Adopt an estimator when it improves measured future
quality/work or reduces the audited objective's residual enough to pay the
complete overhead at matching fidelity. There is no general theorem of
"almost free" meta-learning; shared packets, sparse support and amortized
continuation are specific mechanisms to test.

## 11. Executable research consequence

The reciprocal design is: one joint intervention/acquisition policy; shared
persistent value/response/utility functions; one connected credit system
usable for every module; sampled local derivative/return anchors; and
versioned packets/propensities. Gate -> lazy proposal -> probe -> reuse packet
-> intervene/update -> issued error -> credit/value correction -> next gate.

First analytical/numerical contracts should check support, expected estimator
means, finite graph tracking bounds and repeated packet causality. Native
TRAIN/DEV pilots then compare uniform/no-extra/learned acquisition at equal
complete work, including delayed task benefit and drift. No queue is admitted
by this note, and no current route generalization or efficient learning result
is inferred from these sufficient conditions. The existing temporal races,
addressed persistent values, separate keys, optional-route credit and
silence-aware supervision remain the target substrate.
