# Task-relevant credit compression and jointly learned depth transport

R1/B1,11 October2026. The prospective two-layer token transport result motivates
this formal target; it does not demonstrate deep-model scaling or an optimal
bit representation.

## What is compressed

Let H be issued history, a an optional route and C(a) its later task cost.
A learned representation Z=E_psi(H,forward activations,optimizer state) supports
h_phi(a,Z). The target is influence on future loss/learning, not reconstruction
of the full history or imitation of a fixed backpropagated gradient.

A useful compression objective is score-weighted prediction risk plus paid
representation and work budgets:

    E[(C(A)-h(A,Z))^2 ||s_A||^2] + lambda Bits(Z,versions,state)
        + mu Work(encoding,discovery,projection,learning,correction).

Count parameters, moments, retained activations, address identities, clocks and
precision. A64-dimensional numerical state is not64bits. Model weights may
encode reusable structure across many messages, but their residency/training
cost remains part of the boundary. No generic neural-network theorem makes
this representation sufficient or near-optimal.

For a fixed representation, the scalar weighted-risk minimizer is

    h*(a,Z) = E[w C(a)|a,Z]/E[w|a,Z],  w=||s_a||^2,

where the denominator is positive. When w is known from(a,Z), this reduces to
E[C(a)|a,Z]. This is an optimal projection for the stated risk/information,
not perfect prediction of an unknowable future. Conditional gradient covariance
also subtracts the squared residual mean and includes future variation; risk
is a useful bound/objective, not identical to every variance criterion.

## How errors accumulate through depth

For compatible layer representations, true transport T_l and learned transport
T_hat_l satisfy

    e_l = T_hat_l e_(l+1) + (T_hat_l-T_l)c_(l+1).

In declared compatible norms, suppose the amplification is at most kappa_l and
the local discrepancy at most epsilon_l. Then

    ||e_l|| <= kappa_l ||e_(l+1)||+epsilon_l,
    ||e_0|| <= product_l kappa_l ||e_L||
               + sum_j epsilon_j product_(l<j) kappa_l.

Uniform kappa<1 bounds the additive sum by epsilon/(1-kappa); kappa=1 gives
linear growth; kappa>1 can give geometric growth. These are conditional bounds.
Changing metrics needs inter-metric factors. Normalizing a tensor while hiding
its gain does not remove underlying sensitivity or preserve credit. Keep
legitimate gains and charge their numerical precision/dynamic range.

## The learning aspect

Train a connected credit representation/transport jointly across layers and
time against later task consequences. Adjacent interfaces can learn compatible
coordinates, rather than accumulate errors from isolated modules. Shared
forward activations supply learned feature geometry, observed loss and state
updates. Subsequent forward observations train the retained credit/optimizer
activations; versions and causal/reset boundaries stay explicit.

Three complementary objectives are useful: downstream task/value prediction,
multiscale composition consistency, and direct longer-range anchors. Compare
short composed transports with longer predicted/observed consequences, then
correct discrepancies. Local exact tangents/cotangents are consistency
instruments where paid; they need not be the sole learning target. A scalar
critic output does not constrain the capacity of its connected function.

Adaptive correction allocates computation where score-weighted uncertainty or
amplification is high. Full actual-return residual correction preserves issued
route-policy expectation when the common predictor is independent of sampled A.
It does not establish exact derivatives for every deterministic native weight.
A bootstrap prefix objective is not a full-return objective without the required
later observations/correction. General asynchronous delivery needs recorded
versions and measured staleness effects.

## Current state and decisive depth experiment

One fixed trained native two-layer source currently supplies data-dependent
geometry to transported64-dimensional alternative lexical Adam learners.
Eight fresh48-target TRAIN prefixes pass prospective accuracy; the acquisition
pipeline is2.09x faster than exhaustive alternatives. Fixed connected-controller
conditional gradient noise is77.8% below static credit on four earlier prefixes.
No full-carrier or deep-model prediction/learning-cost superiority follows.

The next depth study should train owned integrated members at increasing depth,
with equal token/data protocols and complete resource accounting. Report local
and end-to-end transport error, amplification, retained state, exact correction
frequency, task quality and total work together. Compare independently learned
local transport with shared/joint connected transport and long-range anchors.
Keep capacity, selected activity, candidate discovery and learning work separate.
Numerical contracts at depths2/4/8 check identity, not useful deep-model learning.
