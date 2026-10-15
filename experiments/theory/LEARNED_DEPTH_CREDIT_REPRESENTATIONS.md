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

### Distinguish missing information from missing learning

Depth-first extension,13October2026. Let F be all legitimately available
issued information, Z=f(F) the retained query representation, and G a later
credit-effect or projected-update label under its declared horizon/learner.
Let mu=E[G|F] and m=E[mu|Z]=E[G|Z]. For a fixed positive-semidefinite metric M,

    E||G-C(Z)||_M^2
      = E||G-mu||_M^2
        + E||mu-m||_M^2
        + E||m-C(Z)||_M^2.

Conditional expectation makes both cross terms zero. The terms are respectively
unpredictable feedback under F, information lost in Z, and imperfect learning
of the best predictor under Z. A larger critic or more labels can reduce the
third term; it cannot recover information that its input has erased. More
causal side-information can reduce the second term, but fetching/encoding it
costs work. Changing the prediction horizon changes the label and all three
terms. None is a measured number merely because the architecture is expressive.

The same decomposition holds when M is measurable from Z. If the response
metric depends on information erased by Z, the optimal predictor instead
satisfies E[M|Z] C*(Z)=E[M G|Z], with appropriate nullspace handling. Substituting
an ordinary conditional mean then changes the target. This is why supplied
signed actor responses and actual optimizer geometry can matter even when a
world forecast is well calibrated.

For the next shared native cut-query critic, retain target identity/coordinates,
current forward state/forecast, declared time/horizon, optimizer-response
features and learner-version compatibility information. Treat these as typed
observations processed by native temporal functions and addressed memory, not
as a mandate to read every actor parameter on every query. Shared route legs
suggest compositional reuse, but address/prefix similarity alone does not imply
small conditional effect distance. Learned consequences must establish transfer.
Unavailable future labels, hindsight-selected counterfactuals or hidden item
identities cannot be inserted into F. A supplied version tag also does not
magically summarize the changed operator.

Use the completed deep language quality/path diagnostics to choose the next
integration. A capacity intervention diagnoses the third term; a response-input
intervention diagnoses the second only under a matched learning comparison.
Current incoming-content cuts measure the actor's predictive dependence, not
the Bayes information floor of an unimplemented critic. Keep those conclusions
separate. The actualD8/D16/32M acquisitions continue; this derivation adds no
standalone fit or small-data quality veto.

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


## A concrete joint-learning objective across depth

At an issued parameter/state version, let K_(i,j) be a connected learned
transport from a downstream cut j to an upstream cut i. Retain the forward
packet X and the credit-query activation u_j. Use the same issued X/version
for local compositions and longer-range anchors. For i<k<j, train

    L_comp = E[||K_(i,j)(u_j;X)
                - K_(i,k)(K_(k,j)(u_j;X);X)||^2_Mi].

For nonlinear transports this is function composition, not multiplication of
constant matrices. Composition alone permits the all-zero solution and can
learn mutually consistent wrong credit. It must be paired with an observed
consequence objective and nonzero anchors, for example

    L_anchor = E[||K_(i,j)(u_j;X)-v_i||^2_Mi],
    L_future = E[loss_after_issued_update on subsequent causal observations].

Here v_i is a paid exact local/long-range response or an independently audited
response estimate for the declared objective. The future loss is measured
under the issued update; identifying its derivative needs differentiation
through the update, randomized interventions, or a declared estimator. Merely
regressing that loss does not produce a correct meta-gradient. Charge anchor
acquisition and all optimizer/representation learning.

Train shared parameters jointly with L_future + lambda L_comp + beta L_anchor,
then test whether long-range discrepancy falls as depth increases. Pairwise
consistency supplies a learning signal for errors that isolated local fitting
cannot see. It does not by itself prove a contraction or task-quality gain.

A stronger economical design represents normalized direction plus explicit
gain, and learns when to stop composition and request a longer-range anchor.
For a local error budget epsilon_j and amplification estimates kappa_l, the
upstream contribution is bounded by

    B_j = epsilon_j product_(l<j) kappa_l.

Allocate extra credit work to large B_j per unit acquisition cost, provided the
estimates are calibrated. Under positive-probability correction sampling,
record inclusion probabilities and bind corrections to issued versions. Do
not treat a confidence score as a rigorous bound without validation.

This gives a specific possible learning advantage: jointly learned coordinates
and multiscale anchors reduce epsilon_j and the amplification of representation
errors, while retaining actual task sensitivity as explicit gain. The decisive
comparison is independent local predictors versus this connected composition
at equal total work, across depth, with future prediction quality and correction
rates measured. The current two-layer lexical experiment is the precursor, not
that depth comparison.

## Expected error, coherent bias and learned cancellation

The worst-case recurrence is not the only useful depth criterion. For linear
issued transports, write B_j=T_hat_0 ... T_hat_(j-1) and the local discrepancy
xi_j=(T_hat_j-T_j)c_(j+1), including the terminal term as another contribution.
At the issued packet/version,

    e_0 = sum_j B_j xi_j.

When B_j are fixed under the stated conditional distribution, with local means
m_j and covariance blocks Gamma_(i,j),

    E||e_0||^2 = ||sum_j B_j m_j||^2
                 + sum_(i,j) tr(B_i Gamma_(i,j) B_j^T).

Condition on all B_j before using this expression; unconditional multiplication
of marginal covariances is invalid when transports and errors are correlated.
For nonlinear credit functions, the local linearization adds its declared
remainder and does not give a globally exact matrix identity.

With bounded B_j, zero conditional bias and independent errors, squared error
can grow linearly with depth (RMS as sqrt(depth)), even though a norm worst-case
bound grows linearly. Coherent bias can instead yield squared-depth growth.
Joint downstream learning can reduce aligned bias and cross-layer covariance,
or learn cancellation of local errors. Independent local MSE training does
not optimize these transported cross terms. The relevant training target is
end-to-end consequence risk plus calibrated multiscale anchors, not forcing
every local error to look small in an arbitrary coordinate system.

Cancellation is useful only if it generalizes to issued future packets. A
consistent but unanchored network may manufacture cancellation on training
packets. Measure local errors, amplification, end-to-end bias/covariance and
fresh prediction/work together. This is a second mathematical route to better
depth scaling, in addition to learning compatible stable coordinates; neither
route is established by the current two-layer lexical experiments.
