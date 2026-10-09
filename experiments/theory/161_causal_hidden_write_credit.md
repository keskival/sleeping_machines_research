# 161 — Hidden writes need a causal likelihood, and adjoints supply a measured approximation

9 October 2026. B1/R1 route-credit integration; mathematical development while
curie completes B2 and AWS owns the pairwise-credit pilots. This note does not
replace an inference architecture or report a fitted quality result.

## The bridge from output clocks to persistent memory

Note 160's pairwise learner can learn the odds between two output causes from
their exact likelihoods. A hidden write is different: its consequence is a new
state, which changes later clocks. Giving a write the output clock's
responsibility does not give it the responsibility for that changed state.

Fix a causal observed prefix and its memory z. At the current observed input,
route i has normalized prior π_i and writes v_i at the input's timestamp. Before
the next observation, the memory becomes z_i = A(Δ)(z + v_i). The decoder defines
a normalized next-observation law L_i(y) from that state, including the gap,
mark and recording-cell likelihood. The legal hidden-write score is

    s_i = log π_i + log L_i(y).
    ρ_i = exp(s_i) / Σ_j exp(s_j).

The prior and write use only the prefix. The outcome y enters the learning-time
posterior, never the inference-time prior or address. Keys select the write;
values carry its consequence. Every candidate likelihood is evaluated from
the same prefix state, with the candidate write forced. This is a stochastic
route model: a deterministic inference policy requires its own quality check.

For a realized next-observation likelihood, differentiation gives

    ∇ log Σ_i π_i L_i = Σ_i ρ_i ∇(log π_i + log L_i).

Exact pair targets sigmoid(s_j − s_i) teach a credit model the correct local
posterior odds. An independence-MH step using these scores corrects proposals
at fixed forward parameters. Note 160's finite-step and moving-target tracking
conditions still apply. A learned proposal cannot replace the causal score.

The restriction matters: this diagnostic writes at a known observed timestamp.
If internal route arrival times are latent and route-dependent, they belong in
the cause and score too; their likelihood must be integrated or sampled and
charged. Fixing them silently would remove temporal computation from the test.

## The dual memory gives a cheap first-order score, with a price for curvature

Let ℓ(z) be the next-observation log-likelihood from the reference state, and
let λ be its adjoint. A forced write changes the target state by δ_i = A(Δ)v_i:

    log L_i ≈ ℓ(z_ref) + Re(λ* δ_i).

The decay and conjugate rotation of note 160 transport this score backward
exactly for the linear memory. The approximation enters at the nonlinear
likelihood, and at any nonlinear state-dependent future processing. If the
Hessian norm of ℓ is bounded by B throughout the segment to each alternative,

    |log L_i − ℓ(z_ref) − Re(λ* δ_i)| ≤ B ||δ_i||² / 2.

For a decaying-rotating mode, ||A(Δ)v_i|| = exp(−rΔ)||v_i||, so its certified
remainder shrinks as exp(−2rΔ). Long delays make this linear credit approximation
more accurate when the assumptions hold; they also shrink the useful signal.
A hard downstream routing change can break the smooth-segment assumption.
Do not use this bound across such a change without evaluating the branch.

This connects two existing mechanisms: the backward memory carries the exact
linear sensitivity, and pairwise forward interventions measure the nonlinear
surprise. The next model can learn residual odds rather than re-learning the
entire transported sensitivity. Whether that saves fitting work is a measured
development question, not a consequence of the algebra alone.

## An approximation certificate for the credit distribution

Let exact scores be s_i and approximate scores be t_i. Common shifts do not
change a posterior. Define ε as half the range of residuals s_i − t_i. Then

    TV(softmax(s), softmax(t)) ≤ tanh(ε/2).

Proof: the density ratio between the two distributions has maximum/minimum
ratio at most exp(2ε). Under normalization its worst possible total variation
is (sqrt(exp(2ε)) − 1)/(sqrt(exp(2ε)) + 1) = tanh(ε/2), attained by a two-point
distribution with exchanged masses. If each exact component gradient has norm
at most G, using the approximate posterior with those component gradients
changes their weighted mean by at most 2G tanh(ε/2). Approximating the component
gradients introduces a separate error; this certificate does not cover it.

MH acceptance against approximate scores has the approximate posterior as its
stationary law. Better proposal calibration or more MH steps cannot remove
this score error. Three development diagnostics must therefore stay separate:
pairwise prediction error, chain-tracking error, and forced-write score error.

## A local posterior is not a history posterior

With two observations and two writes, the first write's posterior is

    ρ(i | y₁,y₂) ∝ π_i L_i(y₁) Σ_j π(j | z_i,y₁) L_{ij}(y₂).

The next-event posterior drops the continuation factor. They coincide only
when that factor is constant across first writes. A route that stores a fact
for a distant query can have no next-event credit and substantial later credit.
The predecessor-message results of note 156 make this a concrete R1 test.

For deep persistent inference, retain causes/eligibilities across the credit
horizon or replay their counterfactual continuations. Evaluating the next event
alone is a declared truncated objective, not an exact full-history estimator.
Charge each replay, candidate key, value intervention, credit-model update and
optimizer visit. One MH step scores two causes; a constant number of causes
does not make either score independent of its continuation length.

## Next integrated test

Keep the causal R1 keyed predecessor message, temporal memory, separate keys
and values, hard selected writes and losing-write credit. At a single declared
write, compare exact forced-write likelihoods with adjoint-plus-residual scores
on the same development examples. Measure residual half-span, posterior TV,
gradient disagreement, retained-cause tracking and all intervention work.
Then compare next-event and delayed-query credit with a fixed credit horizon.
Use the existing full differentiation and local race-credit fits as saved
references. Admit a bounded integrated pilot after those contracts and the
AWS pairwise development gate; use fresh sources/tags and no benchmark test.

Numerical checker: `experiments/credit/hidden_write_math.py`. It checks a
normalized recording-cell exponential race, its adjoint, exact forced-write
MH stationarity, the sharp posterior bound, approximate-score bias, and the
next-event/full-horizon distinction. Queue execution and result packet pending.
