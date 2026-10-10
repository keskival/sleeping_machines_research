# 163 — Correct the detached producer, not an invented latent route

Curie, R1/B1 credit enabler. Complements notes 110, 139, 145, 160–162.
Objective: repair the actual local-credit learning gap while preserving the
native inference model, its temporal computation and its predictive loss.

## The concrete failure and the exact graph boundary

`recall_tpp_v4.py` local mode detaches the backbone state entering **both**
the key map and the query map, and detaches the predecessor embedding entering
the key map. Those maps and their decaying memory traces still learn. The
backbone and embedding lose those receiver-to-producer gradient paths.
Message-only recall is 77.4 ± 5.4% with local credit versus 97.5 ± 1.7% with
full differentiation, three seeds. Normalized local credit is 90.0 ± 5.5% at
eight pairs and 41.9 ± 32.4% at 16. These are completed task measurements;
they do not by themselves identify which missing gradient causes the gap.

The current addressed writes are deterministic. Note 162's optional-route
correction cannot repair a missing continuous producer derivative by itself.
The correct next contract is therefore over the existing detached inputs.

Let U_j(θ) contain the backbone state and predecessor message supplied at
event j. Replace those inputs to the key/query receivers by independent
leaves u_j, while preserving the same numerical forward values. Define

    L_cut(θ,u) = the unchanged native loss with these receiver inputs cut,
    g_local = ∂_θ L_cut,       a_j = ∂_{u_j} L_cut,
    J_j = dU_j/dθ.

Here θ includes every native parameter. The read maps' own parameter gradients,
clock gradients, value-memory gradients and all uncut backbone paths are
already in g_local. The repair must not add them again. The chain rule gives

    g_full = g_local + Σ_j J_jᵀ a_j.                         (1)

In this R1 construction, keyed memory and queries do not feed the backbone
or predecessor-embedding producers. These producer nodes form an antichain:
none descends from another cut producer node. Shared parameters and shared
backbone computation are allowed; they do not invalidate (1).

## Forward predicted cotangents plus sparse residual audits

Freeze a credit predictor before the audit draw. It predicts â_j for each
cut input from retained forward information, optionally paired with an
outcome feature available at learning time. Detach these predictions from
the actor objective. An auxiliary predictor has its own training objective.

Prepare the baseline

    g_0 = g_local + Σ_j J_jᵀ â_j.

Independently include each producer audit with B_j ~ Bernoulli(p_j), p_j > 0.
The corrected update is

    ĝ = g_0 + Σ_j (B_j/p_j) J_jᵀ(a_j − â_j).              (2)

Conditional on the factual graph, parameter version, predictor and propensities,

    E_B[ĝ] = g_full,
    tr Cov_B(ĝ) = Σ_j (1/p_j − 1) ||J_jᵀ(a_j − â_j)||².  (3)

Prediction error affects variance, rather than the raw-gradient mean. Improving
a predictor by â'_j = α a_j + (1−α)â_j multiplies audit variance by
(1−α)². Exact cotangents give zero audit variance. These are identities at
fixed weights, not claims that a learned predictor attains those cotangents.

Parameter sharing is fully retained in J_j. The covariance cross terms vanish
because the inclusion variables are independent, not because the producer
gradients are orthogonal. A fixed-size or correlated audit scheduler needs its
actual inclusion covariance. Using estimated propensities instead of the
probabilities actually used can bias the correction.

Zero probability is legal only for a certified zero **residual parameter
gradient**, including its baseline treatment. An unaudited low predicted
surprise is not such a certificate. Predictor updates from the current audit
must affect the next draw; fitting it and recomputing the current baseline
breaks the conditioning argument.

## Temporal transport supplies the receiver cotangent

For the unnormalized keyed memory, the addressed key state is

    S_t = D_t S_{t−1} + address(mark_t) k_t.

The true write cotangent is the sum of future receiver cotangents transported
by D_{j+1}ᵀ … D_tᵀ back to the original addressed slot. This preserves actual
elapsed time and the complete declared credit horizon. Key normalization
adds its real normalization Jacobian; query normalization adds its own.
The key/query maps then transport these cotangents to the detached backbone
and predecessor inputs. This is the reciprocal temporal geometry already
present in the family, with the missing producer boundary made explicit.

In the current small R1 implementation, local receiver differentiation can
already supply the exact cut cotangents for all queries. The expensive part
being audited is the deep producer VJP. In a large asynchronous implementation,
retaining those cotangents may itself require replay or eligibility packets.
Every receiver, replay, packet and producer cost belongs in the account.

## Do not double-count overlapping recurrent cuts

For a feedback architecture, cutting all u_j makes later producer functions
depend on earlier cut values. Equation (1) still holds with **cut-graph**
cotangents and total producer sensitivities. It is incorrect to multiply full
original-graph adjoints by total producer Jacobians at overlapping nodes.

Witness: x₁ = θ, x₂ = θ + x₁, L = x₁ + x₂. The derivative is 3. Total producer
Jacobians are (1,2), cut-graph cotangents (1,1), giving 3. Original-graph
adjoints are (2,1); pairing those with the total Jacobians gives the wrong 4.
Alternatively, original adjoints paired with local parameter Jacobians (1,1)
give 3. The two valid decompositions cannot be mixed.

This is why a learned scalar backward recursion cannot stand in for an
arbitrary producer Jacobian. Native contracts must name the cut graph and
the sensitivity convention, especially before adding memory feedback.

## Learning work, allocation and parameter versions

For independent audits with additive costs c_j, the fixed-state variance
optimum is p_j = clip_floor,1(||J_jᵀ(a_j−â_j)|| / sqrt(λ c_j)). This is the
same independent-inclusion allocation derived in note 162. Norm estimates
can guide allocation with positive support; exact residual norms require
paying for their discovery. Shared-prefix or batched VJP execution needs
its actual cost model rather than an additive fiction.

A dense baseline VJP through the entire backbone may already cost as much as
full differentiation. Correcting it adds work. Savings require an economical
forward eligibility/packet baseline or measured reuse; they do not follow
from sampling fewer sites. Dense eligibility has P×r storage, and nonlinear
producer sensitivities cannot simply be replaced by diagonal decay. The native
contract computes and charges genuine VJPs; it is a correctness instrument,
not a sparse runtime or efficiency result.

Retain the factual parameter version with delayed packets and audits. A correct
old-version gradient is stale at changed weights. Clipping and Adam also change
the update distribution: an unbiased raw gradient is not an unbiased adaptive
parameter step. Measure those effects separately, as note 139 requires.

## Evidence, attribution and the next owned comparison

Two stdlib proof tests pass: independent residual means/covariance and
quadratic improvement with shared parameter vectors; the overlapping-cut
double-counting witness. `check_continuous_producer_credit.py` is a queued
native contract, not a completed numerical result. It uses the actual
two-layer R1 class with plain and normalized keys, exposes precisely the
existing local cuts, and compares every parameter against full differentiation.
All 64 independent audit outcomes are enumerated on a fixed six-event witness.
No fitting or benchmark data is used. Numerical outcome is pending.

Predicting future cotangents is established synthetic-gradient work:
[Jaderberg et al., Decoupled Neural Interfaces](https://arxiv.org/abs/1608.05343).
Learned control variates with unbiased corrections are also established:
[Grathwohl et al., Backpropagation through the Void](https://arxiv.org/abs/1711.00123).
The contribution here is the exact detached-input decomposition for the native
model, its full-support residual correction, temporal interface and graph-cut
contract. It does not claim invention of synthetic gradients or importance
correction, or global convergence of the coupled learner.

After the native contract, measure a same-model smoke before admitting a
bounded R1 DEV fit. Compare full differentiation, both existing local designs
and corrected producer credit on the same data and loss. Keep the eight-,
16- and 32-pair DEV errors together; use no TEST score for development. Record
actual parameter-gradient variance, optimizer update behavior, packet storage,
receiver/producer/replay/critic work, wall time and RSS. Preserve the current
temporal core, clocks, silence likelihood, addressed keys/values and predecessor
message. This repairs a learning path rather than substituting an architecture.
