# Downstream representation credit and relational composition

15 October 2026. Development implementation and numerical contracts; no trained quality claim.

## Concrete failure and repair

The first spatial 32-block construction trains each route against an immediate
pooled linear class head. This teacher answers whether the block helps its own
classifier, not whether its output enables later nonlinear composition. Giving
all scales the same label also offers a shortcut toward redundant class codes.
The new `sleeping_machines/downstream_race_hierarchy.py` keeps 32 feature-race
blocks, separate keys/values, temporal input surfaces and hard races. It removes
intermediate class objectives and uses final-task counterfactual replay.

A transition now retains the arrangement-sensitive stride-2 linear path and adds
six learned pairwise products between four relative child roles. For children
z_0,...,z_3, parent = Conv(z) + a W concat_{i<j}[tanh(U_i z_i)*tanh(U_j z_j)].
The shared child features can be reused in multiple relational combinations.
The mixed finite difference in two different children can be nonzero; a linear
pool cannot represent that interaction. This is an expressivity result, not a
proof that learned roles correspond to fur, parts or objects. Relative positions
are fixed; variable object binding and equivariant pose remain engineering work.

## Exact conditional gate credit

Represent race sampling with mutually independent uniform random tapes U_r. A
site r is one group at one spatial position of one block. Conditional on its
prefix, its winner has probability pi_ri = softmax(s_r)_i. Let L_ri be the final
loss after forcing candidate i and recomputing the entire continuation with the
same other random tapes. Downstream scores and winners must be recomputed.

The gate term in the stochastic computation graph derivative is
G_r = sum_i (grad pi_ri) L_ri.
Ordinary factual pathwise differentiation supplies continuous selected-path
terms. Averaging G_r over prefixes and other tapes, then summing sites, supplies
the categorical terms; this follows by conditioning on each categorical draw.
Keeping old downstream winners fixed would instead differentiate a different
conditional objective.

For an independently sampled site R with full-support probability q_r, the
zero-forward surrogate
S_R = sum_i [pi_Ri - stop(pi_Ri)] stop(L_Ri-b_R) / q_R
has E_R[grad S_R] = sum_r G_r. The baseline b_R must be independent of candidate
i. The implementation enumerates all candidates at one uniformly sampled site,
q=1/N, using the candidate mean as baseline. The caller must draw the site
uniformly independently of race noise; cycling deterministically is not this
unbiased estimator. No importance clipping or division by N is hidden.

This is established categorical counterfactual credit applied to our deeper
spatial graph, not a novelty claim. It integrates the project's existing replay
and residual-audit direction. It evaluates the current frozen-parameter forward
computation, not hypothetical future optimizer updates or changed observations.

## Signal and resource consequences

A long chain can attenuate child influence even with a correct teacher. The
initial 32-block, width-4 fixture had zero float32 first-route loss contrast;
float64 resolved nonzero final-loss key gradients. This numerical failure is
useful evidence about signal transmission at initialization, not a trained
architecture verdict. Increasing precision is a diagnostic; preserving useful
Jacobian transport and child information is the architectural objective.

Reference backend costs one factual forward, K full no-gradient replays and one
backward per update. All proposals/spatial sites are evaluated. Uniform site
sampling can have high N-weighted variance, and factual pathwise learning does
not explicitly train every losing value on every update. Event-sparse dispatch,
raw asynchronous timing, persistent cross-window memory and efficient suffix
caching are absent. Do not replace the integrated flagship with this diagnostic.

## Contracts and next admission decision

13 tests pass across original spatial and new downstream tests: deterministic
32-race replay; force-factual equivalence; override cleanup; early-key final-loss
credit without local heads; exact site-importance algebra; odd-shape relational
composition; nonzero mixed-child interaction; existing locality and local-credit
contracts. These are numerical development tests, not benchmark fits.

Before training v2, measure per-block final-loss alternative contrasts in both
float32 and float64, replay RSS/time and gate-gradient variance on real TRAIN
batches. Admit a fit only if useful early contrasts survive the selected
precision and resource limits. Compare the acquired v1 geometry/weights where
compatible, fixed observed window and identical data partition. The decision is
whether relational composition plus final-task credit preserves usable signal
through deep representations. Score unseen arrangements and child reuse when
appropriate data/labels exist; gesture classification alone does not prove an
object-part hierarchy. No new training job was launched by this implementation.
