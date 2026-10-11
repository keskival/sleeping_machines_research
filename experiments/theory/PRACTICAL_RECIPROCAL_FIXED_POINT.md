# Practical reciprocal fixed-point credit

B1/R1 AWS implementation, 11 October. Companion to CLOSED_LOOP_EXPLORATION_AND_CREDIT.md. Preserve active actor/critic sources; this is a reusable solver, not a replacement architecture.

Solve λ=b+Kλ with an issued, fixed operator K. A callback applies K to a vector, so no full Jacobian or infinite tape is required. For a coupled update map this callback may apply local VJPs; for learned transport it may apply the learned reverse map. The latter solves that learned operator, not automatically the true derivative. Forward probes and sparse exact residual audits still train/check the transport.

`credit/fixed_point_credit.py` performs bounded residual iteration. If a valid Euclidean contraction bound ||K||≤κ<1 is supplied, the current residual r=b+Kλ−λ gives

    ||λ−λ*|| ≤ ||r||/(1−κ).

The solver stops at the stated error bound or a maximum operator-call budget, returning explicit convergence status. The bound is conditional on the supplied κ; the implementation does not certify an arbitrary native map's norm. Scalar discount alone does not prove contraction. Hard routes require their own discrete counterfactual correction; this continuous solver does not differentiate an unrealized route for free.

Warm credit carry is an initial candidate, not an assertion of correctness under the next operator. Each solve rechecks it using current issued context/version and source b. This permits economical carry across events/updates without silently accepting stale derivatives. Freeze operator parameters/state during a solve. Complete changing-learning-map differentiation may involve mixed derivatives/optimizer state; a local state-only operator is a narrower target.

Storage inside the solver is a few credit-sized vectors. Callback activation storage, persistent model state and sparse auditing remain additional. Solve cost is number of calls times actual transport cost, plus vector operations. A map near unit contraction can require many calls; enforce a budget and label unresolved solves, rather than claiming constant total work. Learned transport may amortize that cost, but predictor fitting and correction costs must be charged.

Practical integration: obtain local error b from observed forward loss; predict continuation credit; use it as a warm candidate; run a bounded number of local transport/refinement operations; issue the forward update; teach transport from forward packets and occasional exact audits. No third learned trainer is required: observations/local derivatives anchor the teaching signals. Native task quality, drift and resource scaling remain the tests that decide whether this construction is economical.

## Completed numerical contract

`aws_fixed_point_credit_20261011_v1` completes via AWS slot3/run_safe. Frozen32-dimensional synthetic operator, known norm bound0.45: exact solution error4.574e−11 versus residual error certificate8.662e−11. Cold solve18 transport calls; warm carried candidate12 versus18 for a fresh solve on the same changed source. Cold measured0.168ms on this tiny case only. Stale version rejected; two-call budget reports unconverged. Source-bound results preserved. No native coupled-learning or end-to-end efficiency claim.

## Native changing-operator witness

`aws_native_reciprocal_fixed_point_20261011_v2` completes on a two-event native TemporalMemoryLayer over four ordinary Adam parameter updates. Sixteen forward directional probes teach a256-float dense transpose at each version. The declared local continuation equation uses gamma0.1846–0.1866, selected from exhaustive current operator norm to bound contraction0.4. Forward-taught versus exact VJP solutions differ by at most2.54e−16; direct-solution errors stay within the supplied residual certificates. This is a frozen local linear continuation target at each version, not the existing whole-model loss gradient.

Teaching costs51.5–86.5ms/version; learned-map solve0.143–0.189ms versus exact VJP2.42–3.58ms. Thus learned queries are cheaper but exhaustive teaching dominates a single query; no net speed win is claimed. Carried candidates take10/17/18/18 calls versus cold10/16/17/18: cross-version carry does not improve these cases. Prior synthetic warm-start savings cannot be generalized to changing nonlinear operators. Native v1 failed the projection-class import; its failed log is preserved and v2 fixes only that wiring.

Next investigate selective teaching and repeated-query amortization at issued versions, or train context-generalizing transport under the compact-credit owner. Do not scale dense exhaustive teaching or assume cross-version warm carry helps. B4 confirmation source bytes and queues remain unchanged.

## Shared-parameter flips

For u=F_w(x), y=C_w(u), and loss L(y), the shared parameter gradient is (partial_w C)^T grad_y L + (partial_w F)^T (partial_u C)^T grad_y L. A single connected reverse pass accumulates both contributions. Reusing parameters does not make these standalone gradients: the early use is weighted by downstream sensitivity. In repeated updates, detach and ordinary optimizer operations can truncate this chain. At a fixed point z=T_w(z), implicit differentiation gives grad_w L=partial_w L+(partial_w T)^T lambda, with lambda=grad_z L+(partial_z T)^T lambda. Sharing closes parameter identity but does not itself prove stable dynamics, useful credit or contraction.
