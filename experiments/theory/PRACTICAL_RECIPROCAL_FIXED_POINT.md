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
