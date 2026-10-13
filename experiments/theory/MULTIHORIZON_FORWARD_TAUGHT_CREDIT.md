# Multi-horizon forward-taught credit for delayed learning

13 October2026, R1 shared credit direction. Standard telescoping control variates and randomized truncation are the mathematical primitives. This construction applies them to actual native learner consequences and one horizon-conditioned credit function. It is a proposed implementation, not the teacher used by the current deep language fits.

## 1. Fix the learner and objective before selecting a horizon

At a causal cut X retain parameters, optimizer moments, native memories, clock/RNG state, policy versions and the proposed intervention a. Reference and intervention worlds follow the same declared future learning policy and paired exogenous observations. Recompute each world's own later gradients and updates; do not substitute frozen reference gradients if the target is future learning advantage.

Let delta_l(k)=l_reference(k)-l_intervention(k) be the prequential loss difference after k future outcomes. For fixed nonnegative weights w_k, define

    V_H = sum_(k=1)^H w_k delta_l(k).

This is a cumulative weighted advantage, not an unweighted horizon-dependent mean; changing its denominator changes the target. Finite H is sufficient for a practical objective. An infinite target V_infinity requires convergence and integrability. Recorded language observations may be paired because the predictor does not change the corpus; embodied actions that change observations require the actual action-conditioned world model or environment.

A positive short-horizon loss does not rule out a favorable later advantage. Conversely, a favorable prefix can reverse. Increasing depth without extending consequence teaching can leave useful slow learning effects weakly observed. The current AWS language driver retains factual creditwindow16 and chunk-local alternative teaching; it does not already implement this learner-consequence target.

## 2. One shared predictor, many horizon queries

Choose increasing horizons H_0,H_1,..., with H_-1=0. At issue, one native conditional function supplies b_j=C_phi(X,a,H_j) and an aggregate endpoint prediction b_infinity. Require b_j to converge to b_infinity for an infinite construction, or use a finite terminal b_K for K levels. Set b_-1=V_(H_-1)=0.

Define residual increments

    R_j = (V_(H_j)-V_(H_(j-1))) - (b_j-b_(j-1)).

Telescoping gives

    sum_j R_j = V_infinity-b_infinity,

or V_(H_K)-b_K for finite K. This identity requires the same issued learner/world/weights across horizons. Separately reset learners or different weighting protocols do not supply nested increments.

Sample one level J from a known positive distribution q_j, fixed before its audit. Return

    Vhat = b_infinity + R_J/q_J,

or b_K+R_J/q_J in the finite case. Conditional on a complete paired world and issued predictions, E_J Vhat=V. A query needs only b_J,b_(J-1) and the endpoint prediction, not a sum of all horizon predictions. The two nested labels can reuse a rollout to H_J, but its full rollout, memory and subsequent learning are paid work. The endpoint prediction is not a substitute for the required telescoping boundary condition.

For finite levels the proof is a finite sum. Infinite exchange requires absolute integrability and the stated convergence. An unconstrained horizon head with an unrelated endpoint cannot be called an unbiased construction. A practical first implementation should use a finite registered H_K and leave all levels supported.

## 3. Exact variance and the long-horizon cost condition

For fixed residuals and independent categorical level selection,

    Var_J(Vhat) = sum_j R_j^2/q_j - (sum_j R_j)^2.

For random future worlds define v_j=E[R_j^2|issued context]. Their noise is included in v_j. Reducing predictor error does not erase irreducible uncertainty in the future world. The unconditional variance also contains variation of the true target across worlds.

Let c_j be expected complete cost of auditing level j. The second-moment/cost product satisfies

    [sum_j v_j/q_j] [sum_j q_j c_j] >= [sum_j sqrt(v_j c_j)]^2.

This is Cauchy-Schwarz. Equality is attained by q_j proportional to sqrt(v_j/c_j), when normalization and moments are finite. Zero-risk entries need a coverage policy while risk is learned. A cost-constrained variance optimum can instead have q_j proportional to sqrt(v_j/(lambda+mu c_j)); these are different optimization constraints.

For dyadic horizons, suppose c_j=O(2^(kappa j)) and v_j=O(2^(-alpha j)). A geometric proposal q_j proportional to2^(-beta j) has finite mean cost if beta>kappa and finite second moment if beta<alpha. Therefore alpha>kappa supplies a sufficient interval of economical proposal exponents. This is the engineering objective: predictable residual increments must decay faster than rollout cost grows. It is not a property guaranteed by shared weights or temporal sparsity.

The sufficient condition can hold through both objective weighting and acquired prediction. If w_k=gamma^k with0<gamma<1 and per-outcome losses are bounded, raw dyadic block magnitudes eventually decay faster than any geometric rate. The constants can still be expensive when gamma is close to1. Choosing a short discount timescale merely to make this condition easy can erase the delayed learning benefit we seek. Unbounded NLL needs explicit moment conditions rather than a bounded-loss claim. Nonstationary undiscounted infinite sums may not converge; finite registered horizons avoid that issue.

## 4. Train credit without introducing a higher trainer

Issue and retain the baseline predictions before selecting J. After its forward consequence matures, the corrected Vhat becomes a stopped target for the current endpoint output B. Holding the actor/world target fixed, the raw endpoint squared-loss gradient is

    g_phi = 2(B-stop(Vhat)) grad_phi B.

Because E_J Vhat=V, its conditional expectation is2(B-V)grad_phi B: the exact gradient of the endpoint squared error at that cut. Dependence of the issued control variate on phi does not require differentiating through its target; the conditional mean cancels it. This is a supervised semigradient identity, not the derivative of an arbitrary meta-objective or a guarantee for clipped/Adam training.

If actor and critic share parameters, the identity applies to the declared frozen actor-label cut. Actor utility and label-path derivatives are separate terms; shared parameters do not make them disappear. Re-evaluating one part of an issued prediction with new weights before its correction invalidates the issued baseline calculation unless a new consistent baseline is formed.

The same admitted rollout supplies short-horizon outputs, longer increments, prediction residuals, cost and proposal-calibration labels. A shared temporal credit model receives horizon/time/version queries. There is no separate neural trainer for each horizon, and no trainer above the actual forward-outcome teacher. Ordinary local differentiation and known importance weights close this loop.

## 5. Reuse factual futures and retain their causal types

The factual future losses often already occur during ordinary learning. They can be cached for admitted receipts; this reuses their evaluation, not the alternative world's learning. An intervention world needs its own state/moments, later gradients and policy decisions until the selected horizon. Coinciding route draws and shared exogenous futures reduce some contrast noise but are not independent repeated labels.

Keep reference/intervention identities; actor/critic/optimizer versions; terminal objective weights; paired future/RNG identity; issue and maturity times; selected horizon and exact probability; learned baseline outputs; cutoff policy; full audit and storage costs. A scalar immediate loss receipt cannot be silently relabelled as future learning utility.

Current native teacher work and forward packets provide potential inputs, but not already-computed long learner-world targets. Bounded sparse state and compact snapshots can reduce the storage boundary; actual implementation and contracts must demonstrate that economy.

## 6. Integration after the admitted deep acquisition

Keep D8/D16 and32M running. Inspect whether deeper-stage information and longer acquired histories improve language quality. The next forward-taught compact-credit integration should register a finite horizon ladder, retain a terminal anchor and supported draws, then audit the telescoping mean/variance and endpoint gradient inside that battle's pipeline before fitting.

Select by actual later quality and complete learning work. A better local MSE, a smaller teacher count or a short-prefix gain is insufficient when later utility reverses. Conversely, an early deficit is not an automatic rejection when the registered longer-horizon target and acquisition stage remain valid. This construction makes the delayed benefit measurable and trainable rather than asserting it will appear.

## 7. A resource cutoff is part of the estimator

The intuition is to predict the complete delayed consequence and occasionally pay
to repair one missing time block. The importance correction pays for rare blocks
in their statistical weight. It does not make an abandoned block disappear.

Suppose level j is selected with q_j, but its rollout completes only when A_j=1.
Returning the terminal prediction on aborted audits yields

    E[Vhat_abort | issued cut] = b_K + sum_j E[A_j R_j | issued cut].

Its bias relative to E[V_HK] is -sum_j E[(1-A_j) R_j]. Thus timeout, RSS failure
or stopping an unfavorable prefix can select precisely the missing delayed
effects. A recorded completed-audit fraction alone cannot repair this bias.

If completion is a genuinely randomized, predictable admission with known
positive probability s_j, independent of the unseen residual conditional on
the admission information, A_j R_j/(q_j s_j) restores the mean. A deterministic
resource kill with zero completion probability in some worlds does not meet
that condition. Preserve failed receipts and repair execution or register a
different finite objective prospectively; do not quietly drop them.

For a deterministic registered truncation H_m < H_K, omitting all later
corrections while keeping b_K gives expected target

    E[V_Hm] + b_K - b_m,

with bias -sum_(j>m) E[R_j]. If TRAIN-only evidence supplies valid bounds
E[R_j^2] <= u_j, Cauchy-Schwarz gives absolute bias <= sum_(j>m) sqrt(u_j).
These are population bounds only when u_j are justified population bounds;
empirical residual means are estimates. Replacing b_K by b_m produces the
exact shorter objective instead, which must be reported as a changed horizon.

This gives two distinct engineering options: preserve the long target with
supported paid audits, or explicitly shorten the target with a quantified
missing-tail bound. A shorter wall time by itself does not identify better
learning efficiency.

## 8. Executable receipt boundary and next integration gate

`experiments/credit/finite_horizon_receipt.py` implements only the finite issued
receipt and scalar correction. Immutable tuple snapshots retain predictions
and full-support probabilities; objective/world/learner identifiers reject
mixed receipts; level zero enforces the zero-horizon anchor. It returns a
stopped scalar target and deliberately has no actor-gradient or replay engine.
Identifiers do not prove that replay is faithful. The caller must retain paired
RNG/state, optimizer versions, cumulative weights and complete audit costs.

The next R1 pipeline contract must enumerate a finite ladder with nonuniform
probabilities and signed increments, checking the exact terminal mean, the
variance formula and the endpoint raw-gradient mean. Include an early negative
advantage with a positive terminal advantage, container mutation, mismatched
world/version rejection and the aborted-audit bias above. The source-bound R1 scalar contract completed through guarded AWS slot3:
`results/credit/aws_finite_horizon_receipt_contract_20261013_v1.json`. Terminal
mean, raw-gradient mean, supported completion mean and abort-bias errors are
zero; variance error is1.42e-14. Eight invalid-receipt checks pass; peakRSS20,988KB.
This is finite scalar enumeration, not native learner coupling or a fitted
credit model. Native learner rollout integration remains pending. No new fit is
admitted; current D8/D16/32M source-frozen drivers remain the acquisition test.

## 9. Horizon reach and audit frequency share one work budget

Predictable optional admission can spend no rollout on some issued cuts while
preserving the finite target. Let pi_j=q_j s_j be the probability of receiving
one completed residual audit at level j; the remaining probability returns b_K.
For a fixed paired world, the corrected target has

    Var(Vhat) = sum_j R_j^2/pi_j - (sum_j R_j)^2.

For random paired worlds replace R_j^2 by conditional second moments v_j for
the first term and include the appropriate target-world variation. The relevant
work is complete marginal audit cost c_j, including the alternative world's
own future learning and snapshot storage. Under a predictable randomized
admission made before starting that work, expected audit cost is sum pi_j c_j.
Failed attempts that already consume rollout work require an additional cost
term; treating those costs as zero would invalidate the economics.

For a fixed expected audit budget B, minimizing sum v_j/pi_j with
sum pi_j c_j=B gives, when all v_j,c_j>0 and no support/categorical constraint
is active,

    S = sum_j sqrt(v_j c_j),
    pi_j = B sqrt(v_j/c_j) / S,
    minimum second moment = S^2/B.

This follows from Cauchy-Schwarz, or from -v_j/pi_j^2+lambda c_j=0.
It is feasible for the one-audit-or-none construction only when sum pi_j<=1.
At larger budgets that constraint is active and the solution becomes
pi_j=sqrt(v_j/(lambda c_j+mu)), with nonnegative mu and constraints solved
together. Enforced coverage floors introduce their own active constraints.
Conditional risk estimates and costs must be learned from earlier TRAIN
receipts, not the unseen current residual. Zero estimated risk does not justify
zero support while the long target remains registered.

This connects useful depth to a concrete teaching objective: acquire a predictor
that reduces long-block residual risk enough to afford long consequences. A
small completion probability restores the mean through a large importance
weight, but raises variance; unbiasedness alone is not efficient learning.
Global clipping and Adam may then change actual update utility, so select the
eventual integration by later language quality and full work, as the current
depth/acquisition program does. No separate trainer for each horizon is needed.

The updated receipt helper accepts a known supported completion probability.
The finite R1 contract enumerates completed and uncompleted outcomes rather
than claiming unbiasedness from a sample average. This algebra gate cannot
certify real resource-kill propensities or native rollout coupling.
