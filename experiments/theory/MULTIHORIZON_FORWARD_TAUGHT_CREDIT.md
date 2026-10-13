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

## 10. Preserve the objective while changing the audit resolution

`credit/nested_horizon_audit.py` records caller-produced paired prequential
losses in one contiguous stream. It freezes the terminal weights at issue,
checks common observation and issued-world/version identity, captures the lower
boundary and accumulates to the selected upper boundary with compensated sums.
Incomplete streams cannot return completed corrections. Rejected observations
leave the ledger unchanged. The ledger retains constant scalar accumulation
state plus O(H_K) explicit objective weights, not a trajectory of activations.
Actor, optimizer and native-memory snapshots are separate paid storage. This
component records outcomes; it does not recompute native gradients or prove
that the supplied worlds follow their declared learning policy.

Audit resolution and the learning objective are different choices. For any
partition 0=H_-1<H_0<...<H_K with the same terminal outcome and weights,
telescoping still gives V_HK. Merging adjacent residual blocks replaces R_j and
R_(j+1) by their sum; the terminal mean is unchanged while variance and audit
cost change. For random worlds,

    E[(R_j+R_(j+1))^2] = v_j + v_(j+1) + 2 E[R_j R_(j+1)].

Thus positively correlated errors can make a merged block expensive; opposite
errors can cancel. Earlier TRAIN receipts can inform a future partition, but
the current unseen residual must not choose its own partition without a new
selection correction. Coarsening the ladder is not the same as shortening the
terminal horizon, discounting late outcomes more heavily, or replacing weighted
sums with per-horizon means. Those changes alter what the learner values.

For implementation error analysis, suppose supplied paired differences differ
from the declared learner-world differences by epsilon_k. Then

    |V_tilde_H - V_H| <= sum_(k<=H) w_k |epsilon_k|.

The sampled-block correction amplifies its local label error by1/(q_j s_j).
Perfect receipt algebra cannot repair a frozen-gradient alternative world,
mispaired corpus slice or changed objective. The next native integration gate
must compare each world's own gradients/moments/state evolution to an explicit
uncached replay, then test nested prefix equality under shared observations
and paired RNG. It must charge both worlds' complete learning work. This is
the concrete boundary between a correct recording interface and actual
forward-taught long-horizon learning.

The source-bound R1 nested-stream contract completed through guarded AWS slot3
(`results/credit/aws_nested_horizon_audit_contract_20261013_v1.json`): terminal
mean error4.44e-16, six invalid-stream checksPASS, peakRSS20,476KB. Its fixed
paired losses give nested advantages-2.5,-0.5,+3.5, demonstrating that the ledger
retains a sign-reversing consequence rather than selecting on the early prefix.
This is a recording/algebra contract, not a native learner rollout or quality win.

## 11. Native replay must name the future learning policy

`credit/native_token_future_world.py` implements a cloned native token learner
with its own factual-gradient, norm1-clipped AdamW updates. Cloning model and
optimizer jointly preserves parameter bindings. Numerical persistent state and
race RNG are copied; the source learner is not advanced by either replay.
Both worlds score the same legal token observations before their own update.
An initial addressed route intervention is cleared before subsequent steps.

This is a declared diagnostic future policy, not the production probability0.25
importance-corrected alternative-teacher policy. Hard races, addressed state,
key/value separation and actual optimizer moments are retained; future
alternative-write teacher work is omitted in this diagnostic. A gate for this
policy cannot establish the production learner's delayed utility. The next
integration must replay the actual future teaching policy with its own private
teacher/admission RNG and recomputed alternative-world teaching labels.

Formally let z contain the complete learner state, including optimizer and
private memories. For shared exogenous inputs/random tape let production and
diagnostic updates be T_k and S_k. On a region where T_k is L_k-Lipschitz and
delta_k bounds ||T_k(z_tilde_k)-S_k(z_tilde_k)||, their trajectory error satisfies

    e_(k+1) <= L_k e_k + delta_k,
    e_k <= sum_(i<k) delta_i product_(i<j<k) L_j,  when e_0=0.

If the queried prequential loss is C_k-Lipschitz there, objective discrepancy
is bounded by sum_k w_k C_k e_k. An intervention contrast has one such bound
for each world. This follows by adding/subtracting T_k(z_tilde_k) at every
step; it is conditional finite-trajectory perturbation algebra, not global
Adam stability. Hard route changes can invalidate the local Lipschitz region.
Those route boundaries require explicit replay observations or separately
justified probabilistic bounds rather than inventing small L_k.

The practical conclusion is precise: a clean cloning/replay gate removes
implementation ambiguity, while matching the declared learning policy removes
target ambiguity. Shared random tapes pair uncertainty; they do not force
the worlds to choose the same future routes or acquire the same moments.

The native D8 contract completed in guarded AWS slot3:
`results/credit/aws_native_token_future_world_contract_20261013_v2.json`. Three
prefix loss errors are0 and the final actor/optimizer/state/RNG fingerprint
agrees exactly after resuming from a cloned prefix. Source and global RNG are
unchanged; paired race RNG agrees. Own moment L2distance is0.004764623. Across
three four-target blocks the intervention has summed advantages+0.003373146,
-0.003073692,-0.003390312: the initial benefit reverses to cumulative-0.003090858.
This is a synthetic legal-token/D8payload8 numerical witness under the declared
factual-gradient policy, not population utility or corpus quality. All nine
executed updates and36 scored/own-gradient targets are charged;1.338s and
423268KBpeakRSS. v1 failed in moment inspection because inactive parameters
lack Adam state; corrected v2 preserves the actual learner execution. The
production alternative-teacher policy remains the next replay gate.
