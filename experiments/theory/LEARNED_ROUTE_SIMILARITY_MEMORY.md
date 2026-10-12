# Learned route similarity and activation-distribution memory

12 October 2026, B1/R1 construction. User-directed formalization; not an implemented architecture or measured learning advantage. Keep Curie's current native-credit pipeline and AWS completed language evidence intact.

## Define similarity by transferable consequences

A route r is a typed intervention, specifying producer/receiver, layer/head, time and proposed message/write. Let X be the complete issued learner state (parameters, optimizer moments, numerical memories and causal history). Let omega specify a future demand and exogenous continuation. Define its effect relative to a common reference route 0:

    u_X(r,omega) = L_future(U_0(X,omega)) - L_future(U_r(X,omega)).

Positive u means improved future task loss. Include intervention and continuation costs in L if the target is quality/work. U contains the actual declared learning policy and optimizer, not just an immediate decode. The distribution nu_X of future demands is part of the definition, estimated from causal past observations or specified by the evaluation protocol.

Define squared behavioral distance

    d_X(r,s)^2 = E_(omega~nu_X) [(u_X(r,omega)-u_X(s,omega))^2].

Its square root is an L2 pseudometric: nonnegative, symmetric and satisfies the triangle inequality. Distinct routes can have zero distance when their effects coincide almost surely under nu_X. The geometry changes when future demand, current operators or the learner change. This is not an assumed fixed Euclidean activation manifold. If only mean effects matter for the decision, a coarser distance can compare mean utilities; matching full effect functions is the stronger transfer target.

Similarity can be K_X(r,s)=exp(-d_X(r,s)^2/(2 tau^2)). Under square-integrable effect functions and fixed nu_X this is a positive-definite Gaussian kernel on their Hilbert-space embeddings. It is expensive to know exactly; it defines the estimator target. The construction does not assert that the entire route space is a smooth manifold.

## Activation distributions supply causal conditioning

Let h_t(r) be the activation/message seen at the route, recorded before the outcome. A compact history summary includes exponentially weighted mean and diagonal/low-rank covariance, effective observation count, elapsed-time coverage and regime/freshness metadata. These statistics describe where the route has been used; they do not alone identify its future effect.

For a time-dependent forgetting factor a_t in [0,1), update

    mu_plus = a_t mu + (1-a_t) h,
    C_plus = a_t C + a_t(1-a_t)(h-mu)(h-mu)^T.

This is the exact covariance update of the mixture a_t P_old + (1-a_t) delta_h when mu,C describe P_old. With a fixed initial prior it includes that prior's mass; do not call early summaries unbiased sample covariance. A diagonal or fixed projection sketch costs O(d) or O(kd), not O(d^2). Store the projection identity/version when using a sketch. Long-lived and recent summaries can use separate timescales; forgetting is learned or task-selected, not synonymous with erasing all persistent memory.

Activation similarity is only a proposal prior. The same h can have different effects under different outgoing weights, optimizer moments or future demands. Include a current-operator anchor/sketch and optimizer/update summary as defined in CREDIT_STRUCTURE_SELECTION.md. Match units and transformation laws; obsolete operator information must be corrected or marked stale.

## Learn reusable coordinates from forward-observed effects

Issue a causal route descriptor z_X(r) from operation type, current activation, historical summaries, current operator/learner information and temporal state. A native temporal encoder produces e_psi(z). A shared demand-conditioned head predicts utility

    u_hat(r,omega) = k_X(r,omega) + f_psi(e_psi(z_X(r)), q(omega)),

where k is a declared known local-effect anchor and q encodes demand. At decision time use only causally available demand forecasts. Arrived future demand can supply teaching labels later; it cannot become a pre-decision oracle feature.

A paired-effect training loss is

    L_pair = w [ (u_hat(r,omega)-u_hat(s,omega))
                 - (u_observed(r,omega)-u_observed(s,omega)) ]^2.

Independent copies of the same issued learner and coupled exogenous futures provide the label. Full paired learning-world labels are paid; an immediate forced-route loss is a different, shorter-horizon label and must be tagged accordingly. If sampling probability is p(r,s,omega)>0 and the desired distribution has mass density rho, use w=rho/p with the appropriate density ratio. Clipping that weight changes the objective and must be declared.

With unbiased label noise independent of the issued prediction conditional on the packet, expected squared regression equals squared mean-effect error plus label variance; the extra term is constant in psi. Thus the encoder can learn effect coordinates without learning the noise. Correlated labels reused to fit the same issued predictions invalidate this simple argument. Directly regressing squared noisy route differences to a distance also includes a noise floor; use independent repeat products or model/subtract that floor rather than treating uncertainty as route dissimilarity.

An estimated distance can use the learned effect head over a bounded set of retained demand sketches:

    d_hat(r,s)^2 = sum_j v_j [u_hat(r,omega_j)-u_hat(s,omega_j)]^2,
    v_j >= 0, sum_j v_j = 1.

This is a pseudometric after square root for the fixed evaluated head. Forecast error still determines whether it is useful. The shared encoder transfers across routes; individual addresses need not each relearn an unrelated predictor. Past activations plus paired effect labels are the information that teaches this geometry.

## What belongs in a memory record

Keep the following typed fields rather than one unlabelled vector:

* Operation/address and retained causal issue descriptor; layer/head/time and proposed write magnitude.
* Activation mean, variance/sketch, effective count and coverage, with recent and persistent timescales where useful.
* Known local-effect anchor or operator sketch, actor/optimizer version and freshness/drift summary.
* Learned reusable key and effect/value representation; encoder version is needed when keys change.
* Observed paired effect receipts, their reference intervention, horizon, continuation policy, arrival time and sampling propensity; bounded sufficient summaries can replace raw receipts only for the declared objective.
* Prediction residual/uncertainty and actual acquisition, query, repair and storage costs.

The numerical representation can be a compact native message; the logical types prevent immediate loss, future learning effect and uncertainty from being silently mixed. Memory budget and expiry remain paid policy decisions. A count is exposure, not proof that the current relation is calibrated after drift.

## Use the geometry for both retrieval and allocation

Retrieve candidate memories by learned effect keys, then score current usefulness and freshness. Discovery and key scoring are charged; evaluating every retained demand head against every record is not sparse execution. Similarity supports transferring a residual or estimating uncertainty, not assigning identical credit by fiat. A shared predictor can combine current anchors with retrieved residuals.

Choose audits/discovery by predicted net future decision improvement, as in DECISION_VALUE_AND_DISCOVERED_OPTIONS.md, while retaining positive support. High activation novelty can flag a gap in coverage, but does not itself establish high learning value. Record propensities so sampled receipts can teach future estimators without survivor-only selection bias.

Actual later task loss is the ultimate adoption criterion. In a differentiable finite retained learning graph,

    grad_psi L_future = (D_psi U_r)^T grad_X_future L_future,

plus any explicit psi-dependence of the terminal loss and the required score terms for stochastic choices. Auxiliary L_pair trains reusable effect prediction; it is not a replacement for this task-valued teaching. Known local operators and paid observed outcomes close the loop without a learned trainer above the estimator.

## One integrated next comparison

After the current owner fit, compare native credit with and without causal activation-distribution summaries under the same actor, data, acquisition/audit budget and future-effect objective. Keep current-activation inputs in both arms, so the contrast measures history rather than the existence of an input. A matched reset/shuffled-history diagnostic, constructed without future leakage, distinguishes useful distribution memory from extra parameters. Then test frozen acquired geometry on changed route addresses, current operators and dependencies, allowing declared numerical memory updates.

Adopt on prospective task quality per complete acquisition/reuse work; report effect calibration and retrieval recall as explanations. This note admits no standalone geometry grid, no new dense critic and no hidden generator-identity features. It specifies a concrete future extension of native relational memory, not a claim that the extension has already been fitted.

## Implemented observation-memory component

`sleeping_machines/route_activation_memory.py` now implements bounded numerical activation means, diagonal variances and caller-supplied projection covariance, effective weighted counts and version-change metadata. Independent issued snapshots cannot be rewritten by future observations; out-of-order observations and retrospective queries fail. Same-time events remain distinct observations. Memory capacity exhaustion requires an explicit retention decision. Numeric storage and operation counters are exposed, with Python overhead and issued-copy costs excluded explicitly. Three guarded R1 checks pass in `results/diagnostics/aws_route_activation_memory_20261012_v1.json`. This implements conditioning statistics only; typed future-effect receipts, a trained similarity encoder and an integrated quality/work comparison are the next construction steps.

## Native outcome-model integration (12 October)

`causal_induction/history_world_credit_sensor.py` is a separate child of Curie's WorldCreditSensor. It conditions native outcome-memory content on read-before-write target/context activation summaries, with a zero-initialized bounded linear adapter (1248 added parameters at the current payload). Actor features retain their original stop-gradient boundary. The first placement before that boundary failed its gradient check; v1 logs are preserved. The repaired placement injects the learned history input at the critic's native content projection, after detachment. Native temporal races, addressed records, independent noise, outcome teaching and known BCE contrast algebra remain unchanged.

Each outstanding query stores its issued numerical history token. Feedback reuses that token rather than recomputing statistics after intervening observations. Summaries are per public target address over the current target/context activation pair; this does not yet implement learned retrieval between route addresses. Parent numerical forecasts and RNG initially match exactly; history-disabled comparison has the same added parameters and numerical memory work. In-process parameter edit counters mark freshness; they are not portable operator fingerprints.

Guarded R1 `aws_history_world_credit_20261012_v2` passes the two-query native witness: initial forecast error 0, outcome-loss gradient into the history adapter 0.0118768, and zero adapter gradient in the disabled-history arm. The witnessed numerical memory uses 1664 bytes, excluding Python objects, issued copies and all parent state. These are executable-path checks, not task-quality results. No parent driver/source/checkpoint is modified.

Before acquisition/reuse fitting, register a matched history-enabled/disabled protocol and validate multiple outstanding-query receipts. A source checkpoint with the parent optimizer requires explicit name-based moment transport into the expanded parameter set; do not silently reset Adam or load an incompatible state. Charge projection, statistics, snapshots, edit-counter reads, adapter backward/optimizer and source acquisition. The current Curie outcome-law cohort completes first; the child does not displace it.

## Current deployment diagnosis and fitting decision

The parent world-credit cohort completed its eight cases. Learned actor gain is +0.000538 NLL over exact and +0.000034 over fixed-prior halfblend, but generating-depth-8 gain over prior is -0.000018; the declared primary gate fails. No automatic history fit follows. Curie's persistence-posterior control is running and remains the owner's next comparison.

AWS's read-only, source-bound `aws_world_credit_sensitivity_20261012_v1` reconstructs the immediate route-sensitive error from completed receipts. For factual/alternative logits z_f,z_a, binary BCE contrast is softplus(z_a)-softplus(z_f)-q(z_a-z_f). The halfblend q_tilde=(y+q)/2 gives squared residual (q-y)^2(z_a-z_f)^2/4. Thus four times the reported residual sum divided by the logit-gap-square sum measures route-sensitive Brier. Learned value 0.159867 versus fixed prior 0.25 is a 36.1% error reduction; ordinary learned outcome Brier is 0.170538. RMS route logit gap is 0.102374. Exact teaching has zero residual because it uses the revealed label, not because its issued forecast is perfect. Each adapting arm visits its own route distribution, so these are per-arm calibration diagnostics, not a fixed-route causal comparison.

Better immediate route calibration already exists without a material actor gain here. Historical summaries may improve conditional forecasts, but adopting them requires future task utility and total-cost evidence; we cannot justify the extra module by repeating a calibration-only endpoint. The next diagnosis is route-credit influence on actual actor updates and future learning under the stronger conditional control. Saved aggregate receipts cannot identify that whole derivative. The source-matched public systems are now restored on AWS; the acquired model checkpoint still needs source-matched transport before fitting. No substitute stream is admitted.

## Byte-matched causal replay and interpretation

`aws_world_public_restore_20261012_v1` restores the 16 original acquisition systems and eight deployment systems from the exact recorded generator/seeds/options. Every public NPZ matches its original SHA-256; no oracle file is written or read by the analysis. Existing files are verified, never overwritten.

`aws_world_answer_changes_20261012_v1` replays the anonymous public events and captures the last observed target bit at each query, then matches saved issued probabilities and delayed feedback by ID/time. Learned, exact and prior arms each make zero binary-answer changes relative to persistence on all 1024 targets. Their measured NLL differences are confidence/calibration differences under this factual deployment. This does not imply all unexecuted routes would give the same answer, that future memory is unchanged, or that calibrated confidence has no task value. It also does not establish discovery of a different answer or a routing-dependent binding skill.

The actor decoder admits a concrete development hypothesis: ResidualHead uses z_r=(2*b_target-1)*(base+residual_r). Whenever base+residual_r remains positive, changing r changes confidence while preserving the target-bit answer. The function can reverse that answer if its residual crosses the threshold; this is an early-learning/observed-regime explanation, not a representational impossibility or a measured condition for every alternative. Saved factual traces establish the answer-preservation result directly; they do not reveal every alternative residual.

Curie's stronger conditional-persistence control completed: learned NLL0.418781 versus0.419187, gains+0.000212 at generating depth4 and+0.000600 at8. The original neutral-prior depth gate remains failed. The owner is confirming the unchanged member on fresh systems; no variant fit is admitted during confirmation. The next route-geometry evaluation should explicitly report confidence quality and routing-dependent answer recovery as separate endpoints. A task requiring transferred information to change the prediction is the decisive binding complement to this forecast-calibration cohort.
