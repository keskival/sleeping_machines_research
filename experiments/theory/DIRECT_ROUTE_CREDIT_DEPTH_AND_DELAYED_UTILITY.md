# Direct route credit, depth, and delayed learning utility

Curie,12October2026. Analytical B5/R1 support during unchanged native world-credit confirmation. These are conditional derivations, not a neural-depth scaling result or a claim of field novelty. The current source-choice pipeline remains unchanged.

## A direct route-score error bound

Fix full issued history H, the current model/numerical state, discovered candidate pool, and all response/read noise before the source-choice draw. Candidate logits z_r are computed before Y; the route does not intervene on the external outcome law. Let mu=E[Y|H], pi=softmax(a), and A_r=softplus(z_r). Stop decoder responses when teaching route choice, as in the implemented surrogate. Conditional route risk is

    R=sum_r pi_r*(A_r-mu*z_r).

For one issued belief q independent of the winner/audited-alternative draw, epsilon=q-mu gives score-gradient bias

    b_r = -epsilon*pi_r*(z_r-z_bar),
    z_bar=sum_r pi_r*z_r.

Consequently

    ||b||_1 <= |epsilon|*sqrt(Var_pi(z)) <= |epsilon|*D/2,
    D=max_r z_r-min_r z_r.

The first inequality is Cauchy-Schwarz; the second is the bounded-range variance inequality. It has no candidate-count or neural-depth factor. Directly projecting one outcome law into evaluated deep-program risks therefore avoids multiplying errors from a chain of separately learned backward maps for this source-choice endpoint. The world must still learn an adequate belief as depth/data complexity grows, and the actual output range must stay controlled. Those premises are measured quantities, not guarantees.

For underlying producer parameters theta, set j_r=grad_theta a_r and j_bar=sum pi_r*j_r. The same bias is

    b_theta=-epsilon*sum_r pi_r*(z_r-z_bar)*(j_r-j_bar),
    ||b_theta||_2 <= |epsilon|*sqrt(Var_pi(z))
                            *sqrt(sum_r pi_r*||j_r-j_bar||_2^2).

This exposes the remaining depth/conditioning factor: the actual producer response geometry. Small score error is not automatically small parameter-update error. Charge those local/temporal response maps, candidate discovery, decoded losing values, optimizer and state transport. Factual value/weight learning is a separate path; these bounds do not replace BPTT or all deep credit.

Current WorldCreditSensor uses an independently queried pooled belief; its completed contract verifies winner-independent world forecasts. Its mean route-logit gap is already measured. The generating-depth4/8 comparison is not a neural-depth test. Next depth integration should measure belief error, response spread, producer-response norm and complete work together instead of treating increasing depth as a sufficient scaling demonstration.

## If forward-response conditioning depends on the sampled pair

The deferred ResponseWorldCreditSensor prototype conditions on the actual factual and sampled alternative responses. That gives useful information, but q_ij can now vary with winner i and alternative j. One belief shared across that pair is not one fixed risk potential across all possible pairs. The preceding exact bias formula must not be reused silently.

Let j be sampled with recorded positive proposal p(j|i), independent of Y conditional on issued H and i. For the implemented pair estimator, conditional mean bias is

    b=-sum_i pi_i*sum_{j!=i} grad_a pi_j*(q_ij-mu)*(z_j-z_i).

If |q_ij-mu|<=e uniformly and the evaluated logits have range D,

    ||b||_1 <= e*D*sum_j 2*pi_j*(1-pi_j)^2 <= 2*e*D.

Proposal probabilities cancel in the conditional mean, not in variance or work. This remains a depth/candidate-count-independent score bound under its uniform-error premise, but is weaker than the fixed-belief bound. Proper calibration under a restricted pair summary need not establish that premise: credit-weighted conditional error can persist, as proved in FORWARD_RESPONSE_SUFFICIENCY_FOR_CREDIT.md.

Two clean implementation choices are a common pre-draw conditioning set/reference with charged discovery/decoding, or explicit pair-conditioned predictions with this different estimator/consistency accounting. Computing all candidate outputs creates a common conditioning set but increases value work; it is not free sparse execution. The deferred prototype currently implements the latter interface and needs that scope in its contract before fitting. If actions alter external outcomes, use intervention-conditioned laws; neither common-mu derivation applies directly.

## What the next forward pass can teach about an optimizer intervention

Predicting outcomes and training credit for future learning utility are distinct. The current source-choice contrast stops its inferred belief; paired_route_credit also stops loss_difference. Therefore no derivative through the actor's credit-generated parameter update teaches the outcome readout. Actual outcomes teach its belief, and future factual tasks teach its shared memory. Do not describe that as a completed optimizer-utility meta loop.

For a finite issued actor update, let m_k denote an adjustable teaching mean and v_k its known source-choice gradient sensitivity. With B averaged targets, a stopped-response representation can write

    g(m)=g(m0)-(1/B)*sum_k (m_k-m0_k)*v_k.

Let U(theta,S_optimizer,g) be the actual declared update, including clipping/moments, and let g_future be a valid terminal-loss cotangent at the reached learner. Conditional on the issued state and held response maps,

    dL_future/dm_k = -(1/B)*v_k^T*(D_g U)^T*g_future.

Thus a whole parameter update can return one scalar teaching signal per issued credit decision. The optimizer response combines local operations with shared clipping/moment state; omitting those interactions changes the derivative. If future hard routes are stochastic, their score terms or supported counterfactual estimates also belong in g_future. Holding persistent state fixed declares a truncated intervention, not a full-history derivative.

An issued native control activation can retain m_k's response to its own parameters, so the next forward outcome supplies this scalar supervision. Responses v_k can be stored only on touched producer blocks, or replaced by a validated directional response with explicitly paid error/audits. Rank is bounded by the number of retained credit controls, not necessarily by total parameter count. Obtaining g_future and v_k, retaining optimizer/state versions, and computing the actual update response remain charged. A dense vector implementation would not establish sparse-hardware economy.

Keep the proper outcome forecast separate from a prospective teaching preference. A native controller could emit calibrated q plus a bounded mean-changing offset or blend/step control taught by future utility. A probability output optimized only for downstream learning utility need not remain a calibrated outcome law. Both outputs can share native temporal state/reads/writes; dense recurrent controllers are unnecessary. No such offset/control fit is admitted by this note.

## Current engineering consequence

The unchanged fresh-system actor194 stage passes all three controls overall and at both generating depths. Fresh source195 then196 remains the priority, with the registered stop gates and complete publication. If confirmation fails, use completed traces to distinguish insufficient conditional response information, shared-gradient covariance and optimizer-utility mismatch before selecting one native intervention. If it completes, carry the confirmed rule into the owned integrated depth/token path and measure cost to matched quality. Do not replace this work with a standalone critic or hyperparameter grid.
