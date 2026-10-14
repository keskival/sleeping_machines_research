# Learn the consequences; retain the exact conditional audit mean

14 October 2026 · R1 prepared learning primitive. No native baseline fit or measured learning-cost gain.

The concrete learning problem is that longer route consequences can add useful delayed signal and large importance-weighted noise. Global clipping/Adam can turn that noise into biased actual updates. Replacing exact consequences outright with a learned critic changes the raw learning target when predictions are wrong. A control variate offers a different integration: predict most of the consequence and audit the residual while retaining the existing conditional raw mean.

## Conditional identity

Fix an issued actor/path/query and factual winner w. Let pi_j be route probabilities, g_j=grad pi_j under the declared factual differentiation, and Delta_j the first-time-held alternative-minus-factual continuation loss. Delta_w=0. Sample J among nonwinners with full-support q_j, independently of unseen alternative outcomes. The existing raw estimator is

    G = g_J Delta_J/q_J.

Issue a baseline vector b before sampling/observing the audited alternative. Stop its gradient for this actor update. Define

    G_control = g_J(Delta_J-b_J)/q_J + sum_(j!=w) g_j b_j.

Then E_J G_control=sum_(j!=w) g_j Delta_j=E_J G. This holds for any issued baseline accuracy, because the finite residual sum and analytic correction cancel exactly. The omitted winner matters: subtracting a constant from nonwinner labels without adding the correction generally changes the mean. Site/admission/suffix importance weights multiply the whole controlled term, not just its sampled residual.

`controlled_route_credit.py` implements this as a zero-forward surrogate. The baseline and labels are detached, only route probabilities carry actor credit. It leaves factual prediction, temporal races, addresses, private state, keys/values and the current teacher's first-time convention intact. It is a standard control-variate construction, not a new theorem or a replacement of the temporal actor by a dense model.

## Variance and the proper learning target

Conditional on fixed coupled-world contrasts, put e_j=Delta_j-b_j. The candidate-sampling covariance trace is

    sum_(j!=w) ||g_j||^2 e_j^2/q_j
      - ||sum_(j!=w) g_j e_j||^2.

The analytic correction is deterministic under this conditioning. A perfect full contrast vector makes candidate-sampling variance zero; that oracle vector is not free to acquire. Poor baselines can increase variance. Weighted residual regression targets the first term, but its cross-route correction means it is not the exact variance objective. Route-logit norms are a cheap declared proxy; actual parameter or optimizer-response geometry requires its own paid measurement.

Random future position/site sampling and teacher admission have additional variance. A learned conditional mean does not remove unpredictable future noise, stale-operator mismatch or clipping bias automatically. Verify actual prediction and complete work, rather than promoting a critic on MSE alone.

## Versioned native integration

Baseline inputs should be typed native receipts: causal cut/route identity, actor/state/version, declared consequence horizon and factual forecast/context legitimately available before the alternative audit. Shared native temporal functions and addressed memory predict route consequences. Updating the critic using the current selected alternative and then using its new predictions in that same correction breaks the issued-baseline condition. Freeze or retain the issued prediction until its receipt matures.

The analytic sum evaluates baseline values for all scored candidates. Their discovery, critic inference, storage, audit labels, fitting and optimizer work remain charged. At current pool4 this is bounded; larger dormant capacity requires economical candidate/baseline discovery, not a claim of zero work. An inference actor still delivers winner-only values. The audit is a learning operation.

The current D16/8M and D8/32M ladder, longer-horizon comparison remain unchanged; automatic64M continuation is withdrawn. Focused unit tests enumerate this gradient identity. It is not added as another actor-pipeline gate and no native baseline fitting or variance result exists. Actual baseline fitting belongs after the completed horizon's clipping/quality diagnosis, with causal/version contracts and the complete acquisition ledger.

## Populated-state path diagnosis

The completed D8 full-stream projection cuts all produce the same7.933941NLL. A bias-free cut from an empty downstream state supplies zero content at every subsequent token. The existing units cannot create nonzero content from those empty values, so the readout falls to its intercept distribution regardless of which stage is cut. These results measure whole-chain input dependence, not distinct learned stage contributions. Their raw scores are preserved.

The prepared, unqueued warm-path diagnostic retains256factual inputs in each1024history before cutting one stage's projection output. Its downstream banks are populated, and its first256targets must replay exactly. The remaining768target loss tests persistence and dependence after the intervention. It still does not establish added-depth information or achievable shallow loss; a trained model comparison is needed for those claims.
