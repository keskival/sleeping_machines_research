# Coherent world beliefs must produce coherent updates

Curie,12October2026. B1/R1, analytical support for native outcome-law credit and learned route-response geometry. Derived while the unchanged registered acquisition/reuse comparison runs. This note changes the diagnosis and next design criteria; it does not change that experiment or assert a measured advantage.

## Scalar credit accuracy is not enough

For binary forecasting, one causal outcome belief q=P(Y=1|H) gives coherent risks R(a)=softplus(z_a)-q*z_a. A predicted q_hat supplies every pairwise contrast without independent loss heads. The earlier note proves contrast bias epsilon*Delta_z and conditional MSE of the fixed half blend. That result concerns the contrast scalar. A shared producer can receive both factual loss gradients and route-choice gradients from the same outcome; their noise is correlated.

Fix the issued history, parameters, memories and sampled route pair. Let a denote outcome-independent gradient terms in a declared parameter block, u the factual prediction's outcome-sensitivity direction, and v the route-teaching direction. With both routes evaluated on the same target, the observed-label gradient is

    g_Y = a - Y*(u+v).

For the actual source-choice surrogate, u=grad z_w and v=(z_j-z_w)*grad pi_j/proposal_probability, with issued alternative logits detached as implemented. More general shared producer blocks include every relevant factual/route contribution in u,v; unrelated auxiliary objectives are outside this definition. The identity conditions on the pair and covers label noise, not route-sampling noise or later optimizer drift.

V1 smooths only the route teaching term:

    g_alpha = a - Y*u - [(1-alpha)*Y+alpha*q_hat]*v.

Writing epsilon=q_hat-q, its conditional mean changes by b=-alpha*epsilon*v. Its conditional covariance is

    Cov(g_alpha|H) = q*(1-q)*(u+(1-alpha)*v)*(u+(1-alpha)*v)^T.

For a fixed positive-semidefinite update-effect metric G, the change in noise work is

    tr[G*(Cov(g_alpha)-Cov(g_Y))]
      = q*(1-q)*[-2*alpha*u^T G v + (alpha^2-2*alpha)*v^T G v].

Thus perfect outcome calibration can still increase total producer noise. If u=-v, observed-label gradient noise cancels exactly; smoothing only v breaks that cancellation and adds alpha^2*q*(1-q)*v^T G v. This is a constructive counterexample to “better route-credit estimation automatically improves learning.” Separate scalar-estimator scores omit precisely the interaction that a connected critic should model.

## The common world mean gives a coherent alternative

For the same declared actor-gradient block, apply one issued target mixture to both factual and route terms:

    g_joint = a - [(1-alpha)*Y+alpha*q_hat]*(u+v).

Then bias is -alpha*epsilon*(u+v), and covariance is exactly (1-alpha)^2 Cov(g_Y) under the fixed-history premises. A single inferred outcome projection preserves the cancellation structure. q_hat is detached for this teaching operation: otherwise differentiating through a target-generating predictor changes the objective. Native world outcome supervision continues to use actual delivered labels; it must not silently become its own gold target.

For the exponential-family statistic T(Y), replace scalar Y with that statistic and combine all shared producer maps before measuring covariance. This is the mathematical reason to learn connected credit structure rather than a bag of disconnected estimates. Known local maps can project the common mean; a learned causal response model supplies the maps where exact transport is too expensive, with supported forward teaching and paid validation.

The joint construction trades calibration bias for label-noise reduction. It does not prove that a predicted mean is correct or that today's conditional risk is the ideal future objective. It is a proposed comparison if diagnostics identify cross-channel inconsistency, not another fit admitted by this note.

## Future utility exposes the missing premises

Let a declared fixed linear SGD-like update be theta'=theta-eta*g. Suppose the relevant next-window objective is exactly quadratic over these reached updates, with gradient g_F and Hessian H_F at the issued theta. Let mu=E[g_Y], b=E[g_new]-mu and Sigma_new,Sigma_Y be their covariances. The expected next-window loss difference is exactly

    -eta*g_F^T b + eta^2*mu^T H_F b
    + (eta^2/2)*b^T H_F b
    + (eta^2/2)*tr[H_F*(Sigma_new-Sigma_Y)].

With b=0 and H_F positive semidefinite, reducing covariance in that metric improves this quadratic objective. With drift, g_F can differ from the gradient of current data. With negative curvature, less noise can hurt. With imperfect calibration, the bias terms may dominate. With clipping/Adam, use the actual nonlinear state transition or its explicitly justified local expansion; this identity cannot certify those optimizers by renaming their gradients.

Actual future-loss contrasts and native forward response observations supply evidence for these quantities. The architecture should retain enough joint response information to learn which teaching modifications help, not optimize a local estimator MSE as though it were future utility. Worlds predicting a learner must include the learning transition and optimizer state when claiming that effect.

## Consequences for route similarity and depth

Two route programs are credit-similar when their effects align in shared producer/future-output space, not merely when their scalar losses agree. The same error-sensitive geometry must combine the factual and counterfactual routes that share a producer. Independent metric-weighted norms can miss cancellation or amplification, just as separate content/time risks omit cross terms in FORWARD_OBSERVED_CREDIT_RISK.md.

At depth, shared world statistics can compress correlated outcome uncertainty, but accurate local transport and its cross-route products remain necessary. A learned response block can retain combined sensitivities and correct compositional residuals from future observations. A tiny local head cannot be assumed sufficient; neither can a larger head be assumed useful without the joint future-task endpoint.

## Registered diagnostic priority

Finish the current exact/learned/prior native comparison unchanged. Its actual risk/blend contract remains correct. If it fails the future-actor gate, distinguish (1) insufficient conditional outcome information, (2) sensitivity-weighted miscalibration, (3) cross-channel noise cancellation, and (4) mismatch to future learning utility. Measure a shared-state factual/route response rather than inferring any cause from global NLL alone. A joint actor-target projection is justified only by that diagnosis and must retain actual outcome training, native memory/race mechanisms, and complete cost accounting in its comparison.
