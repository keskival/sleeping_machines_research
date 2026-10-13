# Causal action-space credit and learner-state transport

13 October 2026 — B10/R1 implementation direction.

The completed B10 actual-utility experiment shows a distinction: a numerically correct current-action derivative can still give a weak or unhelpful adaptation rule. Online1024decision loss5.756685231 versus frozen5.756599087; retain the frozen recipe. This is evidence about this bounded recipe, not a ceiling on learned credit.

Let x_t contain actor parameters, optimizer moments, causal event memory and pending receipts. Let c_t=C_phi(h_t) be a native temporal critic action issued from observed history. A learner transition is

    x_{t+1}=F_t(x_t,c_t,y_t),    ell_{t+1}=L(x_{t+1},y_{t+1}).

The current implementation differentiates the parameter update and refreshed memory with respect to c_t, holding issued gradients, incoming state and old moments fixed. Its one-step target is J_t^T grad_x L, where J_t=dF_t/dc_t. It does not differentiate the complete historical learner trajectory or exact discrete hard-route selection.

## Credit geometry is the geometry of useful actions

For a forecast vector f over causally revealed future events, define A_t=(df/dx)J_t. With positive semidefinite forecast curvature H, the local action metric is

    G_t=A_t^T H A_t.

Credit Euclidean MSE weights every coordinate equally; this metric weights effects on forecasts after optimizer and memory transitions. Null directions may consume credit-prediction capacity without improving forecasts. A local quadratic update solves

    (G_t + lambda I) delta c = - grad_c L.

A low-dimensional clock action permits explicit small matrices; larger sparse actions require products, local blocks or sketches with a separately charged budget. Damping bounds sensitivity. Estimating curvature or rescaling actions does not guarantee lower future loss: use causal held-out forecast comparisons and retain the strongest completed model.

The existing parameter-step trust penalty instead uses J_theta^T J_theta. It limits parameter displacement but does not measure whether that displacement matters to predictions. Thus the experiment's small mean step deviation2.44e-5 and gradient norm7.64e-5 motivate checking action sensitivity before increasing critic size or training length. These statistics alone do not prove a vanishing-gradient mechanism.

## Future decisions require learner-state credit

For a horizon K, learner-state cotangents obey

    lambda_t = grad_x ell_t + (dF_t/dx_t)^T lambda_{t+1},
    q_t = (dF_t/dc_t)^T lambda_{t+1}.

This includes future demands on Adam moments and persistent state, not just immediate parameter demand. Full unrolling is one teacher, not an economical deployment requirement. A native asynchronous critic can learn delayed q receipts, with explicit truncation and sparse admission; conditional importance corrections require nonzero predictable admission probabilities. Delayed labels must identify the issued action and actor/optimizer version. Do not silently apply an old derivative to a changed learner state.

AWS's ACTUAL_OPTIMIZER_CREDIT_GEOMETRY.md and learning/adam_credit_transport.py supply matrix-free clip/Adam moment transport. Integrate it into our existing temporal critic, retaining private causal memory, hard races and sparse selected writes. Charge the global clipping contraction, teacher transport, pending activations, candidate work and optimizer work. Matrix-free does not imply clockless or zero traffic.

Next implementation gate: compare analytic action sensitivity against functional actual-update derivatives on the same recorded learner states, then test a bounded optimizer-aligned proposal against the retained frozen critic on the same8+8forecast schedule. No new external architecture, sealed test access or blind larger fit is needed for that gate.

## Integrated optimizer witness and teaching rule

The B10 native-state integration now matches AWS transported action cotangents to the complete refreshed-future functional derivative at three actual actor/Adam versions: maximum errors2.22e-18,2.17e-19,8.14e-20. Four parameter-proposal JVPs give the exact local parameter-step pullback. At the first version its eigenvalues span2.26e-11to3.26e-5 (ratio1.44million); subsequent versions have substantially less unequal sensitivity. This is a measured local conditioning effect, not a general depth/scaling claim.

The first-step Adam formula explains a possible source. With zero moments, no clipping and nonzero scalar g, its update is −eta g/(|g|+epsilon). The derivative is −eta epsilon/(|g|+epsilon)^2, so actions that preserve gradient sign can have extremely weak local influence. Global clipping and later moment history modify the exact geometry; large sign-changing counterfactuals are not represented by an infinitesimal derivative. Thus a small local utility derivative is not evidence that all finite credit alternatives are unimportant.

For normalized critic actions u and c=S u, use G_u=S G S and q_u=S q. The damped solve d=−(G_u+lambda I)^−1q_u has q_u^T d≤0 for lambda>0; a positive scalar norm cap retains that sign. Supervise the native critic with the issued stopgradient target u+d. This separates action conditioning from learning the future demand, while retaining causal outcome labels. It is an optimizer-conditioned teaching proposal, not a theorem of forecast improvement or a novel Adam primitive. Four exact diagnostic JVPs are paid; a cheaper analytic tangent implementation needs an independent numerical comparison before substitution.

## Sparse optimizer metric without a dense action tangent

Let B contain disjoint clock-action supports, D the diagonal Adam derivative, g the complete issued gradient and k its clipping scale. In the clipped region, a=B^T g/[r(r+epsilon_c)], U=k D B and w=k D g. The action tangent is J=U−w a^T. Hence

    G=diag(U^T U) − (U^T w)a^T − a(U^T w)^T + ||w||² a a^T.

In the unclipped region a=0. The disjoint-support assumption makes U^T U diagonal; overlapping supports require their cross terms. Compute full-vector D and ||w||² once, then contractions on the64clock-support entries and a4×4matrix. This avoids a dense18,164×4tangent allocation and four dual proposal traversals. Globalnorm and scalar reductions remain, along with ordinary actor/optimizer work; this local algebra alone is not a whole-fit efficiency result. `market_sparse_action_metric.py` implements this decomposition and its native-state contract compares against exact functional proposal JVPs.

## A conditional descent test, without assuming stationary biases

Classical Taylor descent supplies a useful acceptance condition for the native construction. Let R=G+lambda I and let q_hat estimate the conditional future action demand at the issued learner state. Put A=||q_hat||_(R^-1), epsilon=||q−q_hat||_(R^-1) and d=−alpha R^-1 q_hat. If the future action-loss Hessian is bounded above by kappa R along this proposal, then

    L(c+d)−L(c) ≤ −alpha A(A−epsilon) + (kappa alpha²/2) A².

Thus epsilon<A and 0<alpha<2(1−epsilon/A)/kappa imply local improvement. A positive norm cap only reduces alpha. This is conditional on the true demand/curvature bound and issued learner version; the parameter-step Gram alone supplies neither assumption. Finite hard-route changes need explicit consequence comparisons instead of this smooth-region bound.

No stationarity premise enters the inequality. Changed strategy pools or inductive biases change q, R and the conditioning history. Causal critic adaptation must track that changing demand quickly enough that epsilon remains below A; delayed receipt staleness contributes to epsilon. The economical goal is therefore accurate future demands in action-relevant directions, not uniformly accurate retrospective gradients. Conditional validation can measure demand error and proposal consequences before making a scaling claim. The inequality is standard preconditioned descent reasoning; the experiment-specific contribution is the native event learner cut and sparse actual-optimizer geometry that makes the relevant quantities implementable.
