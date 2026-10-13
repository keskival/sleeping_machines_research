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
