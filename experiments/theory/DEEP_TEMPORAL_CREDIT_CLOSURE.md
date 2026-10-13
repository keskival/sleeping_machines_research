# Useful temporal depth and a closed credit-learning system

13 October 2026. R1/B1 derivation supporting actual native D8/D16 acquisition, not a new training admission. Existing races, addressed persistent state, separate keys/values and counterfactual write teaching remain the substrate. Curie owns compact-credit integration; AWS owns the deeper language acquisition. The propositions below describe explicit mathematical constructions, not measured properties of the current driver.

## 1. Depth is a computational composition, not a readout collection

On a finite causal event horizon, let x_v be an event's incoming packet and s_v its addressed memory. A selected native event applies

    (y_v, s_v^+, tau_v^+) = F_v(x_v, s_v, tau_v; theta_v, a_v).

The route a_v is chosen by the temporal race. An edge delivers a value through its time transport T_e before a later event uses it. Persistent state is another causal edge. The deployed predictor reads the terminal computation. More layers increase the number of compositions available to transform and retain evidence; merely concatenating earlier outputs increases an interface and does not establish useful depth.

For a fixed factual route realization, parameter version and event horizon, collect continuous state/output derivatives in the event graph. The chain case has

    lambda_d = r_d + J_d^T lambda_(d+1),
    J_d = partial x_(d+1) / partial x_d.

Here J includes native transport and state-mediated paths inside the declared cut; r is direct loss demand. A hard-route selection has an additional alternative/route-credit contribution. Differentiating the realized continuous path alone does not supply that contribution. All total parameter-credit targets below include the admitted route term as well as factual paths.

Useful depth requires both informative forward transformations and a learning signal that reaches their producers. A nonzero derivative is necessary for a differentiable path to learn through that path, but does not certify a well-conditioned or useful gradient.

## 2. Finite causal closure needs no hierarchy of trainers

Stack the fixed-realization continuous demands as lambda. Write

    lambda = r + B lambda.

B transfers demands from later events to earlier events. On a topologically ordered finite causal graph with at most K events along a dependency path, B is strictly triangular and B^K = 0. Therefore

    lambda = (I-B)^(-1) r = sum_(k=0)^(K-1) B^k r.

This is a finite algebraic fixed point, unique without requiring ||B|| < 1. K measures the unrolled causal graph, including persistent-memory edges, rather than just the layer count. Large ||B|| can still make this solution poorly conditioned; existence does not establish stable learning.

A trainable predictor C_phi(q_v, revealed_feedback) approximates a cut demand or the producer's complete future utility. Its training targets come from realized forward consequences and selected audits of that finite graph. Its ordinary supervised derivative is computed locally or in the existing backward call. That derivative is an analytic differentiation rule, not a third learned trainer. Freeze/stop-gradient the issued target and audit policy at this update. This closes the operational loop: forward consequences teach credit; credit updates forward parameters; later consequences supply the next labels.

A target involving later weight adaptation must condition on the learner cut: parameters, optimizer moments, refreshed memory and versions. Its forward consequence includes the actual later update policy. The AWS clip/Adam cotangent transports a future learner demand through a known local update, but does not replace learning that future demand. Sharing actor and critic parameters does not eliminate these causal boundaries. At a frozen cut, gradients of a specified combined objective add; sequential optimizer steps generally produce a different trajectory.

## 3. Direct cut prediction avoids multiplying random corrections

Unbiased local corrections are not enough to make deep composition economical. Consider a scalar chain with true derivative a_d and a locally corrected random estimate A_d, independent across layers, with mean a_d and variance v_d. Then

    E product_d A_d = product_d a_d,
    Var(product_d A_d) = product_d (a_d^2 + v_d) - product_d a_d^2.

Proof: independence factors the first and second moments. If a_d = 1 and v_d = 1, the variance is 2^D - 1. Thus a stack of individually unbiased backward estimators can have exponentially growing composition variance.

The alternative is to predict each producer's complete cut-to-future credit G_d directly, rather than multiply random local backward maps. Let b_d be its issued prediction and draw an audit coin A_d ~ Bernoulli(p_d), after fixing the prediction, target availability and propensity. The residual correction is

    Ghat_d = b_d + A_d/p_d (G_d - b_d).

Conditional on the full fixed target and issued prediction,

    E_A Ghat_d = G_d,
    E_A ||Ghat_d-G_d||^2 = (1/p_d - 1) ||G_d-b_d||^2.

For disjoint parameter blocks under a Euclidean norm, total squared error sums over blocks exactly. No cross-block independence assumption is needed for that norm. Under a general positive semidefinite update metric M, independent audit coins conditional on all targets make cross-block error terms average to zero; correlated coins do not generally have this property. Shared parameters must first aggregate their issued contributions or retain the cross terms. A target-dependent audit decision is invalid unless its correct propensity and resulting conditioning are used.

Consequently the audit component of error can grow as a sum of residual risks rather than a product of local second moments. This is an estimator result, not a guarantee that the residual risks remain bounded with depth or that complete teacher work is linear. A cut-to-future teacher can itself require a long suffix, route alternatives, refresh and optimizer replay. These costs stay explicit.

If the teacher is a noisy unbiased observation of a conditional expected target, the formula applies to the observed target; its squared residual includes irreducible teacher noise. Conditional unbiasedness must be proved at the declared information boundary. Native hard-route sampling, producer credit and future adaptation cannot be silently omitted.

## 4. Train in the producer's actual update geometry

Let U map an issued credit vector g into the next learner cut u^+, including the actual optimizer and any state refresh. With a chosen positive semidefinite future-demand metric H, define

    M = J_U^T H J_U,    J_U = partial u^+ / partial g.

To first order, credit error e incurs quadratic learner displacement e^T M e. Raw state-cotangent MSE is not this quantity. The actual clip/Adam transport includes moments and a charged global clipping contraction. H must be specified or estimated from consequences; it is not automatically available from the optimizer.

Condition a critic on the causal information Q_d it actually receives. For fixed conditional M_d and a square-integrable target, the optimal weighted predictor minimizes

    E[(G_d-b_d)^T M_d (G_d-b_d) | Q_d].

If M_d is a deterministic function of Q_d, any conditional mean E[G_d|Q_d] is a minimizer (unique on the positive-definite subspace). If M varies with unavailable future outcomes, the optimum solves

    E[M_d | Q_d] b_d = E[M_d G_d | Q_d],

on the identifiable subspace. Replacing this with unweighted regression can select the wrong predictor. This distinction connects directly to Curie's optimizer-context and actuation diagnostics.

A low-dimensional packet or action chart can identify only its retained directions. Additional depth or more data cannot resolve exact input aliases. Store associations between route/mode, activation statistics, issued errors, signed update response and version; pooled norms alone can discard the signs or alignments required for future utility.

## 5. Optimal audits and the economic test for deep credit

Suppose a producer's predicted metric residual risk is R_d^2 and a complete audit costs c_d > 0. With independent coins and the block conditions above, minimize expected audit work subject to an audit-variance budget V:

    minimize sum_d p_d c_d,
    subject to sum_d (1/p_d - 1) R_d^2 <= V,  0 < p_d <= 1.

The interior Lagrange equation gives

    p_d = min(1, sqrt(nu) R_d / sqrt(c_d)).

Choose nu to satisfy the budget when feasible; zero-risk blocks need no audit for that frozen target. During learning, retain positive coverage unless a zero-risk claim is exact. R_d and c_d can be trained/estimated from paid receipts; the allocation remains predictable before its audit draw. Estimator error in those quantities changes efficiency, while known full-support propensities retain the residual estimator's unbiasedness.

With N updates, critic cost kappa per update and acquisition cost C_acq, a necessary economic comparison to auditing every block is

    N sum_d (1-p_d)c_d > N kappa + C_acq.

The residual-risk budget must also be small enough for useful actual updates. More depth helps this economic case when expensive suffix audits become predictable and shared critic work amortizes; it hurts when unpredictable residual risk or teacher cost grows faster. This is the precise quantity to measure rather than assuming sparse inference implies inexpensive deep training.

## 6. Recurrent bootstrap closure: stability is an engineering objective

A recurrent/infinite-horizon critic bootstrap is a different construction from the finite graph. If its target map T is a contraction with constant rho < 1 in the declared norm, it has a unique fixed point. If a learned solution has residual ||Chat-T(Chat)|| <= epsilon, then

    ||Chat-Cstar|| <= epsilon/(1-rho).

Proof: triangle inequality and contraction give error <= epsilon + rho error. A discount gamma alone is insufficient: if continuation transport has norm L, a linear bound needs gamma L < 1. Physical-time decay and rotation can bound the transport leg, but gates, state updates, route responses and the learner update must also be included.

Stable forward information and stable learned credit are not the same as forcing every useful long-range signal to vanish. Preserve identifiable memory modes with controlled propagation; audit and correct uncertainty in high-impact directions. A blanket contraction across all memory would also erase evidence needed for long-range language. Finite consequence targets offer closure without requiring that blanket constraint.

## 7. Predictions and measurements for the admitted D8/D16 program

The current AWS fits use full factual window backpropagation, dense Adam and sampled alternative-write teaching. They test actual forward depth and acquisition; they do not yet implement the direct-cut critic or prove the audit savings above.

Record depth, capacity, selected writes, scored keys, acquisition and complete fit resources beside every quality row. The admitted smokes verify finite fitting and nonzero input-bank gradients at each depth. Inspect gradient distributions across layers, saturation, memory rank and route occupancy from the saved curves; these diagnose learning access, not usefulness by themselves. Frozen long-history DEV scoring tests whether improved short-window learning survives the declared context protocol.

At fixed 8M acquisition, D8 versus D16 tests the additional native stages. D8 at8M versus32M tests whether their benefit develops with acquisition. Same recipe and final-stage decoder, no direct early-layer readout bypass. A larger model's early deficit does not cancel reserved acquisition. If gradients collapse, repair depth conditioning while retaining temporal computation; if gradients arrive but quality stalls, investigate representation and credit horizon. Compact-credit integration then measures producer/update-weighted residual risk, full teacher cost and work to matched quality, with original saved references retained.

No Transformer superiority or asymptotic scaling exponent follows from this derivation. The next concrete competitive result is the selected deeper checkpoint under the pinned public scoring protocol. These propositions identify how to build and evaluate that checkpoint rather than replacing its evidence.
