# Shared forward teaching of counterfactual optimizer-state transport

R1/B1,11 October2026. Longer48-target calibration finds static contrast MSE
4.901e-8 and reference-only projection6.241e-8 on four fresh TRAIN prefixes.
Reference adaptation alone is27.3% worse here. First8 exact/zero-tail MSE is
1.974e-7; all three have useful predictive structure. This directs the next
engineering step toward branch learning-state influence, not more blind actor
training or a claim that reference projection always improves credit.

## A cheap exact observation and an approximate transported learner

For reference lexical state z_b, the forward pass supplies the normalized
probability p and supervised gradient

    g_b = mu_p-e_y.

A one-address native key perturbation changes p_r to q_r. At the same z_b,
its gradient is exactly

    g_a(z_b) = g_b+(q_r-p_r)(e_r-mu_except_r).

The second term is target-independent; already-revealed dense feedback supplies
g_b. One O(Vd) reference computation plus O(Wd) alternative feature operations
therefore teaches the gradient change at the shared state for W routes. This
is not yet g_a(z_a) after the alternative learns differently.

Maintain alternative numerical optimizer states (z_tilde^a,m_tilde^a,v_tilde^a),
starting at the common issued snapshot. A transport model estimates

    g_tilde^a = g_a(z_b) + H_tilde_t(a)(z_tilde^a-z_b).

Apply the actual optimizer update rule to each transported state. The optimizer
nonlinearity and moments remain explicit; they are not folded into an assumed
constant SGD step. An initial H_tilde can be zero or the shared forward-derived
feature-covariance diagonal. A connected learned low-rank/contextual operator
can replace this approximation after finite contracts and task-loss fitting.
It is shared across route/event history, not W isolated predictor networks.

Reference loss plus first-order learner influence predicts

    C_tilde_t(a) = C_t(a,z_b) + g_a(z_b)^T(z_tilde^a-z_b).

For fixed-feature softmax, the scalar Taylor remainder at the true branch state
is at most R^2 ||z_a-z_b||^2/2 when all feature norms<=R. Add gradient/operator
approximation and transported-state discrepancy bounds through the complete
optimizer-state recurrence in REFERENCE_LEARNER_RESIDUAL_BOUND. No global Adam
contraction is assumed. A missing Hessian-vector term can amplify transport
error; calibration determines whether the approximation is useful.

## How later forward computation trains the transport function

A candidate H_psi or recurrent state-transition function retains its issued
input activations, parameters/moment influence and version. Later supervised
forward losses teach predicted route consequences (or future-loss objectives),
rather than requiring imitation of backpropagated gradients. Native features
and covariance structure supply a data-dependent geometry; psi can adapt it
causally with history. Dense forward feedback teaches the shared component,
while actual interventions and positive-support full factual corrections teach
and audit the remaining branch discrepancy.

Subsuming optimizer hyperparameters requires differentiating their issued
update maps against later loss with recorded versions. The mathematical
one-step metric derivative is already contracted; no optimizer meta-learner
has yet been fit. A transport approximation is a base for that learner, not a
claim that small diagonal operators are the final representational capacity.

## Accounting and decisive gate

Independent exact worlds pay O(WVd) vocabulary/readout work each target. Shared
reference plus sparse transported optimizer states pays O(Vd+Wd) for the
restricted head, plus all active keys, projections, discovery, optimizer work,
state storage and operator evaluation. Full carrier learning has additional
cross-layer state and derivatives; this head calculation does not establish
whole-model/depth supremacy. It does identify a measurable restricted region.

Next: contract no-op/zero-learning/reference Adam parity, then compare zero-
curvature and diagonal transport predictions against the saved exact48-target
TRAIN world costs, recording complete wall and arithmetic. Proceed to a learned
connected operator only if measured residual structure and prospective data
support it. Exact factual correction remains required for actor correctness.
