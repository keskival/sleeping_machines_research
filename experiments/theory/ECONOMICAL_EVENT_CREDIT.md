# Economical event-driven credit — R1/B1 implementation design

User-directed, 10 October 2026. Objective: retain useful deep credit while reducing complete learning work on clockless, asynchronous, sparse hardware. This is a derived design and implementation plan, not a measured hardware or scaling result.

## Causal graph and local state

An event e updates addressed state s_e^+ = F_theta(s_e^-, m_e, delta_t_e). Physical elapsed time drives decay and races; event dependencies establish causality without a globally synchronized layer tick. The exact reverse recurrence is lambda_e = local_loss_derivative + sum_{j:e->j} J_{j,e}^T lambda_j; parameter contribution is B_e^T lambda_e. Losing-route interventions are additional credit targets, not automatically present in the factual graph.

Replace routinely retained state-by-parameter sensitivities with addressed persistent credit state z_e^+ = U_psi(z_e^-, packet_e, delta_t_e). C_psi predicts cotangents or interventions from this state and later error packets. Packets can carry selected address, race margins, gates, age, activation projections and counterfactual summaries. State and parameter version identifiers accompany every packet. Connected critics can exchange sparse messages; they need not be disconnected linear estimators.

For squared-integrable target G, the population reduction from adding history Z to packet P is E||E[G|P,Z]-E[G|P]||^2. This follows from orthogonal conditional-expectation projections. It identifies when persistent information is useful; it does not guarantee finite-data learning or grokking. Preserve learned drift and causal history rather than assuming stationary inductive biases.

## Residual audits and budget allocation

Fix a causal filtration containing predictions and inclusion probabilities before audit outcomes are sampled. For g=sum_i g_i and predictions h_i, use g_tilde=sum_i h_i + sum_i I_i(g_i-h_i)/pi_i, with full support on every potentially nonzero residual. Conditional Bernoulli sampling makes its mean g; independent draws give variance sum_i (1/pi_i-1)||g_i-h_i||^2. Correlated race selection requires joint inclusion probabilities and covariance terms; do not reuse the independent formula for a fixed-size race.

Under sum_i c_i pi_i <= B, minimizing independent variance gives pi_i=min(1, ||g_i-h_i||/sqrt(mu*c_i)). This oracle allocation is an analytical benchmark: residual norms are unavailable until audited. The implementation uses past calibrated residual-risk estimates, a positive exploration floor, and charged discovery work. An imperfect allocation changes variance, not conditional unbiasedness, provided predictions/probabilities are fixed before the draw and the recorded probabilities are correct. Zero estimated risk never proves zero true residual. Predictor overhead and all-candidate discovery are charged. Local addressed candidates do not justify dropping nonlocal contributions without a bound or residual audit.

Unbiasedness applies to the gradient at its recorded version. It does not imply an unbiased nonlinear optimizer update or an unbiased gradient at a later parameter version. Optimizer state, preconditioning and update rules belong to the learned policy and resource account.

## Emission, accumulation and delay

For an L-smooth conditional future objective Q_t, update -eta*h and ||grad Q_t-h||<=epsilon imply Q_t(theta-eta*h)-Q_t(theta) <= -eta(1-L*eta/2)||h||^2 + eta*epsilon||h||. Emit only when the certified descent term exceeds the error term and lambda times charged event cost. Accumulated credit is evaluated again against current state; summing individually useful stale updates is not automatically safe. Bounded-age or bounded-drift flushes prevent permanent starvation; silence itself is supervised where the model requires it.

If the credit function is K-Lipschitz in parameters, stale-version error is at most K||theta_current-theta_version||. Add this to epsilon, together with drift of the future conditional distribution and state mismatch. This is a conditional bound requiring known or validated constants. Recompute, audit or discard when the bound fails. Logical version ordering is compatible with clockless execution; physical time still affects model dynamics.

Exact past-loss derivatives calibrate local transport. Future intervention outcomes train useful updates under Q_t(H,a)=E[future loss + resource penalty | H, do(a)]. Factual observations alone do not identify every action-conditioned outcome. Separate past cotangents, future-policy gradients and intervention values in interfaces and experiments.

## Hardware economics and next gates

Keep separate counters for arithmetic, messages/bytes, local and remote state accesses, retained storage, candidate discovery, audits, critic fitting and optimizer work. CPU wall time is not a clockless-hardware energy measurement. Complete break-even requires prediction + discovery + communication + audits + critic training + optimizer to cost less than the exact reference at matched quality/data.

1. Implement a reusable independent residual estimator and floor-constrained cost allocation. Exhaustively enumerate small audit masks to verify expectation, variance, cost and biased predictor cases; reject invalid probabilities and costs. R1/B1 contract only, no fit.
2. Use the admitted native conditional-noise measurements to decide whether histories carry predictive intervention information. Fix the measured transfer failure before deep critic fitting.
3. Integrate addressed credit state with native producer/key/optional-write contracts; compare reset, frozen and learned persistent state at depths 2/4/8. Charge dense key scoring and full derivative contractions until replaced.
4. Compare uniform and uncertainty-directed audits at matched budgets; vary event emission and delay independently. Preserve exact factual credit and audited alternatives. Record work-to-quality and memory, not only quality per pass.
5. Validate resource savings in integrated inference/learning before hardware claims. The current CPU scan is an execution enabler with full first-order credit; it is not the sparse event learner and does not support higher-order meta-gradients.

The completed two-layer Taxi exact-trace reference improves final development LL to 0.4770 versus local 0.4565 and BPTT 0.4418 (three seeds, 20 passes), but costs approximately 85 versus 1.3 minutes. The design targets this cost gap; those observations do not establish economical depth scaling.
