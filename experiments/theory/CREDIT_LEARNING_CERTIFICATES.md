# Conditions for useful learned credit

10 October 2026. Founder-directed B1/R1 development. These are sufficient mathematical conditions and proposed engineering tests, not evidence that the current learner satisfies them. Standard conditional regression, control variates and smooth-descent identities support the derivations; their integration with versioned native event credit is the construction to test. No priority claim is made.

## 1. Predict the unresolved part, preserve the known part

At a fixed actor version and declared replay horizon write the exact raw gradient as

    g = g_known + sum_b A_b a_b.

A_b is the local producer VJP map from a cut-state cotangent to parameter credit. The cuts must give a nonoverlapping decomposition; arbitrary overlapping layer cuts cannot simply be summed. g_known includes exact receiver and retained factual-path terms. For a prediction a_hat_b made before its audit draw, define

    r_b = A_b (a_b - a_hat_b).
    g_hat = g_known + sum_b [A_b a_hat_b + I_b r_b / pi_b].

Condition on the trajectory, frozen predictor, parameter versions and probabilities. Independent Bernoulli I_b with 0 < pi_b <= 1 give

    E[g_hat] = g,
    E||g_hat-g||^2 = sum_b (1/pi_b - 1) ||r_b||^2.

Proof: each centered audit contribution is (I_b/pi_b-1)r_b, its mean is zero, its variance coefficient is 1/pi_b-1, and independent cross terms vanish. Shared/correlated audit schedules require their joint inclusion probabilities and covariance terms. An exact decomposition is a prerequisite, not a consequence of this estimator.

This identifies the useful training loss: producer-weighted cotangent error ||A_b(a_b-a_hat_b)||^2, rather than an arbitrary unweighted coordinate MSE. A sampled parameter-space projection can estimate this norm, but its own VJP/JVP, sampling variance and discovery are charged. Learning unused cotangent directions can otherwise look successful without helping the actual update.

For event models, compute observed local credit and known elapsed-time transport exactly. Learn the unresolved continuation conditional on currently available evidence. After a later outcome arrives, its versioned packet supplies a training target. Previously unseen future outcomes remain stochastic; a conditional tail predictor is not an exact future oracle. Positive-support audits correct its error for the declared completed/replayed objective.

## 2. Informative features imply a strictly better population predictor

Let y be a square-integrable missing parameter-credit contribution and phi a retained feature vector with positive-definite S = E[phi phi^T]. This is a representational witness; implementation predicts state cotangents and respects its producer map. Put C = E[y phi^T]. For linear prediction W phi, expansion of the squared error gives

    W_star = C S^{-1},
    R_star = E||y||^2 - tr(C S^{-1} C^T).

Therefore C != 0 implies strictly smaller population error than zero prediction. Include a constant feature if an intercept is desired. A singular S uses the corresponding observable subspace/pseudoinverse. The proof completes the square in W. Finite-data estimation, regularization and changes in actor/distribution still have to be measured; a nonzero training cross-moment alone does not establish held-out transfer.

This is a testable reason to expect learning when forward evidence contains information about credit. Estimate cross-moments and held-out prediction on a small fixed teacher before scaling. If features discard the relevant direction or time relationship, C can vanish even though the full forward trajectory is informative. More compute will not repair that interface automatically.

For changing regimes use the time-indexed moments S_t and C_t, conditioned on the legitimately observed context. Persistent credit state and causal parameter updates approximate W_star,t. There is no requirement that useful sharing, locality or timescales remain stationary. Abrupt drift may invalidate a learned approximation; audit support and explicit state/version handling are retained.

## 3. A measurable overhead threshold

For a scalar aggregate residual-energy R and common audit probability pi, audit variance is R(1/pi-1). To meet a positive target variance V, the smallest common probability is

    pi = R / (R + V).

If R = 0 exactly, audits are unnecessary for that fixed target; declaring residual zero from unverified predictions is not admissible. For R > 0 preserve positive support. Let c_a be the measured cost of auditing all blocks under a linear common-probability cost approximation, c_p the predictor/transport/fitting overhead per update, and R_0 the residual energy of zero prediction. Predicted credit beats the zero-prediction audited learner at equal conditional variance when

    c_p + C_pretrain/T < c_a V (R_0-R) / [(R_0+V)(R+V)].

This follows by subtracting their expected audit costs c_a R/(R+V). It explains how a real prediction improvement can fail to pay at small scale and pay after amortization or when audits become costly. It does not say audit cost necessarily grows faster than predictor cost with depth: both are measured. Shared graph costs, discovery, metadata, dense local updates and service overhead can invalidate the linear cost approximation and must replace it with the actual cost curve. The V=0 exact-gradient boundary supplies no saving from an imperfect predictor in this model.

For differing block costs c_b, minimizing sum_b pi_b c_b subject to a variance budget gives interior probabilities proportional to ||r_b||/sqrt(c_b), clipped at one. True residuals are not known without audits: use previous/calibrated predictable estimates with a positive floor, record them before sampling, and measure realized variance. Selecting only apparently easy blocks without support changes the estimator.

## 4. Lower variance supports forward learning under explicit conditions

For a fixed L-smooth objective F and an unbiased raw-gradient estimate with conditional variance V, plain SGD satisfies

    E[F(theta-eta g_hat)] <= F(theta)
      - eta(1-L eta/2)||grad F(theta)||^2 + L eta^2 V/2.

Proof: apply smoothness to the update and use E||g_hat||^2 = ||grad F||^2+V. The gradient must target this objective; horizon truncation and population/minibatch noise require their own terms. At equal eta and mean gradient, reducing V improves this guaranteed upper bound by L eta^2 DeltaV/2. This is not a guarantee that every realized step improves.

If additionally ||grad F||^2 >= 2 mu(F-F_star) and eta <= 1/L, the expected gap contracts with factor at most 1-mu eta plus L eta^2 V/2; the resulting bound on the constant-step noise floor is L eta V/(2 mu). These assumptions provide a local/conditional progress certificate, not global nonconvex convergence or grokking. The current nonlinear learned optimizer is not covered: passing an unbiased noisy gradient through a nonlinear update need not preserve unbiased updates. Evaluate its future-loss utility separately and use the fixed-SGD arm to isolate this certificate.

## 5. Concrete native design and small-scale gates

Prioritized design: addressed cut-state cotangents, exact physical-time transport, and a shared low-rank conditional continuation predictor. Preserve queries, posterior-weighted mark errors, producer messages, elapsed time and parameter versions instead of reducing them to coarse parameter-coordinate averages. Shared functions use consistent address permutations; useful temporal regimes enter through evolving state. Credit clocks may prioritize predicted variance reduction per service cost, with counterfactual scheduling credit and supported audits. This scheduling remains an integration proposal, not measured async execution.

Before scaling:

1. Fixed-packet fit: one tiny teacher and retained TRAIN packet set. Show acquired credit structure through lower producer-weighted error than frozen/zero predictors. Failure here separates optimization/representation trouble from unseen-data generalization. This diagnostic is labelled memorization capacity, not transfer.
2. Held-out learning: fixed final checkpoints on disjoint packets under matched state availability. Test reset-to-reset and, separately, causal prefix-assisted persistent-to-persistent assessment; teacher labels enter only after each assessed prediction. Report at least three seeds and paired uncertainty for the selected variant. Separate fixed-context learning from cross-context/depth transfer.
3. Learning utility: compare later forward loss under fixed updating, same audit budget, data and update count. Report residual variance alongside calibration so credit fit cannot conceal producer-irrelevant predictions. Learned updating is its own controlled arm.
4. Scale only after a useful small signal: measure total-work-to-quality and the overhead threshold across depth/data. Scaling should amortize a demonstrated learner, not substitute for evidence that it learns.

Current coarse-coordinate pilot: mean TRAIN relative error 1.03184 over steps 1–16 and 1.06318 over steps 49–64; held-out error mostly about or above the zero-prediction baseline. TRAIN uses persistent state while assessment resets it, so that run does not isolate persistence utility. It supplies no clear positive learning signal. The state interface, optimization, and context mismatch are hypotheses to diagnose separately; the exact state-key contract is the next already-admitted gate. No new fitting job is admitted by this note.
