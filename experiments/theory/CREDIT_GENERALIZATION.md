# Learned inductive structure for reciprocal credit

10 October 2026; founder-directed B1/R1 shared enabler. Extends the native integration gates and notes 155, 160–166. This is a formal target and development protocol, not a novelty, grokking or scaling-win claim.

## The object whose structure is learned

For a fixed causal trajectory and parameter version, output cotangents e_t produce an earlier activation cotangent a_j through the exact adjoint operator. A learned approximation is

    a_hat_j = sum_(t>j) K_psi(j,t; H) e_t.

H is the retained forward evidence and specified learning-time observations. If K is to be a linear adjoint approximation, its coefficients may depend on the forward trajectory but not on the cotangent being multiplied. A nonlinear credit predictor can also condition on error; it then predicts credit rather than being a linear adjoint operator. These are distinct declared classes.

A compact hypothesis class may factor K as an addressed gate times U_j D(dt) V_t^T, with rank r, learned feature-dependent factors and D(dt) made from decaying/rotating modes. Exact known temporal transport is reused rather than relearned; the learned part predicts remaining conditional influence and unresolved continuation credit. Causal support is mandatory. Locality, shared factors, rank, timescales and residual sparsity are inductive hypotheses, selected on development evidence and tested for failures.

## Inductive bias as transfer to unseen situations

Let B_psi(x) map an error e(x) into parameter credit. For a locally linear optimizer transformation P_omega, the first-order effect on another prediction is

    Delta f(x') = -eta J_theta f(x') P_omega B_psi(x) e(x) + O(||Delta theta||^2).

Thus T_(theta,psi,omega)(x',x) = J_theta f(x') P_omega B_psi(x) is the learned learning-transfer operator. Ordinary gradient descent has B = J_theta f(x)^T and P = I, giving the usual tangent kernel (note 155). Nonlinear update rules require their actual local derivative or measured finite-update effects; a global linear kernel must not be presumed. Learning psi/omega changes where an error transfers, with useful generalization and harmful interference measured separately.

A task-preserving label permutation has parameter action Q and output action R. The required credit equivariance is

    B_psi(g.x) R = Q B_psi(x),

for orthogonal permutation representations, with histories/states transformed consistently. More general coordinates transform gradients as covectors (inverse transpose). An observed key/value relabeling alone is not a symmetry when the learned embedding/output weights are left unchanged. Exact symmetries can be built into the class; domain-dependent sharing strengths are learned. Approximate statistical similarity is not an exact symmetry.

The native coordinate-sharing credit/update functions are structurally permutation equivariant only when their packets bind the correct actor parameter version. The initial symmetry gate exposed a stale-packet implementation: v1–v3 restored module embedding weights were read after a functional call, instead of its actor snapshot. Version-bound v4 packets repair this before development fitting. Prior residual-mean identities still hold with arbitrary/stale predictors, and prior smoke results remain evidence of finite native update execution; they do not certify transfer.

## What mathematical achievability does and does not mean

Let G be the target credit, with finite second moment, and I the retained information available to the credit function. For any I-measurable predictor h,

    E||G-h(I)||^2 = E||G-E[G|I]||^2 + E||E[G|I]-h(I)||^2.

The cross term vanishes by conditional expectation. Therefore the best unrestricted predictor is E[G|I], with irreducible risk E[tr Cov(G|I)]. If the target is determined by retained evidence and a sufficiently expressive class approximates it, there is no representational prohibition against arbitrarily small calibration error on that distribution. If information is lost, no learner can reconstruct every path-specific credit from that packet alone. More retained evidence or reconstructible audits can reduce the deficit. This is an information condition, not an arbitrary ceiling on the family.

For a forward transition derivative A, adjoint transport uses A^T. In reverse time u=T-t, a linear flow's adjoint evolves with the transposed generator; it does not require an inverse of dissipative state evolution. Stochastic time reversal instead uses Bayes conditional transitions and the state distribution. Adjoint transport, posterior cause attribution and finite-update utility are related learning targets, not interchangeable notions of time symmetry.

Learnability and grokking are stronger empirical questions than expressibility. A rich class can memorize without learning useful invariances. Exact audits preserve the declared raw-gradient mean and train future predictions; they do not prove convergence of the nonlinear coupled optimizer. Note 166 supplies a sufficient local contraction condition under its stated assumptions. Neither that condition nor universal approximation establishes a global nonconvex grokking theorem.

## Frozen-teacher transfer and delayed generalization protocol

First freeze native forward snapshots, so an improving actor cannot masquerade as an improving credit system. Fit one shared credit function on TRAIN packets from bounded native recall sequences. Report calibration and residual-audit variance on disjoint DEV sequences; test longer histories, unseen frozen parameter snapshots and deeper native graphs with no credit-parameter refitting. The current prototype predicts missing parameter credit as a correctness bridge; the scalable target predicts state cotangents and contracts them with charged local sensitivities (notes 164–165).

Use fixed final checkpoints and a predeclared observation schedule. Track TRAIN error, DEV error, unseen-depth error, gradient-direction agreement, positive/negative transfer and complete work. A delayed generalization transition requires low TRAIN error preceding a sustained later DEV improvement; label it observed delayed generalization first. A grokking claim additionally needs replicated runs, controls for teacher drift and exposure, and diagnostics of acquired rule/invariance. Failure to grok is not a failure of the model family.

Compare learned credit with a frozen initialization, local/no-missing-credit baseline, reset versus persistent credit state, and eventually structure ablations (elapsed-time transport, shared addressing, learned tail). Optimizer learning has a separate future-utility assessment and fixed-versus-learned control. Prefix evidence and any teacher/audit labels supplied at assessment are declared and charged; do not call a prefix-assisted task zero-shot transfer.

The later coupled experiment uses the same forward model, data/order, initialization, target protocol and validation opportunities under ordinary backprop/Adam, compact corrected credit with fixed updating, and reciprocal credit plus learned updating. Compare complete work to fixed quality targets at depths 4 and 8 before scaling. Charge actor execution, teacher/audits, credit fitting, meta-gradients, optimizer visits, outstanding records and discovery. State-sized messages or summable residual spectra are hypotheses to measure, not proof that baseline application and candidate discovery are free.

## Current owned next gate

Curie: version-bound native symmetry contract; bounded frozen-teacher calibration/transfer pilot; state-cotangent replacement and measured native fit. AWS: existing online3 depth grid, closed-credit pilots and reciprocal-update pilots remain owned there. No dense external architectures, sealed benchmark TEST reads or remote queue changes are introduced.

## Causal nonstationary inductive structure (founder clarification)

The sequence law and useful inductive structure may both change. Write the effective operator as K_(psi_t)(j,t; c_t,H_t), where persistent state c_t adapts quickly and slow structural parameters psi_t may also change. Stationary shared parameters do not imply a stationary effective kernel; conversely, changing the transition law may require learning the shared function itself. Hard causal/version contracts persist; regime-dependent locality, relevant delays, shared influence, audit allocation and update gains are learned preferences.

The adaptation must itself be causal. Let F_t contain only information revealed by time t, including labels when they actually arrive. Require (c_t, psi_t, omega_t) to be F_t-measurable and update them from newly revealed evidence, not a retrospectively inferred regime. A delayed outcome can teach the current credit system using its stored prediction-time packet and parameter version; it cannot alter the prediction or information available at that earlier time. A changing statistical association is evidence for adapting a preference, not proof of a changing causal mechanism. Use intervention evidence or explicit identification assumptions before interpreting learned credit as causal attribution.

Assess prequentially: emit each prediction with the state/parameter version available then, observe its outcome, update credit/optimizer state, and use that information only for subsequent predictions. Regime boundaries remain hidden unless an input legitimately provides them. Compare persistent state adaptation, parameter adaptation, both, and reset controls on smooth drift and abrupt switches, reporting adaptation delay and forgetting alongside stable-regime transfer. A frozen-teacher calibration probe isolates learning capacity; it is not the deployment protocol or an assumption that real-world biases are stationary.

For B10 market-event integration, forward and credit state use actual timestamps. Credit work is an addressed event process: local transport advances lazily, unresolved writes retain versioned records, competing credit clocks choose which messages/audits/updates execute next, and clock silence has an explicit cost/objective where it is part of the model. Learned scheduling receives counterfactual or score-function credit for unselected alternatives, with supported audits and complete discovery costs. Current prototypes use a learner-step clock and dense local backward; neither is reported as asynchronous market learning. Exact tiny event-clock/causality contracts and a prequential DEV smoke precede any market fit or efficiency claim.
