# Protected native language growth: the controlled scaling step

14 October 2026. R1 implementation: `sleeping_machines/protected_depth_language.py`; fitting: `experiments/protected_depth_continuation.py`. Start from the strongest completed proper-token language member, the acquired two-stage all-depth-readout model (paired DEV NLL 6.099337041). Add two genuinely native temporal stages with private addressed memories, computational delays, keys/values and hard races. Preserve the parent feedback, gains, trained readout and parent-owned random streams. Initialize new temporal operators from a **joint** trained stage, not independent parameter histograms.

## What is protected, and when the correction exists

Let the trained parent emit its stored feature tuple z at parent availability t_p, while a private branch emits phi at t_b >= t_p. The endpoint readout uses

    z_corrected = z + W phi,     W_initial = 0.

This is a statistical readout correction over separately referenced stored features, not an assertion that a branch message is available at t_p. The parent service can return its original prediction at t_p; a nonzero correction requires branch availability t_b. The current eager implementation evaluates both paths and scores the corrected endpoint. Its `query_arrivals` field in parent-output mode describes parent-feature availability; it is **not** the corrected prediction's service deadline. Parent recurrent feedback uses the original parent values and their original timestamps even while the new private state evolves. There is no measured physical asynchronous speedup in this implementation.

This distinction prevents prediction containment from being mistaken for latency containment. Learning optional emission, deadline-conditioned query selection and informative silence is subsequent work; the current test measures whether independently stateful added computation is worth having before sparsifying its activity.

## Why the zero start can learn, without an invented activation claim

For differentiable loss ell(z+W phi_theta), the initial derivatives are

    grad_W ell = grad_z ell * phi_theta^T,
    grad_theta ell = J_phi(theta)^T W^T grad_z ell = 0 at W=0.

A useful first output update can therefore open internal learning on subsequent updates. This is not guaranteed: orthogonality can also make the output gradient zero, and clipping plus Adam need not improve risk even when raw credit is live. The warmed 64-update native pilot produces a nonzero correction map; the subsequent TRAIN gate measures nonzero added-stage internal gradients and verifies a real local update and its exact recovery. These gates authorize a controlled quality experiment, not a depth-quality claim. No unsupported statement that new-route teaching alone defeats the zero-output deadlock is needed.

## Preserve acquired teaching rather than dilute it with depth

Condition on the issued factual packet, including its state and clock tape. Let X_o and X_n be supported importance-weighted old-stage and new-stage audit estimators, with conditional means G_o, G_n. Independently sample Bernoulli audits B_o, B_n with probabilities p_o, p_n. The raw correction estimator is

    G_hat = B_o X_o/p_o + B_n X_n/p_n.

With conditionally independent audit choices, its conditional mean is G_o+G_n and trace variance is

    E||X_o||^2/p_o - ||G_o||^2
      + E||X_n||^2/p_n - ||G_n||^2.

The independence statement is conditional on the same issued factual packet; it does not remove variation of that packet across training. Uniformly choosing one site over all four stages would halve each old stage's audit opportunity relative to the two-stage parent. The implemented two streams retain the old Bernoulli rate 0.25 and original old-site distribution, and pay an independent 0.25 new-stage stream. Each stream's site inverse probability uses its own two-stage support. Alternative routes retain a 0.1 uniform mixture for full support. Hypothetical suffixes rewind both parent and private branch clock noise to the issued cut and restore factual-end private noise afterward.

For a fixed expected audit cost C and costs c_i, minimizing the conditional variance proxy gives p_i proportional to sqrt(E||X_i||^2/c_i), capped at one. This follows from the Lagrange derivative -E||X_i||^2/p_i^2 + lambda*c_i = 0. We do not deploy this unmeasured optimum here: probabilities remain fixed so depth and teacher opportunity are explicit. Nor does unbiased raw credit imply unbiased locally clipped or Adam-applied displacement.

## The experiment that decides the next step

Two arms continue the same acquired actor, optimizer moments, numerical state and original random streams. Both have the completed 64-update warmup and receive 4,096 additional updates (2,097,152 token targets), with the same fresh TRAIN blocks and order. Each arm calibrates fixed local receiver caps at its actual warmed actor: 24 calibration plus eight heldout TRAIN chunks, aggregate cap budget 1.25, complete bank schema. Offline global reductions and calibration exposure are paid; subsequent clipping uses local bank norms. The extra branch and new teacher are charged to the grown arm.

Use one predeclared final endpoint: 8,192 shared reused DEV targets with 1,024-token histories and no EOS reset. The controlled construction gain is

    NLL(continued parent) - NLL(grown model).

Report quality together with actual fitting time, teacher replay/scoring and capacity. This is a matched-data development comparison, not iso-FLOP supremacy and not a public-test claim. A positive result selects this complete protected hierarchy for the next cost/scale test; a negative result triggers branch contribution/credit error analysis, not another blind depth increase. The branch causal ablation is required before attributing a gain specifically to its temporal hierarchy rather than the complete correction construction.
