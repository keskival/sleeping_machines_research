# Causal forward feedback trains an update function

R1/B1 shared learning enabler, 11 October 2026. This note defines the implemented local objective and its asynchronous extension. It does not treat a failed first pilot as a model-family verdict.

Let θ_t be forward parameters, Ω_t optimizer state, x_t the current causal packet, and ψ_v the version of a learned update function. After observing target y_t, compute g_t and issue

    (θ_{t+1}, Ω_{t+1}) = U_{ψ_v}(stopgrad(θ_t, Ω_t, x_t, g_t)).

Retain the derivative tape A_t = ∂θ_{t+1}/∂ψ_v, or an economical representation permitting its action. On the next target, forward computation supplies g_{t+1} = ∂ℓ_{t+1}/∂θ_{t+1}. The exact local meta-gradient is

    a_t = A_tᵀ g_{t+1}.

This is exact for the one-update objective holding the incoming state fixed. It is not the derivative of the entire continually adapted trajectory. A full horizon also includes optimizer-state sensitivity, later updates and shared-state dependence. A truncated objective can prefer a useful immediate step yet worsen cumulative loss. Preserve that distinction in every quality claim.

## Fixed-version aggregation

Issue K updates at a pinned ψ_v. Each forward target releases its own update tape after producing a_t; numeric learner state continues to evolve. The commit uses ā = K⁻¹ Σ_t a_t. This is the exact gradient of the mean of the K declared local surrogate objectives, each with its incoming state held fixed. It is not the gradient of the K-target rollout with all state dependencies restored.

For random feedback, conditional covariance is

    Cov(ā) = K⁻² Σ_{i,j} Cov(a_i, a_j).

Independent identically distributed samples give a 1/K variance reduction; correlated causal sequences can retain substantial variance. Bias of the local surrogate is not removed by averaging. Aggregation increases response delay and may weaken regime tracking. The batch size is a resource/latency choice, not a mathematical guarantee of better learning.

Parameter version is part of the tape. Differentiating a retained issued graph after an in-place function update is invalid. The present implementation pins the version until aggregation completes, consumes each pending graph before a commit, and issues the next graph only afterward. End-of-phase commits flush partial feedback; EOS discards unrealized pending feedback while retaining already observed gradients. Feedback is causal even when a parameter update happens after an episode boundary.

## Toward sparse asynchronous learning

A clockless implementation can associate a tape with an event id, parameter version and addressed learner state. It issues only the selected update, queues feedback when the supervised event arrives, and accumulates into addressed credit-parameter blocks. Useful economy requires bounded tape support and inexpensive application of A_tᵀ, not merely a small message carrying a dense global derivative.

Concurrent parameter versions require immutable snapshots or an explicit stale-gradient algorithm. A derivative at ψ_v is not automatically a derivative at ψ_current. If a gradient map is L-Lipschitz, the discrepancy is bounded by L times the version displacement, but L and displacement must be measured or bounded. A reset/cancellation protocol must prevent an old event from updating a new numerical state without its declared semantics.

The Curie diagnostic uses a dense 64-dimensional lexical state, a 25,027-parameter connected function and a 64×64 solve. It measures forward teaching of the update function within native temporal/keyed inference; it does not demonstrate sparse backward hardware execution. Integration should replace or factor costly actions only after derivative parity and quality/work tests. Optional writes, counterfactual correction and continuous native producers are separate mechanisms to retain in the integrated model.

## Current evidence and decisive tests

The one-target commit pilot fails its primary gate: online-over-frozen gain −0.031139 nats/target on four 512-target private episodes. The two largest regressions coincide with strong step-size growth and larger lexical-state norms. This is a measured association, not proof that feedback variance causes the regression. The registered sixteen-feedback experiment uses fresh private episodes, repeats disclosed pretraining and records both feedback and optimizer-update counts.

A useful result must first improve prediction relative to the same frozen pretrained function with full acquisition and fitting costs. Next tests extend the objective horizon, restore optimizer-state teaching, and compare integrated native producer/route learning across depth. Hardware claims require measured tape memory, addressed traffic, latency and energy. No public benchmark score or investor headline follows from an algebraic contract alone.
