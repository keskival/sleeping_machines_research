# Route similarity is similarity of causal response

Curie, 12 October 2026. B1/R1 shared learning enabler; design theory supporting the next native integration. This is a derived proposal, not a measured advantage or a claim that kernels, causal models or operator learning are new. Connects [native outcome-law credit](NATIVE_OUTCOME_LAW_CREDIT.md), [reciprocal shared memory](SHARED_MEMORY_RECIPROCAL_EVENT_MACHINE.md) and [forward-observed risk](FORWARD_OBSERVED_CREDIT_RISK.md).

## The antenna intuition needs an operational definition

A route is a timed computation: addressed reads, selected messages, receiver writes, delays and the learning transitions they cause. Sharing an edge is a structural feature. Sharing a useful response to future errors is the reason to share credit. Two disjoint paths can perform equivalent computations; two paths through the same node can have opposite effects because of content, arrival order, memory version or producer sign.

Specify the issued causal state X, route intervention a, continuation policy pi and horizon. X includes forward/critic memories, clocks, available history, parameter versions and relevant optimizer state. Forced selection, changing a delay and changing a write destination are different interventions. The route includes its intervention interface; comparing an address with a parameter update without specifying their units has no mathematical meaning.

The relevant bias is a reusable regularity of this response law. A node-ID equality feature is an input prior, not proof that the learned system acquired the generating process. A useful learned bias predicts consequences at untrained routes, times or structures under a declared transfer protocol.

## A family of geometries rather than one static distance

Suppose a causal simulator or paid paired intervention supplies a common exogenous continuation xi. Let O_h^a(X,xi) be the future task observable at horizon h after do(a), including the declared continuation learning policy. Use a common reference a0 and define

    S_X(a;xi,h) = O_h^a(X,xi) - O_h^a0(X,xi).
    d_X(a,b)^2 = E_xi sum_h w_h ||S_X(a;xi,h)-S_X(b;xi,h)||_G_h^2.

w_h>=0 and fixed positive-semidefinite G_h declare which observable differences matter. The same reference cancels from pair distances. A common observable interface is necessary: compare predictive distributions or aligned producer effects, not arbitrary hidden coordinates from different nodes. A fixed metric on observables avoids treating a mere rescaling of a hidden representation as a new functional relationship.

This is a pseudometric when all signatures share this Hilbert space: zero distance identifies routes equivalent for the selected observables, continuation law and horizon. It is not equivalence for every possible task. Hard races, delayed writes and optimizer changes are allowed because finite interventions need no derivative through the winner. Event counts alone cannot replace physical arrival time in the continuation.

The coupling matters. Independent rollout noise would add noise distance even between identical stochastic routes. Common exogenous randomness defines a particular causal response geometry and requires an actual structural coupling. If only intervention-conditioned observable laws are available, a Wasserstein distance on those laws is a different admissible geometry; it does not identify individual paired effects. Observational similarity alone cannot identify unseen intervention responses without support or causal assumptions.

Do not reduce the signature immediately to one realized loss. Equal loss can hide opposite predictive effects and a different response to tomorrow's targets. Use several horizons and task-relevant output directions, then project into the current objective. Geometry changes causally with the inferred regime and learner state; stationarity is not assumed.

## What closeness actually guarantees

Let c_X(a)=E sum_h w_h ell_h(O_h^a,Y_h), with the same exogenous Y_h under both interventions. Suppose ell_h is L_h-Lipschitz in the declared G_h seminorm on the reached states. Then

    |c_X(a)-c_X(b)| <= sqrt(sum_h w_h L_h^2) * d_X(a,b).

Proof: apply the Lipschitz bound per horizon, Cauchy–Schwarz over horizons, then Jensen over continuations. Therefore copying an estimate of c_X(b) with error epsilon_b incurs at most epsilon_b plus this distance bound. BCE is not globally Lipschitz in probability near 0/1; finite logit coordinates or a justified bounded domain are needed. If actions change target/environment laws, include the changed outcomes in O and assume Lipschitzness of the complete cost there instead of reusing a common exogenous target.

A learned distance needs its own error bound. If ||S_hat_a-S_a||_H<=eta_a and likewise for b in the signature Hilbert norm, then

    d_X(a,b) <= d_hat_X(a,b) + eta_a + eta_b.

Small embedding distance without calibrated eta is not a transfer certificate. No theorem here establishes these constants for the present native models.

## Credit operators and error-dependent similarity

For smooth local transport, let T_a map a downstream cotangent e into a common observable update interface. This might be a producer-parameter correction followed by its future predictive effect. Align the interfaces and fix a positive-semidefinite metric G. For downstream error covariance C=E[ee^T|X],

    d_C(a,b)^2 = E ||(T_a-T_b)e||_G^2
               = tr[G (T_a-T_b) C (T_a-T_b)^T].

This says precisely why routes may be similar for one class of future errors and different for another. A transport can disagree in a direction the current data never excites; that disagreement becomes important after a regime change. Sparse forward probes can observe supported response directions without storing dense Jacobians. Their acquisition, local derivatives, discovery and fitting are charged. Probe support cannot certify unobserved directions.

Route-value equality does not imply transport equality. Two equal-valued routes may need opposite parameter updates. Conversely different present losses can share the same error-transport mechanism. Keep finite route selection credit, local derivative transport and acquisition value as typed projections of a shared world belief.

Optimizer-dependent geometry compares future predictive effects of the actual update, including moments, clipping and versions. Raw parameter-gradient Euclidean distance is only a declared approximation. Nonlinear optimizer response requires finite interventions or the full local augmented-state derivative.

## A constructive specialization for the next model

In exogenous binary prediction, one native world belief q_X=P(Y=1|X) yields R_X(a)=softplus(z_a)-q_X z_a. The slope of a pairwise risk contrast with respect to an error in q is -(z_a-z_b). Thus posterior calibration error epsilon changes that contrast by at most epsilon*|z_a-z_b|. This is a task-derived sensitivity, not an arbitrary embedding metric.

With latent world uncertainty W, suppose a declared sufficient-statistic projection has risk R(a)=A_a-mu^T f_a. Under posterior covariance Sigma_X=Cov(mu|available evidence),

    Var[R(a)-R(b)|evidence] = (f_a-f_b)^T Sigma_X (f_a-f_b).

This identity concerns epistemic uncertainty over the projected conditional mean, distinct from realized-label noise. Directions already learned need few additional credit observations; uncertain directions need evidence. Shared uncertainty yields coherent pairwise differences rather than unrelated loss heads. Neither a learned posterior covariance nor calibrated epistemic uncertainty is provided by the current scalar native critic; implementing it is a proposed capability.

The innovation target is a native temporal causal response model that learns reusable computational relationships, with coherent objective projections and paid uncertainty-directed interventions. It is not merely replacing a distance formula in nearest-neighbor regression.

## Depth: learn composable response, test the residual

Similar complete responses can arise from different numbers of internal events. A representation should allow those programs to share credit without collapsing their work or delay accounting. For smooth transport composition T_total=T_1...T_D, the telescoping identity bounds substitution error by

    ||T_total-T_hat_total|| <= sum_i
      (product_{j<i} ||T_hat_j||) ||T_i-T_hat_i||
      (product_{j>i} ||T_j||).

This explains amplification, not automatic depth advantage. Learn signatures for useful multievent blocks directly, compare block predictions with paid realized responses, and refine the blocks whose residuals matter for future task effects. Such block teaching can avoid repeatedly composing inaccurate local estimates; it needs actual long-horizon information and sufficient representation. A block residual is a training signal, not a contraction proof. Hard route programs use finite-response blocks instead of this smooth product identity.

Adaptive reset/transport of credit memory must preserve forward memory and carry version/context information. Old response evidence is reusable only when its predicted response changes remain within the calibrated transfer budget. Unknown drift cannot be wished away by encoding time.

## Credit compression has a precise task quotient

Suppose, at a fixed issued state, route risks have the sufficient-statistic form R(a)=A_a-mu_X^T f_a. Fix a reference a0 and let V_X=span{f_a-f_a0: a in the declared route pool}. All pairwise credits depend on mu_X only through its orthogonal projection P_V mu_X. Two world beliefs mu and mu' produce identical credits for the entire pool exactly when mu-mu' lies in V_X's orthogonal complement. This is a functional quotient induced by interventions and loss, rather than an arbitrary low-dimensional bottleneck.

Proof: pairwise risk equality requires (mu-mu')^T(f_a-f_a0)=0 for every a, which is precisely orthogonality to the span. In a linear representation, the required number of independent coordinates is rank V_X, provided the possible beliefs vary in those directions. No universal bit bound follows: precision, uncertainty, additional objectives and future route pools can enlarge the requirement. The map f_a may itself need costly causal computation.

This yields a constructive architectural decision: learn a rich connected native world state, but expose credit through the task's currently relevant response directions. Capacity of the state and size of the emitted message need not match. A small message can encode a projection from a large learned causal model; small isolated estimators are not required.

For BCE, d=1 and f_a=z_a. The outcome belief supplies all present route-risk contrasts; their outcome-induced covariance is q(1-q) v v^T with v_a=z_a-z_a0, rank at most one. Independent route-loss heads can waste effort fitting correlated noise. Coherent projection represents this shared direction exactly, while prediction errors in q also correlate all route credits. Shared structure is an opportunity and a failure mode, so calibration must be measured.

When new routes, horizons or regimes expand V_X, old compressed messages may be insufficient. Keep richer persistent evidence or pay for new observations; do not treat yesterday's quotient as a stationary inductive bias. At deep producer boundaries, known local maps still project output credit into parameter effects, or a separately learned and validated response map must replace them. Outcome compression does not erase that transport cost.

## Implemented first step (12 October)

The registered WORLD_CREDIT_PROTOCOL.md implements one native target-addressed outcome query and revealed-outcome update, shared forward/critic memory and coherent BCE route projections. Its common-acquisition exact/learned/fixed-prior branches isolate outcome-based source-choice teaching with the same memory capability. Contracts pass actual native outcome/read gradients, future task memory gradients, causal timing and public relabeling. The acquired q and risk algebra are the first quotient implementation, not a learned general causal-response distance or depth replacement. The broader geometry comparison remains the next test after useful causal credit reuse.

## Decisions this formalism changes

1. Preserve public relational addresses as candidate-discovery features. Learn response geometry from outcomes and supported interventions; do not call address overlap the similarity target.
2. Use the same native sparse temporal machinery for the rich world/credit representation. Persist typed route/event evidence with content, producer/version, delay, revealed outcome and declared continuation. Forward and critic choose their own reads/writes.
3. Share one world belief across route projections when the conditional outcome law is common. Retain action-conditioned response when reads/writes change future learner or environment state.
4. Transfer credit using predicted response similarity plus calibrated uncertainty. Unsupported alternatives receive uncertainty and an exploration decision, not fabricated confident credit.
5. Learn discovery as well as credit: the forward actor may propose routes that improve future prediction and identify uncertain response directions; the critic proposes useful alternatives. A candidate absent from the pool can receive an extrapolated prediction, but validating it requires positive exploration support or explicit structural assumptions.
6. Admit a probe only for expected future decision improvement exceeding declared acquisition/learning/delay cost. Trace reduction alone need not improve decisions; evaluate effects on route ranking and future learning. Avoid extrapolating the local covariance identity into a universal value-of-information theorem.

## Decisive next tests within B1/R1

Keep the previous four shared-bank fits and their failed adoption gates. Stop further 512-unique-target patches. In the next integrated native pipeline, compare (a) structural/address similarity, (b) learned predictive-content similarity and (c) learned causal-response similarity, at matched charged work and with the same acquisition data. Use disjoint but functionally equivalent routes, overlapping routes with opposite effects, delayed competing writes, fresh public address relabelings and causal regime changes.

Freeze issued signatures before their audit outcomes arrive. Score response calibration, uncertainty coverage, unsupported-route behavior and future task improvement separately. Separate benefit from better forward memory from benefit from learned credit. Compare direct block teaching with composed local teaching across neural depth, with full fitting work to matched quality. A similarity model is useful when it transfers correct future teaching and reduces total work, not when its embeddings look clustered.
