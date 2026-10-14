# Temporal layers: function, composition, geometry and statistical learning

R1 · 14 October 2026. Derived for the native sparse token core, with restricted linear examples labelled explicitly. This is analytical development, not measured language improvement. Read alongside STAGE_GAIN_LANGUAGE_DEPTH.md, theory34 identity-depth contracts, and LOCAL_UPDATE_GEOMETRY_AND_USEFUL_DEPTH.md.

## 1. The actual layer is a stateful temporal transducer

Write a stage as

    F_l : (x,t,S_l,xi_l) -> (x',t',S_l'),
    S_l = (M_l, cached_keys_l, write_times_l, seen_l).

M contains private real-valued modal memories for every head/receiver. The message x and time t are interfaces; xi supplies exponential race noise. This signature separates the emitted representation from available persistent capacity. Receiver identities are discrete; memories, timestamps and messages are continuous. Current inference scores all cached keys, chooses one receiver per head and writes only those memories. Sparse writes do not imply sparse key discovery or zero optimizer cost.

The equations below follow sparse_training._unit and sparse_counterfactual_layer, omitting batch indices. Set z=W_l x, split z into head messages z_h and form q_h=Q_lh LN(z). For receiver r,

    s_hr = clamp(q_h dot cached_key_hr / sqrt(P) + b_hr, -12,12),
    T_h = min_r xi_hr exp(-s_hr),   w_h = argmin_r xi_hr exp(-s_hr),
    d_h = .001 + .010 T_h/(1+T_h).

Only the winning receiver's state is evolved and written. For its previous write time tau and Delta=max(t-tau,0),

    M^- = decay(Delta, rate, forget(z_h)) * rotate(Delta, frequency) M,
    M^+ = M^- + write(z_h) B z_h,
    y = LN(C M^+ + z_h),
    v_h = z_h + alpha_l y sigmoid(G GELU(y)+b),
    cached_key^+ = k + K M^+.

Here forget=softplus(control_0)/log(2), write=2 sigmoid(control_1); multiplication of the gate and y is coordinatewise. The memory write is stamped at t. Emitted head values have delivery times t_h=t+d_h; head transport aligns them at t'=max_h t_h before concatenation into x'. The transport is another learned decay/rotation, separate from the private memory dynamics.

Thus a layer does five different things: content-dependent selection, elapsed-time memory evolution, addressed writing, gated residual correction, and time-dependent delivery/alignment. In general it is not x+alpha f(x): W_l, hard selection, memory and transport all matter. Alpha=0 removes one correction term but does not remove W_l, memory writes, clocks, transport or recurrent context effects.

The emitted value is factual winner-only. Learning includes factual differentiation and finite-horizon alternative-route audits. The implementation's first-time derivative -T softmax(s) is a declared factorized race-time surrogate, not the ordinary derivative of min_r xi_r exp(-s_r) under fixed individual xi (which is winner-only away from ties). Those derivative policies must not be conflated with the forward function or with the complete derivative of an arbitrary long-run learning objective.

## 2. Composition creates a hierarchy of dynamical summaries

Within an event, lifted composition is

    F_D o ... o F_1,

where each stage has its own S_l, while x and t pass forward. Across events, the complete model is a stochastic state transition

    (S_{n+1},h_n) = Psi_theta(S_n,u_n,xi_n).

S also includes final delivered context and its timestamps: this context feeds the next token's source message. Deeper memory receives representations already shaped by upstream selection, memory and timing. This is the precise hierarchy: later stages summarize transformed trajectories rather than merely storing another copy of the token. It need not become useful automatically.

Conditional on a fixed route itinerary and fixed clamp regimes, finite executions are differentiable compositions of gates, modal evolution and normalization. Their Jacobians act on lifted message/time/state coordinates, not just the payload. Changing a route changes the branch. A continuous-path Jacobian alone cannot assign credit to an unchosen write. Parameter sharing also couples nominally separate stages/receivers; shared leaves must be counted once.

A restricted linear example shows what composition can add before discussing nonlinear routes. For unit-spaced events and fixed linear recurrent layers,

    m1[n]=a m1[n-1]+b u[n],
    m2[n]=c m2[n-1]+d m1[n],

an impulse at n=0 produces

    m2[n]=bd sum_{j=0}^n a^j c^(n-j)
          =bd (a^(n+1)-c^(n+1))/(a-c),   a != c,
          =bd (n+1) a^n,                a = c.

D identical cascaded modes give b_product * binomial(n+D-1,D-1) a^n. They can represent delayed, broader response shapes absent from one scalar mode. This is a discrete event-sampled cascade, not a claim that our nonlinear receiver banks are continuous convolutions. With irregular gaps, replace a,c by event-dependent decay/rotation factors; the impulse response becomes a sum over ordered intermediate event indices. Content gates and routing make those kernels history-dependent.

For an ideal identity-residual linear cascade, product_l (I+alpha_l K_l) expands over all ordered subsets of temporal operators. This exposes short and long computational paths. Our native projected/timed stages do not automatically satisfy that identity form. The implemented per-stage gain adapter removes one configuration restriction; true contained growth still requires clock/context/optimizer contracts.

## 3. Representations and their geometry

**Memory:** each real payload is equivalently C^(P/2), one complex number per decay/rotation mode. With writes disabled and no decay, fixed amplitudes trace phase circles; independent phase coordinates form a torus. A single elapsed-time parameter traces a one-dimensional orbit, whose closure may fill a torus for rationally independent frequencies. Positive decay gives damped orbits, not a fixed torus. Writes add vectors and change amplitudes, so general stored memory occupies a subset of Euclidean R^P, not necessarily a low-dimensional manifold.

**Normalized features:** the kernel uses LayerNorm with positive epsilon. For centered v=z-mean(z),

    LN_epsilon(z)=v/sqrt(||v||^2/P+epsilon),
    sum LN=0,   ||LN||^2=P||v||^2/(||v||^2+P epsilon) < P.

Its image is the open radius-sqrt(P) ball in the mean-zero hyperplane, dimension P-1. It is not exactly a sphere. Only the epsilon=0 idealization has a sphere of dimension P-2 and exact positive-scale invariance. A learned affine normalization transforms that geometry; subsequent residual addition and gating do not confine the complete payload to the normalized ball.

**Routing:** for fixed keys, unclamped score comparisons give winner regions defined by score/noise inequalities. These regions are smooth only piecewise; clamp surfaces and race ties create boundaries. The full representation support is a hybrid union of route-conditioned continuous pieces. Finite empirical clouds, covariance effective rank and a global smooth manifold are different objects. Discrete token histories alone do not justify a manifold assumption; timestamps and race randomness supply continuous coordinates.

**Prediction:** a finite-vocabulary normalized head maps features to the interior of the probability simplex. With an affine full-softmax head, log-odds lie in an affine subspace of dimension at most feature width, while probabilities form a curved exponential-family image. An adaptive head adds conditional branches but is still a smooth function of the available feature vector. Low empirical logit covariance rank is not a proof of missing predictive information or of an intrinsic representation dimension.

**Parameter geometry has an exact hidden-state gauge.** Let R be block diagonal, with a planar rotation in each modal pair. Rotate a private memory basis by M'=RM and transform

    B'=RB,   C'=CR^T,   K'=KR^T.

Keep keys, controls, gates, rates and frequencies unchanged. R commutes with the paired modal decay/rotation. Induction gives M'^+=RM^+, unchanged cached keys, routes, times and emitted values. The hidden basis is therefore non-identifiable. This is a symmetry of a unit's memory coordinates, not of its incoming message coordinates. At a stationary point the loss Hessian has zero directions tangent to these continuous gauge orbits. Parameter dimension is consequently not identifiable functional dimension.

Euclidean SGD is equivariant under this orthogonal reparameterization. Diagonal Adam generally is not: coordinatewise squared moments cannot express a rotated full second-moment matrix. Even the first-step map g -> g/(|g|+epsilon) is not rotation-equivariant. Functionally equivalent temporal representations can therefore have different Adam trajectories. Per-bank radial clipping preserves an orthogonal bank rotation's gradient norm, but does not repair Adam's coordinate dependence. No coordinate basis is inferred optimal here.

## 4. Statistical meaning of useful depth

Let H be the parent's representation and Z the added temporal representation. If both decoders are unrestricted Bayes predictors, the optimal expected log-loss gain is exactly

    H(Y|H)-H(Y|H,Z) = I(Y;Z|H).

This defines predictive information beyond the parent. It is a population statement. Finite decoders, limited fitting data, optimizer effects and distribution shift can dominate the observed gain. Z may improve a restricted decoder by reshaping information already in H even when I(Y;Z|H)=0; conversely, useful information in Z can remain inaccessible to the chosen decoder or learner.

For fixed features and additive affine logits, use parent tangent features and new branch tangent features. The log-loss Hessian is the categorical Fisher matrix F. Partition it into parent/new blocks. Local reoptimization of the parent gives the quadratic new-direction quantities

    g_eff = g_Z - F_ZH F_HH^dagger g_H,
    F_eff = F_ZZ - F_ZH F_HH^dagger F_HZ.

Under the usual quadratic range/curvature conditions, the best local quadratic reduction is (1/2) g_eff^T F_eff^dagger g_eff. This follows by minimizing over parent directions first, then new directions. Singular unpenalized directions with nonzero gradient require separate treatment; a quadratic approximation is local, not a finite-step guarantee. The actual nonlinear model Hessian can contain additional second-logit-derivative terms, so one must not silently replace it everywhere with Fisher.

The meaning is concrete: added features need prediction-error alignment after accounting for what the parent can already learn. More parameters, nonzero memory activity, spectral rank, and an arbitrary nonzero gradient are not certificates of new predictive information.

## 5. Exact containment can create a learning deadlock

Take independent fair signs Z1,Z2 and target Y=(1+Z1 Z2)/2. Let a new branch have logit

    logit = a (Z1 + theta Z1 Z2),

with the parent predicting1/2. The available pair (Z1,Z2) determines Y, so its conditional information over this constant parent is log(2). At a=theta=0, both population gradients are exactly zero. The output map has no error correlation and its zero value blocks internal feature credit. Exact zero-output insertion can therefore be stationary despite available useful information.

The Hessian in (a,theta) at zero is

    [[1/4, -1/2],[-1/2, 0]],

which has a negative eigenvalue: it is a saddle, not an irreducible quality floor. A small nonzero a=epsilon,theta=0 costs exactly log(cosh(epsilon/2)) expected NLL, or epsilon^2/8+O(epsilon^4), and gives theta-gradient -epsilon/2. Thus bounded nonzero coupling can unlock internal learning at second-order initial quality cost in this witness. Real-language feature correlations and finite-sample noise must decide whether exact containment or a bounded perturbation is the better initialization; neither is universally best. Do not claim the zero-output market growth has solved this obstruction from its nonzero synthetic gradient fixture.

## 6. Statistical credit and depth budget

At an issued cut, suppose a new-coordinate risk is L-smooth, M is a fixed positive-semidefinite update preconditioner, and an unbiased gradient estimate has mean g and covariance Sigma. Before nonlinear clipping or moment adaptation,

    E[R(theta-eta M ghat)] <= R(theta) - eta g^T M g
       + (L eta^2/2) [g^T M^2 g + tr(M Sigma M)].

For positive signal a=g^TMg and b=g^TM^2g+tr(M Sigma M), eta=a/(L b) gives a certified bound reduction a^2/(2 L b). Averaging independent audits divides their covariance by sample count; correlated or adaptively coupled audits require their actual covariance. A small gradient can be useful if noise/curvature are correspondingly small. Global clipping and Adam are nonlinear history-dependent updates, so the unbiased raw teacher theorem alone does not certify this bound for the production optimizer.

For disjoint layer contributions Z_l, select a layer with predictable full-support probability q_l and use Z_l/q_l. With second moments A_l=E||M Z_l||^2, its preconditioned second moment is sum A_l/q_l. Equal demands and q_l=1/D give D^2 A; the complete mean-squared signal scales only D for equally sized disjoint banks. A fixed audit count consequently loses aggregate signal/noise ratio as depth grows. Nonuniform allocation helps unequal demands but cannot eliminate the cost of learning D equally important independent contributions.

If a sampled layer costs c_l and the expected total audit budget is C, the raw-second-moment noise bound is proportional to

    (sum A_l/q_l)(sum q_l c_l)/C.

Cauchy-Schwarz minimizes this bound at q_l proportional to sqrt(A_l/c_l), before imposing a positive exploration floor. This is a second-moment-bound optimum, not the exact variance optimum after subtracting the mean term or a wall-time optimum. Complete transported parameter/update demand differs from a local score-gradient proxy; our current allocator measures the latter.

Finally, rescaling a branch by alpha scales its gradient by alpha and Fisher block by alpha^2. An ideal local natural-gradient quadratic benefit is invariant for alpha>0, whereas fixed-rate Euclidean learning slows with alpha^2. Smaller residual gains need not remove representational potential, but can change acquisition time. In the actual family they also change the learned forward function, and changing depth currently rescaled old stages. Per-stage gain control and acquired-learner migration now let a future intervention separate these effects.

## 7. Decisions this analysis supports

1. Describe/develop a hierarchy of persistent state transducers, charging selection, timing, state and alternative learning separately.
2. Use controlled growth that retains acquired gains and optimizer state; require separate clock/context contracts. Do not infer full identity from gain0.
3. Check prediction-error correlation and meaningful internal feature credit at the proposed insertion, rather than only gradient nonzeroness on a convenient fixture.
4. Treat neutral zero-output insertion as one candidate with a known deadlock; consider a bounded nonzero coupling only with TRAIN-derived quality/credit evidence.
5. Preserve phase-gauge equivalence in representation explanations; diagnose update geometry before attributing a bad Adam trajectory to a weak temporal function class.
6. Budget actual deep credit against useful update signal, not a fixed total number of audits irrespective of depth. Adopt any repair only on completed quality/full-work comparisons against richerD2 and then the pinned public protocol.

No new training, queue, public-validation selection or competitive claim is admitted by this derivation.
