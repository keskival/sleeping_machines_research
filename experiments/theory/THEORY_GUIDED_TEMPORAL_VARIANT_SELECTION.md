# Choosing temporal variants by useful signal and experimental value

R1/B10 · 14 October 2026. Analytical selection rules; no new fitting admitted. Extends TEMPORAL_LAYER_FUNCTION_GEOMETRY_STATISTICS.md and preserves the currently admitted calibrated-clipping pair and market growth protocol.

## 1. Separate added depth from the benefit of more fitting

For an acquired parent P0, a continued parent PN and a grown successor GN with the same allowable additional data/update budget,

    R(P0)-R(GN) = [R(P0)-R(PN)] + [R(PN)-R(GN)].

The first term is continued learning; only the second isolates the successor's advantage over that continuation procedure. Comparing a trained successor only with a frozen parent cannot identify useful extra depth. If actual budgets, early stopping, optimizer reset or data order differ, record those differences rather than calling the contrast an isolated depth effect. Quality per complete work is a separate comparison from equal additional data.

This explains why a continuation control is valuable when a depth result is ready for a causal architectural claim. It does not automatically admit a control for every exploratory variant. The current B10 single-arm experiment remains development evidence; an improvement over its saved parent alone will not be described as isolated depth gain. A confirmation decision must consider a continued-parent contrast before spending on repeated successor seeds.

The existing AWS clipping pair has higher immediate language decision value than another mixed change: it starts from the same acquired learner and changes one update law. Norm matching does not prove direction matching or quality, so complete the pair before adding horizon or another cap grid.

## 2. A trainable neutral hierarchy has a cross-credit operator

Consider a fully contained feature branch with output coefficients a and internal parameters theta. Let its finite or population risk satisfy

    R(0,theta)=R0 for every admissible theta.

This is stronger than matching the parent at one initialization. Clocks, recurrent context, private state interfaces and every downstream use must satisfy containment. The market growth wrapper has this output property for finite internal values; the native timed language hierarchy still needs its separate clock/context construction.

Define the output teacher and its internal response

    g(theta)=partial_a R(0,theta),
    C(theta)=partial_theta g(theta).

At a=0, internal prediction gradients are zero even if C is nonzero. If g=0 as well, the Hessian is

    H = [[F,C],[C^T,0]],   F=partial_aa R.

When C!=0, this stationary point has negative curvature. Choose v with Cv!=0 and u=-epsilon Cv. Then

    (u,v)^T H (u,v)
       =epsilon^2 (Cv)^T F (Cv)-2 epsilon ||Cv||^2 < 0

for sufficiently small positive epsilon. This proof does not require F to be invertible or positive definite; finite local curvature suffices. A completely neutral branch can be a saddle rather than a function-class failure.

## 3. Error-conditioned feature seeding preserves prediction

Keep a=0 and perturb internal features by

    delta_theta = epsilon C^T u,

where u is a fixed output-space probe. The prediction remains exactly the parent's for every epsilon that retains the containment/finite-state contract. At a deadlock,

    g(theta+delta_theta)=epsilon C C^T u + O(epsilon^2).

Thus a probe outside null(C^T) produces a live output teacher without an initial quality penalty. For finite-dimensional continuous random probes with full support, null(C^T) has probability zero when C!=0; use private recorded randomness, not the actor's stream. This is an existence statement, not an estimate of adequate signal in finite data.

If the fixed-feature risk in a is L_a-smooth, an output step a=-eta g at the seeded theta gives

    R(0,theta_seed)-R(-eta g,theta_seed)
       >= eta(1-L_a eta/2)||g||^2,

so eta<=1/L_a gives at least eta||g||^2/2 reduction. This certifies a conditional first useful step, not a long-run optimizer result. Actual inherited Adam moments, clipping and further feature changes require the true displacement and further evidence.

Under a Euclidean trust radius ||delta_theta||<=r, maximizing first-order ||C delta_theta|| selects a top right singular vector of C. There is no need to materialize the full cross-Hessian: C^T u is one reverse-over-reverse derivative and C v can use a directional derivative of the output teacher. A few operator products can be compared with the cost of a full fit; no power-iteration count is intrinsically optimal. Coordinate metrics and parameter sharing must be declared.

`prediction_preserving_feature_seed.py` computes a checked C^T u and a bounded Euclidean direction. It returns tensors; it does not mutate a learner, consume RNG or perform an optimizer update. It rejects nonzero output maps and aliased parameter groups. Its derivative is of the issued differentiable loss. Hard routing's alternative-choice derivative and future learning are not automatically included; a native route-aware application must retain its supported counterfactual policy.

Cross-credit requires retaining a derivative graph and additional backward/operator work. Start with selected new feature leaves on a bounded declared TRAIN panel; measure memory and complete cost before an integrated fit. No claim that this is free or always cheaper than ordinary fitting follows from the formula. For a complete feature-coordinate gauge, g is constant along a gauge orbit, so C annihilates its tangent directions and C^T u is orthogonal to them in the declared Euclidean metric. This targets functional response rather than a change of hidden phase convention.

Exact XOR witness: for logit a(Z1+theta Z1 Z2), a=theta=0, C=-1/2. Internal seeding changes theta while leaving every prediction1/2, then makes the output teacher nonzero and permits a loss-decreasing output step. Unlike a nonzero-output perturbation, this pays no initial prediction loss. A temporal-memory fixture also verifies exact contained outputs after internal seeding and an increase in its output-teacher norm. These are unit witnesses, not corpus evidence or a queued new initialization.

## 4. Learned initialization should match a joint functional law

Trained weights can supply a prior, but matching their marginal histograms is not a useful sufficient condition. A scalar exact counterexample has B uniform on{-1,+1}. Both C=B and C=-B give identical marginal laws for B and C, but the composite CB is respectively+1 and-1. Independent B,C have the same marginals and zero expected composite. Matching means/variances cannot recover learned write/read coupling.

For modal memory, copy or sample whole compatible parameter tuples: write/read maps, keys/key-read maps, controls, decay/phase parameters and their sharing structure. Account for the new layer's input distribution; a trained shallow operator need not be calibrated on deeper transformed messages. Duplicate operators can add computation without new predictive signal.

Prefer gauge-invariant or function-space calibration: elapsed-time half-lives, modal phase evolution, paired transfer/residue operators, actual message and residual scales, score saturation, route occupancy/entropy and downstream error correlation. In a fixed linear modal regime, C_j (I-z^-1 A_j)^-1 B_j is invariant under a joint hidden basis change. Nonlinear gates and hard routes require issued local/trajectory measurements rather than promoting that transfer function to a global description.

A rational next initialization order is: preserve the acquired parent; reuse a compatible joint trained operator prior for new feature branches; verify finite times/state and containment; estimate g and cross-credit C on TRAIN; seed only if signal requires it; then assess useful learning and cost. We are not selecting independent Gaussian draws to mimic a trained weight histogram, and no distribution is declared optimal from this argument.

## 5. Do not train to a noisy alignment statistic without accounting for noise

For independent same-law TRAIN panels and output-gradient estimates ghat,

    E||ghat||^2 = ||g||^2 + trace(Cov(ghat)).

Increasing the empirical gradient norm can increase noise instead of useful population signal. With independent estimates ghat1,ghat2,

    E[ghat1 dot ghat2] = ||g||^2.

Likewise independent cross-credit estimates satisfy E[C1^T C2]=C^T C. These identities require independence conditional on the issued model and a common target law. Disjoint adjacent temporal windows are not automatically independent. Independently sampled, independently warmed panels from a fixed finite TRAIN corpus define an explicit empirical sampling law; that law is not automatically the deployment distribution. Shared route noise or learner updates between panels change the argument.

Averaged cross-products can be negative. Selecting the maximum over many probes produces optimism even when each fixed probe is unbiased. Choose a small theory-derived candidate set, keep assessment panels out of its selection, and preserve all attempts/work. Neither a positive training alignment nor a negative noisy estimate substitutes for completed development quality.

At g=0, optimizing ||g||^2 by its first derivative also stalls: its gradient2 C^T g is zero. A cross-credit probe is specifically useful for escaping this case. If C is zero within the available feature parameterization, allocating more ordinary output audits cannot create its missing first-order signal; change the feature operator or investigate higher-order structure instead of blindly extending acquisition.

## 6. Spend audit budget only where it can change useful updates

For the smooth fixed-preconditioner bound from the preceding note, put

    a=g^T M g, b=g^T M^2 g, v=trace(M Sigma M).

With B independent equivalent audit samples, a fixed step eta has a positive certified expected bound when

    B > v / [2a/(L eta)-b],

provided the denominator is positive. If the denominator is nonpositive, no finite B repairs that bound at the chosen step; reduce/change the update before buying more audits. If a=0, that first-order criterion cannot justify more of the same teacher. If signal exists but audit variance dominates, budget/allocation or a fitted exact-mean control variate addresses the identified failure.

This is conditional risk mathematics, not an automatic production-Adam admission rule. Calibrate using the actual displacement delta. At a common issued cut, the local risk change is

    R(theta+delta)-R(theta)
       = g^T delta + (1/2)delta^T H delta + remainder.

If Hessian is rho-Lipschitz on the entire segment, |remainder|<=rho||delta||^3/6. A held-out TRAIN evaluation gradient/curvature can assess the displacement formed on a separate fitting panel. Matching delta's norm alone leaves its alignment with g and its curvature penalty unspecified. Long-run state/route/optimizer changes require actual continuation, as in the admitted pair.

## 7. Experiment decisions

- Complete the matched clipping continuation before adding another learning change. A loss with retained norm points toward direction/moment or tail distortion, not an automatic horizon fit.
- Retain market growth as an owned development attempt, with a continued-parent comparison required for an isolated depth claim. Do not multiply random seeds before interpreting the correction and internal credit.
- For native growth, retain acquired gains/moments and solve clocks/context first. Then prefer compatible joint learned temporal priors and error-conditioned feature seeding over an arbitrary histogram-matched initialization or arbitrary nonzero coupling.
- If error correlation is absent but cross-credit is present, repair feature learning at a contained start. If useful signal is present but noisy, repair credit budget/allocation. If actual update geometry is harmful, repair the optimizer law. These are different variants with different decisive tests.
- No long fit follows from a unit witness alone. Register the mechanism, source cut, matching protocol, paid measurement cost and the decision changed. Use completed quality/full-work against richerD2, then the pinned public reference.

This narrows the variants; it does not establish a universally best architecture or a competitive language result. No extra queue or fit is admitted here.
