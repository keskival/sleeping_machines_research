# Learned feature geometry is an inductive bias in credit

R1/B1,11 October2026. Derived for the proper-token forward/reference integration. The installed kernel currently uses Adam on a64-dimensional fast lexical state; the SGD/preconditioner identities below explain the geometry and define a next learner, not guarantees about that Adam implementation.

## The forward representation supplies a credit geometry

Write the fixed carrier's native token logits as ℓ_t(v), and the learned token feature row as e_v. A fast state z gives

    p_t(v;z) = softmax_v[ℓ_t(v)+e_v·z],
    C_t(z) = −log p_t(y_t;z),
    μ_t(z) = Σ_v p_t(v;z)e_v,
    ∇z C_t = μ_t(z)−e_(y_t),
    ∇z² C_t = Cov_(p_t)(e_v).

Proof: differentiate logsumexp of the linear feature logits. These are the ordinary exponential-family gradient/covariance identities. Here their parameters and produced carrier logits are supplied by our learned forward model. Dense supervised feedback trains the fast state through this representation; the reference learner's credit projections therefore use more than a small isolated regressor.

An addressed key intervention changes only logit r by β. Let p=p_t(r;z), q the modified probability of r, and μ_except_r the base probability-weighted feature mean over other words. Then

    q = p expβ / [1+p(expβ−1)],
    μ_t^a−μ_t^b = (q−p)(e_r−μ_except_r)
                = [p(expβ−1)/(1+p(expβ−1))](e_r−μ_t^b).

The first form is well-behaved as a geometric interpretation near p≈1; numerical projection uses stable leave-one-out log normalizers, not unstable subtraction of near-unit probabilities. If ||e_v||≤R, the mean shift has norm≤2R|q−p|≤2R. Source normalization, all vocabulary scans and dense feature expectation work stay paid.

## How credit reaches a different future word

For a one-step SGD update z_next=z−ηg_t, the route-induced difference at a common starting state is

    Δz_next = −η(q−p)(e_r−μ_except_r).

For a later target y with loss C_future evaluated under the same future route a, the reference-learning residual has first-order term

    C_future(z^a)−C_future(z^b)
      ≈ −η(q−p)(μ_future−e_y)ᵀ(e_r−μ_except_r).

This is a centered learned-feature kernel. A route acting on r can change learning of a future word y even when y≠r: their relation enters through learned feature inner products and the current probability law. This is the precise sense in which the forward representation supplies an inductive bias to backward route credit. Meaningful semantic structure is an empirical property to measure; arbitrary embeddings do not acquire semantics from this equation alone.

With a fixed positive-definite preconditioner P,

    Δz_next = −η(q−p)P(e_r−μ_except_r),
    K_P(v,r)=e_vᵀPe_r.

Optimizer geometry is now part of credit geometry. A learned, causal P_t can alter which represented directions receive credit as observations and regimes change; no stationary data-bias assumption is required. Time transport of the native key perturbation still supplies β_t through learned per-head decay and query activations.

For a smooth future loss, the scalar first-order remainder is bounded by (L/2)||Δz||² when its Hessian norm is≤L along the segment. Since Cov(e) has spectral norm≤R², L=R² is sufficient for this fixed-feature head. For several updates, apply the complete learner-state propagation bound in REFERENCE_LEARNER_RESIDUAL_BOUND.md; one step is not the entire temporal credit law.

## Coordinate symmetry and a stable restricted update

Hold the carrier logits fixed and change the fast-feature coordinates by e'_v=Qᵀe_v, z'=Q⁻¹z. Predictions are unchanged. Set P'=Q⁻¹P Q⁻ᵀ; then K_(P')(e'_v,e'_r)=K_P(e_v,e_r), and the preconditioned update transforms covariantly. Euclidean SGD preserves only orthogonal coordinate changes unless its metric is transformed too. This is a fast-readout coordinate statement, not a claim that arbitrary coordinate changes preserve every layer-normalized carrier.

For fixed P within an issued learning interval, the update Jacobian I−ηP Cov(e) is nonexpansive in the P⁻¹ norm when η λ_max(P)R²≤1. Proof: conjugate by P⁻¹/², obtaining I−ηP¹/²Cov(e)P¹/² with eigenvalues in[0,1]. There need not be a strict contraction: feature/softmax gauge directions can havezero curvature. Changing P each event changes the norm and requires additional inter-metric comparison factors; arbitrary changing-state Adam is not covered by this bound.

## Let future prediction train the credit/optimizer function

At an issued state snapshot, keep z_t and its recorded gradient g_t fixed. If

    z_next = z_t − ηP g_t,

then the exact one-step derivative of the next revealed loss is

    ∂P C_future(z_next) = −η λ_next g_tᵀ,
    λ_next = μ_future(z_next)−e_(y_next).

A parametrized positive-definite P_ψ can use this derivative to learn from future predictive performance, not to reproduce a past gold gradient. A constrained factorization such as P=B Bᵀ+εI needs its own norm/work controls. A high-capacity connected context/credit function can produce B from temporal/lexical state; the matrix alone is not the capacity of the entire credit construction.

The credit-side activations that must survive until future feedback include the issued z_t, g_t, P/ψ version and input activations, forward-feature version, route/reference identity and declared elapsed-time/reset boundary. Later forward prediction supplies λ_next and trains the parametrized credit/optimizer function for subsequent forecasts. This is an explicit delayed learning loop, not inversion or physical time symmetry.

Keep ψ fixed while computing a block's reference projection and raw model-plus-residual actor update. Factual A-dependent activations may train ψ after that block: next-block history includes past A. Updating a supposedly common current-block predictor from factual A state without changing the estimator violates its independence assumption. Reference-only feedback can support another causal construction, but its shared learner policy must be specified and charged.

The one-step derivative holds with its issued snapshots fixed; it is not a total meta-gradient through earlier z/moment updates or a stale derivative evaluated at a new ψ. Versioned packets and explicit horizon truncation retain that distinction. Sparse/asynchronous delivery can delay the update, but does not erase transport, storage or staleness costs.

## Current evidence and next test

Full-vocabulary token projections and independent native readout gradients/Adam states agree to1.2e−14 at the actual50257-vocabulary checkpoint; small native depth2/4/8/EOS/extreme-normalizer contracts also pass. Proper256-context TRAIN calibration is the next quantitative gate. Its current reference learner is a restricted fast head with carrier weights fixed; neither semantic credit generalization nor learned optimizer geometry is a benchmark claim yet. A next P-learning contract must verify the future-loss derivative, coordinate transformation and retained causal/version/reset rules before fitting.
