# Bounding the part of credit that the reference learner cannot supply

B1/R1,11 October2026. Companion to [forward-derived learner credit](FORWARD_DERIVED_LEARNER_CREDIT.md). This derivation addresses the residual after projecting route alternatives through a causally trained reference learner. It does not assume that an optimizer is globally contractive.

Let the common observed episode be U, and let route a affect addressed output features without changing the common carrier. Before target j, each true branch learner has state z_j^a=(w_j^a,optimizer moments). All branches begin at the same z_0. After scoring target j they update through

    z_(j+1)^a = T_j(z_j^a,a;U_≤j).

A reference b is fixed before sampled A and U. Define C_j(a,w) as the prequential target loss under route a and readout w. The reference-learning contrast is

    h_B(a) = (1/J) Σ_j [C_j(a,w_j^b)-C_j(b,w_j^b)].

The true independently learning contrast D(a,b) satisfies the exact cancellation

    D(a,b)-h_B(a) = (1/J) Σ_j [C_j(a,w_j^a)-C_j(a,w_j^b)].

Thus the remaining discrepancy concerns the effect of a on learning, rather than the entire temporal forward response. At j=0 the discrepancy is zero. A predictor that ignores the evolving learner must additionally approximate the reference's shared adaptation effect.

## A local finite-episode bound

Choose a norm on the complete learner state, with parameter projection P. Along the relevant finite episode, assume

    ||T_j(z,a)-T_j(z',a)|| ≤ α_j ||z-z'||,
    ||T_j(z_j^b,a)-T_j(z_j^b,b)|| ≤ β_j(a,b),
    |C_j(a,Pz)-C_j(a,Pz')| ≤ L_j ||z-z'||.

The second condition measures the route-specific difference between optimizer updates at the same reference state. It includes moments and the optimizer's nonlinear transformations, not merely the difference between two gradients. Then δ_0=0 and

    δ_(j+1) ≤ α_j δ_j + β_j(a,b),
    δ_j ≤ Σ_(k<j) β_k(a,b) Π_(k<l<j) α_l,
    |D(a,b)-h_B(a)| ≤ (1/J) Σ_j L_j Σ_(k<j) β_k(a,b) Π_(k<l<j) α_l.

Proof: insert and subtract T_j(z_j^b,a), apply the two update inequalities, unroll the recurrence, and apply the loss bound. Empty products equal one. These are deterministic episode bounds; conditional expectations or second moments may be taken afterward when integrable. They apply only where the stated local constants hold. Adam need not have α<1; a persistent state difference can grow even if loss fitting looks stable.

If α_j≤α<1, β_j≤β and L_j≤L, the bound is at most Lβ/(1-α), with a tighter finite-horizon geometric sum. If α=1 it grows at most Lβ(J-1)/2; if α>1 it can grow geometrically. This separates the route-induced update difference from learner amplification. Sparse changes in β can remain small even with a large shared carrier, but carrier depth alone guarantees neither small β nor stable α.

## What this implies for economical credit

For an issued route distribution π and score vector s_a=∇θ logπ(a), the full-correction residual contribution has second moment

    Σ_a π(a) E[(D(a,b)-h_B(a))² ||s_a||² | H].

Insert the squared bound above to obtain a sufficient upper bound on that contribution. This is a score-weighted bound, not ordinary contrast MSE, and the total policy-gradient covariance also includes future variation and the deterministic model term. A small measured residual alone does not prove small native-parameter gradient variance or faster optimization.

A useful larger credit model can focus on β and propagation of learner state differences, using the shared forward/reference computation as a base. It need not relearn the whole response from sparse labels. Its recurrent state should carry parameter and optimizer influence, issued versions, elapsed time and counterfactual identities. A learned residual stays detached from the current route-policy graph, or the estimator must explicitly include its parameter derivatives. Positive-support actual loss corrections remain valid when this learned residual is inaccurate.

Current integration uses p=1 because factual and reference losses are already available. To save an actual reference world, a future design must reuse or replace its required learner trajectory while retaining calibrated uncertainty and valid residual support; merely reducing an audit probability cannot remove a world the predictor itself executes. Any persistence must obey causal version/state boundaries and count storage, transport and updating costs.

## Next measurement

The admitted three-arm route-controller experiment determines whether lower reference-projection residuals produce higher fresh DEV utility under the same update recipe and complete wall boundary. Its fixed carrier does not measure α or β for all-carrier learning. Before that extension, measure route-induced update differences and amplification under changing carrier weights; use the result to choose a supported residual learner and a truncation horizon rather than assuming stability from architecture size.
