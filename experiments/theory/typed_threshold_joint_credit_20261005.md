# Learning typed comparisons through winner and computation time

Extends the joint winner/time identity in theory102 to the proposed typed interface. Current interfaces and fits are untouched. Failure addressed: fixed threshold comparisons have no learned threshold credit; simply adding a straight-through derivative would not derive the actual hard-race objective.

For one numeric field use normalized margin m=(x−τ)/s, s>0, rates r_true=exp(m), r_false=exp(−m), and total Λ=r_true+r_false. First winner W and time T have joint density r_W exp(−ΛT). With sign(true)=+1 and sign(false)=−1,

    ∂m log p(W,T) = sign(W) − T(r_true−r_false).
    ∂τ log p(W,T) = −[sign(W) − T(r_true−r_false)]/s.

This is a probabilistic threshold during training. A deterministic zero-noise comparison is a different inference policy and needs its own contract/comparison; it is not silently identical to this race. Category membership is semantic, not arithmetic on category IDs; candidate subset selection requires its separate discrete credit.

For a suffix loss ℓ(W,T), the score contribution is E[ℓ ∂m log p]. Add any explicit factual path derivative when loss depends directly on m as well. Do not double-count a reparameterized time derivative and the same time likelihood term: choose and derive one estimator boundary. Deep neural/state processing stays downstream of the typed comparison, with small messages, separate keys/values and sparse addressed state intact.

Winner and first time are conditionally independent, π_true=r_true/Λ and T~Exp(Λ). Conditional branch-counterfactual evaluations can supply losses of both actual alternative suffixes at common T and matched future noise. The choice term uses those differences; the time term remains required because computation through time can alter memory and suffix predictions. Category-only credit discards that coordinate. Proposal sampling can reduce alternative evaluation work, but must have support and the appropriate inverse probability weighting. Discovery, losing suffix evaluations, optimizer and clock work are charged.

A closed witness ℓ=A_W+B T gives

    Eℓ = π_true A_true + π_false A_false + B/Λ
    ∂m Eℓ = 2π_trueπ_false(A_true−A_false) − B(r_true−r_false)/Λ².

When both A values are equal, categorical credit is zero but timing credit is generally nonzero. Twenty stdlib moment/finite-difference cases verify the joint formula, choice/time decomposition and this witness. They establish this mathematical primitive, not learning in the integrated model.

Positive affine unit changes x'=a x+b, τ'=aτ+b, s'=as preserve m and race distributions; ∂τ'=(∂τ)/a. Parameterize learned thresholds in normalized coordinates to avoid unit-dependent raw optimizer steps. Scale/origin and candidate quantiles come from FIT only.

Next integrated contract: replace one selected predicate outcome, replay the actual row suffix with matched future noise, verify credit against an enumerated small bank, preserve continuous computational time and factual state gradients, then run one bounded mixed-type fit. This addition is outside the pinned language/FAS source closures and must not preempt their admitted jobs.
