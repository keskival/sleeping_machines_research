# Sparse evaluation of a forward-derived credit model

B1/R1,11 October2026. Follow-on to the fixed-policy variance diagnostic and [credit audit work frontier](CREDIT_AUDIT_WORK_FRONTIER.md). The current implementation evaluates54/72 route projections. A small residual does not make that all-route model term free on sparse asynchronous hardware.

## What must be estimated

Condition on an issued actor policy π and arrived exogenous U, with score s_a=∇θ logπ(a). Let D_a be the true learner-aware contrast to a fixed reference, and h_a a legal route-independent predictor evaluated at candidate a. Exact full-model correction uses

    G = g_h + (D_A-h_A)s_A,
    g_h = Σ_a π_a h_a s_a,       A~π.

All quantities h are detached with respect to the current actor update. Their dependence on U and the reference learner is allowed; dependence on selected A inside the shared predictor is not silently allowed. No dense model term is required for unbiasedness, but a substitute must preserve its expectation.

Draw J candidates C_j independently of A conditional on H,U from a recorded proposalρ withρ_a>0 whereverπ_a h_a s_a is nonzero. Define

    Z_C = (π_C/ρ_C) h_C s_C,
    G_J = (1/J) Σ_j Z_(C_j) + (D_A-h_A)s_A.

Then E[G_J|H,U]=g_D. Candidate values, inclusion weights and actor score paths must be available without an uncharged full-pool scan. Hierarchical proposal/race sampling may help only after their actual discovery/normalization costs are measured. Predictable adaptive proposal state and stale issued versions follow the existing causal contracts.

## The price of sparse model-term evaluation

Under the stated conditional independence,

    Cov(G_J|H,U) = Cov(Z_C|H,U)/J + Cov((D_A-h_A)s_A|H,U).

Forρ=π and perfect h=D, the residual iszero but the first covariance is Cov(D_A s_A)/J. At J=1 this equals the conditional covariance of ordinary one-sample paired policy credit. Thus a highly accurate critic evaluated at one unrelated sampled route does not retain the exact full-model term's zero route-sampling noise. Coupling C=A makes the two terms cancel back to D_A s_A; it is correct but removes this control-variate benefit entirely. Other couplings require their covariance to be included rather than assuming independence.

This is a variance/work tradeoff, not an impossibility result. Projected reference routes can cost much less than independently adapting actual worlds. Increasing J may buy a useful reduction even when complete route enumeration is unaffordable. Analytic contractions, structured grouping and amortized predictions of g_h are alternative targets; each needs a correction for its actual model-term error, not a claim that an approximate sum is exact.

For fixed h,s, the proposal minimizing the uncentered second moment of Z hasρ_a proportional toπ_a|h_a| ||s_a||, by Cauchy–Schwarz. The covariance subtracts the same ||g_h||² for every proposal. Obtaining that optimal proposal by scoring every route defeats sparse discovery; it is an oracle allocation formula, not an execution method. Baseline centering changes the sampling variances of both model and residual terms and must be analyzed jointly.

## A measurable working budget

Suppose independent episode gradients have trace variance

    V(J) = V_base + V_model/J,
    V_base = future-gradient variation + mean residual covariance trace.

Assume measured per-episode work W(J)=C0+J*c, with C0 including factual/reference learning, policy and residual-query work. For a fixed long-run work budget, the variance of a sample mean scales with W(J)V(J). If c,V_base>0, its continuous stationary point is

    J* = sqrt(C0*V_model/(c*V_base)).

Clamp to the admitted candidate budget and choose neighboring integers. This iid-average calculation is not a theorem about Adam, correlated online sequences, nonstationary future distributions or asynchronous gradient delays. For those settings account for covariance over time, drift and issued-version errors. Exact complete enumeration haszero candidate-sampling variance and is a separate finite-population endpoint, not simply iid J=pool-size sampling with replacement.

The completed fixed-policy reference diagnostic has future variation making98.79% of the remaining total variance. That can make aggressive elimination of the tiny residual less useful than choosing an economical model-query budget. It does not give J* without measuring candidate-query/discovery costs and V_model. The current all-route fit has a positive exploratory stochastic-utility interval but failed greedy gate; its prospective confirmation keeps full enumeration unchanged.

## Next implementation gate

After fixed confirmation, compare exact model contraction with issued independent candidate budgets1/2/4/8 on the same TRAIN packets and fixed actor, using recorded full-support proposals. Contract the mean with exact policy autograd and charge discovery, projection, score transport, actual worlds, optimizer/state and wall. Choose a budget on TRAIN-only risk/work before task-sized fitting. Preserve hard selected writes, temporal state, causal reference learning and factual correction. No current sparse-query, hardware or depth-scaling win is claimed.
