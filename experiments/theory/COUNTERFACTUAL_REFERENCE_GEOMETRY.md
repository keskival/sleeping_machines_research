# Forward futures can teach which counterfactual is economical

B1/R1,11 October2026. Extends [the credit audit frontier](CREDIT_AUDIT_WORK_FRONTIER.md) and [reciprocal teaching](RECIPROCAL_ACTOR_CRITIC.md). This derives a reference-selection objective; it is not implemented in the active128/512 pilot, which retains the saved greedy reference. Known baseline/control-variate principles are the starting point, not a novelty claim.

## Mean task value and credit noise select different references

Let L_a(U) be the learner-aware loss of route a under common future randomness U, with a separate causal branch learner per world. A reference b(H) is issued before the future and independent of the sampled action A conditional on H. Let f_a be issued critic level predictions and R_a(U)=L_a(U)-f_a. Use h_b(a)=f_a-f_b and the paired residual R_a-R_b in the supported raw policy-gradient estimator. Keep pi, score s(a) and action-dependent audit propensity p(a)>0 fixed during reference comparison.

The deterministic model gradient is independent of b because sum pi(a)s(a)=0. Its residual mean correction is also independent of b. Hence the only reference-dependent term in trace variance is

    J_ref(b)=sum_a w_a E[(R_a-R_b)^2|H],
    w_a=pi(a)||s(a)||^2/p(a).

Choose b to minimize residual fluctuation distance, rather than assuming the lowest-loss route gives the cheapest credit. With perfect f_a=E[L_a|H], write the route-loss covariance C. Then

    J_ref(b)=sum_a w_a C_aa + W C_bb - 2sum_a w_a C_ab,
    W=sum_a w_a.

Thus predicting conditional means alone misses information useful for selecting counterfactual teaching routes. A coherent world model can also learn covariance or shared latent future responses. The covariance must describe interventional worlds under the declared common continuation/learning law; covariance between unrelated observed episodes is a different object.

Define R_bar=sum w_a R_a/W. The exact identity

    sum w_a E[(R_a-R_b)^2]
      = sum w_a E[(R_a-R_bar)^2] + W E[(R_b-R_bar)^2]

makes reference selection a weighted medoid problem in L2 future-residual space. The best executable reference is the available world closest to R_bar. An unconstrained R_bar is an oracle construction requiring all residuals; it is not free to observe. With imperfect f the same identity includes mean error as well as stochastic fluctuation. High mean loss and low residual distance can coexist; a reference is an audit instrument, not necessarily the deployed route.

## Cost, learning and causal scope

A reference world can be expensive or coincide with the factual route. Optimize the declared work/variance objective, e.g. J_ref(b)+lambda sum pi(a)p(a)c(a,b), where c is actual incremental audit work. Joint reference/propensity optimization is coupled. The simple medoid formula assumes fixed propensities and equal reference costs. Candidate scoring, covariance queries, discarded worlds, reference learner state and delayed receipts all remain charged.

For conditional unbiasedness, issue b before observing the unrevealed future, use the same reference in predicted and observed contrasts, retain actual p, and preserve the packet's actor/critic/learner versions. Selecting the empirically best b after seeing U requires correction or an explicitly different experiment. Choosing b as an arbitrary function of sampled A generally breaks the simple constant-baseline cancellation; the present derivation does not authorize it.

Changing references does not invalidate fixed-law evidence packets when they retain both route descriptors and their recorded reference: each label still estimates its own L_a-L_b. A connected level model must fit those pairwise differences jointly. If carrier, optimizer continuation or horizon changes, the target-version problem remains.

This provides a concrete reciprocal mechanism: forward futures teach which alternatives share unpredictable fluctuations, the credit model learns that geometry, and the policy/scheduler chooses a teaching reference that reduces later audit noise. It can complement learned mean credit when future innovation dominates. The next admission criterion is completed evidence distinguishing mean-calibration error from paired innovation, followed by a source-pinned TRAIN-only reference-selection diagnostic. No new benchmark/public TEST or assumption of stationary inductive biases is required.
