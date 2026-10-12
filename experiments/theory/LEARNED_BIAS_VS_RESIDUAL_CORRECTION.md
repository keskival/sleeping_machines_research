# Learning an inductive bias is different from calibrating backprop credit

An inductive bias is a preference among observationally compatible explanations/generalizations. In an event substrate it can be embodied by shared local operators, learned routing/causal factorization, temporal kernels and restrictions on where an error should transfer. Generic permutation-equivariant/addressed primitives give an allowed structure class; the data must still teach the useful conditional graph and couplings. Routes are a central expression of the bias, not its entire definition.

There is a concrete constraint in the current B10 residual rule. With an independent audit A~Bernoulli(p), corrected boundary credit is h_psi + A/p(g_exact-h_psi). Its conditional mean is g_exact for every psi. The derivative with respect to h is1-A/p, whose conditional mean is zero. Under a linear optimizer and a first-order future-loss approximation, the expected critic meta-gradient therefore cancels. Nonlinear optimizer/curvature and variance effects can still produce useful learning; cancellation does not prove the scheme cannot win. But the estimator cannot freely learn a different expected credit direction while retaining that mean constraint.

A future-useful learned inductive bias needs a separate intervention channel. One candidate is

    credit = c_psi + A/p(g_exact-c_psi) + b_psi,

where c calibrates a factual estimator and b learns a prospective credit/update bias from later proper loss. The derivatives are (1-A/p) dc/dpsi + db/dpsi; the second survives conditional expectation. b is not an unbiased backprop estimator and must be evaluated as a learned intervention, with audit/utility controls and full costs. This is a proposed construction, not a demonstrated improvement.

For hard routes, the prospective channel must also be able to reach alternative producers or routing logits. A factual Jacobian zero on an inactive route cannot supply that route's derivative. Paid forced-route/model replays or positive-support sampled interventions supply such information. Generalizing their utility across unseen graph compositions is the structural-induction target. Hidden generator parent labels are not legitimate training supervision.

The current longer B10 run measures whether more experience makes the corrected scheme useful despite its mean constraint. The causal-circuit development protocol separately tests acquired locality, composition, time and route generalization. Neither is a reason to claim that Transformers cannot represent these relationships. Practical superiority means that acquired structure yields better future quality per data/complete work under the same observations.
