# Learn the outcome law; project it into credit

Curie, 12 October2026. B1/R1 shared enabler, motivated by completed B5/R1 native shared-memory development. Construction implemented in the registered WORLD_CREDIT_PROTOCOL.md acquisition/reuse pipeline; measured decisions live in its completed results. Extends [the shared-memory reciprocal machine](SHARED_MEMORY_RECIPROCAL_EVENT_MACHINE.md).

## What the completed integration teaches us

Independent native F/C memory access and forward task teaching of critic operations are implemented. Causal visibility, relabel invariance, real native read/write gradients and preserved outcome information pass. Shared-memory variants do not beat the raw native-credit incumbent0.613521/159.354s. Most recent outcome-preserving reciprocal0.617163 versus untaught0.615792. These are private single-actor selection results, not a family verdict.

A loss-predicting critic must track both the data process and the changing actor: its target BCE(z_theta,Y) changes when theta changes even at the same observed environment state. Current credit tokens omit the exact issued actor logit. Predicting each route's absolute loss separately can therefore mix environment uncertainty, actor-version drift and inconsistent route projections. Centering improves regression error but failed task quality. We should question that target, rather than assume a small loss regressor is the definition of credit.

## A common world belief gives coherent route credit

For exogenous binary forecasting, fix the available history H before the target arrives. Let

    q(H)=P(Y=1 | H),
    p_theta(a)=sigmoid(z_theta(a)),
    ell(z,Y)=softplus(z)-Y*z.

The conditional expected forecast loss is exactly

    R(H,a)=softplus(z_theta(a))-q(H)*z_theta(a).

One learned q supplies the risk projection of every prospective internal route, including unrealized routes whose logits can be supplied or estimated with their cost charged. It is not necessary to fit unrelated c(a) functions. For two routes j,w:

    Delta_R=softplus(z_j)-softplus(z_w)-q*(z_j-z_w),
    partial R / partial z = sigmoid(z)-q.

The critic can be a large connected native temporal/memory model even though this sufficient output is one probability. A small output does not imply small representational capacity. It must acquire whatever causal structure is needed to predict the outcome law; known proper-loss algebra then projects that law into the actor's current coordinates.

This is an exact conditional identity at fixed history and supplied logits. It does not differentiate the probability of reaching H, future optimizer history or action-dependent environmental transitions. If a acts on the environment, replace q(H) with an intervention-conditioned outcome law q(H,do(a)); a common action-independent probability would be wrong. Richer future learning value also requires continuation/learner state.

The preceding shared-memory architecture supplies the native implementation route: an independent world/credit query over common records, a persistent target-addressed belief state, subsequent revealed outcomes, and a shared representation that the forward actor may read. The world model's observation policy need not replay the actor's source selection.

## Calibration has a task-shaped error law

If q_hat approximates q, the route-risk error is

    Delta_R(q_hat)-Delta_R(q)=-(q_hat-q)*(z_j-z_w).

Thus |q_hat-q|<=epsilon gives a contrast error bound epsilon*|z_j-z_w|. The scalar-logit credit error is q-q_hat. Projecting it into parameters multiplies by the actual producer Jacobian; that work and conditioning cannot be omitted.

All route projections share the same inferred uncertainty. Their pair differences are automatically antisymmetric and cycle-consistent: Delta(a,b)+Delta(b,c)=Delta(a,c). This removes independent-head inconsistency for the same declared history/world belief. It does not prove that q_hat is correct, or that histories with different acquired information should use the same posterior.

For the actual observed-label contrast, conditional variance is

    Var(Delta(Y) | H)=q*(1-q)*(z_j-z_w)^2.

The correct posterior mean removes this label noise from the conditional risk target. This is a reason that learning a rich credit model can help; its value is not exact imitation of each noisy realized gradient. A forecast of changing q must condition on causal state/time/history; no stationary-bias premise is imposed.

## Audits preserve a target; they do not create a free variance gain

Suppose independent audit A has probability b>0 and d_hat=Delta_R(q_hat). The residual estimator d_hat+A/b*(Delta(Y)-d_hat) is conditionally unbiased for the actual contrast averaged over Y, but its variance is

    q*(1-q)*Delta_z^2 / b
    + (1/b-1)*(q_hat-q)^2*Delta_z^2.

Proof: write r=Delta(Y)-d_hat; Var(A*r/b)=E[r^2]/b-(E[r])^2, then separate Var(r) and its squared mean. Even perfect q_hat leaves label noise divided by b. Consequently an audit fraction below1 cannot be presented as reducing this component relative to fully observed exact contrast. It may save expensive intervention work, improve over another estimator or reduce total work at matched learning quality; those are measured claims.

If both logits are already computed, this binary contrast is cheap once Y arrives. Do not audit it to save work that has already been performed. On larger routes, acquiring a counterfactual logit or its derivative can be expensive; expose that boundary. A predicted logit adds another estimation error and requires its own calibration.

## Future-mean teaching is a different deliberate estimator

With dense labels and supplied logits, a predictable mixture

    d_alpha=(1-alpha)*Delta(Y)+alpha*Delta_R(q_hat)

has conditional bias -alpha*(q_hat-q)*Delta_z and variance

    (1-alpha)^2*q*(1-q)*Delta_z^2.

Its conditional squared error relative to the true conditional contrast is

    [(1-alpha)^2*q*(1-q)+alpha^2*(q_hat-q)^2]*Delta_z^2.

If the true calibration error were known, the minimizing alpha is
q*(1-q)/(q*(1-q)+(q_hat-q)^2), with degenerate zero cases handled separately. This is an oracle formula for analysis, not an admitted deployed coefficient. It shows the real bargain: useful learned conditional structure can trade bias against noisy per-example teaching. Causal calibration/future-task endpoints must choose and test that bargain; a claim of exact per-example unbiasedness would be false.

Unlike the anchored residual law, this rule deliberately changes the conditional teaching mean when the posterior is imperfect. That is allowed by the future-generalization objective, but must be explicit. Current native posterior/utility fits cannot establish its advantage retrospectively.

## Generalization beyond binary labels

For an exponential-family loss A(z)-z dot T(Y), a conditional sufficient-statistic mean mu(H)=E[T(Y)|H] yields R=A(z)-z dot mu and local credit grad A(z)-mu. A world model can learn that projection from dense outcomes. This does not generally replace the full world distribution: only the declared loss/producer may admit such a sufficient projection.

Categorical hidden-state credit can sometimes use a projected target-embedding mean, while classifier-weight credit may need per-class probabilities. The projection depends on the current producer weights/version. Do not use the low-dimensional hidden-state identity to claim every parameter gradient is available for free. Time/type likelihoods require the appropriate conditional distribution or sufficient functionals.

## Current construction and next decision

The world_credit_sensor.py integration now uses one native target-addressed world query and one delivered-outcome update per forecast, rather than an independent absolute-loss query/update per route. A linear probability head and native memory body learn actual outcomes; route losses use the actor's known current logits and common posterior. Both roles retain independently selected reads/writes over common memory. Actual actor probability, outcome, producer/version and horizon are explicit available side-information.

Potential economy: a common query shares work across alternatives and reduces duplicate world-state updates. Candidate discovery, counterfactual logits/derivatives, native proposals, optimizer and all serving accesses remain paid. Useful posterior learning and sparse activity must survive an integrated fit before larger acquisition or neural-depth tests.

Register the next comparison around future learning quality and complete work, with frozen/reset world-model and memory-serving/credit-output controls. Use chronological prior snapshots: q_hat and alpha must be fixed before the scored target is revealed. A delayed retrospective continuation audit may use future labels only after they actually arrive. Reuse the native family and completed references; no external dense controller, new dense reference, publicTEST or automatic large run follows this note.

The proposal is a rich learned causal world model with economical, coherent credit projections. The probability/risk identities are established mathematics; the native reciprocal construction, its acquired structure and its resource advantage are the engineering claims to demonstrate.

## Which routes should share credit

[Route similarity and credit generalization](ROUTE_SIMILARITY_AND_CREDIT_GENERALIZATION.md) defines similarity through future causal response, derives conditional transfer bounds, and specializes uncertainty geometry to coherent outcome-law projections. This determines which route projections can reuse the same learned world evidence, which need action-conditioned response, and where paid new observations are necessary.

## Coherence includes shared producer noise

[Coherent world credit and shared producer noise](COHERENT_WORLD_CREDIT_AND_SHARED_PRODUCER_NOISE.md) proves why improving a scalar route contrast can worsen the complete update: factual and route channels share outcome noise and may cancel. It derives the joint projection covariance and exact quadratic future-utility terms, distinguishing posterior calibration, shared transport and optimizer effects. This guides diagnosis of the fixed current fit, not a post-result change to its teaching rule.
