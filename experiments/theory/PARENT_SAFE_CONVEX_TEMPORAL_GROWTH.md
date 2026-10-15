# Parent-safe growth and a convex output-credit subproblem

15 October 2026. R1 next-construction rationale from completed evidence; not a
training admission or a claim that the proposed repair improves held-out language.

## Measured failure being repaired

ProtectedD4 loses6.095076744 versus continuedD2 6.065562542 after the same2M
additional targets. Frozen audit removing its correction gives6.072677415.
Thus its excess0.029514201 decomposes into harmful correction0.022399329 and
own-parent-path excess0.007114873. Substituting old parent features into the
learned correction gives6.200268526: different new features matter to this fitted
map, but this does not make the correction useful against zero. All figures use
8192 reused DEVtargets/1024history/seed6; no retrained ablation/causal fit attribution.
Source: results/diagnostics/aws_protected_depth_contribution_audit_20261015_v1.json.

## 1. Preserve learning, not just the initial function

Let theta be acquired parent parameters and psi new branch parameters. Parent
prediction p_theta and corrected prediction p_(theta,psi) share information but
need not share an optimization target. Define parent updates from its original
loss/alternative-credit rule only, and branch updates from corrected prediction
loss plus its supported alternative credit, restricted to new coordinates:

    g_old = declared_parent_credit(loss_parent; theta),
    g_new = declared_new_credit(loss_corrected; psi).

This is a block multi-objective learner, not the gradient of a single asserted
joint potential. It preserves a fallback learner while the branch discovers
useful corrections. Under exact conditions below, parent-trajectory equality
follows by induction over issued updates:

- same parent data, numerical state, feedback, availability and random streams;
- same old parameter credit, bank mapping/caps, optimizer moments/versions;
- old bank updates do not depend on new bank gradient norms or mutable aliases;
- branch reads/counterfactual replays never mutate parent-owned state/noise;
- shared packed parameter arrays merge disjoint old/new coordinate gradients
  with original old moment/step semantics; initialization alone is insufficient.

If these conditions hold at update k, parent forward/credits and optimizer
outputs match, so they hold at k+1. Global clipping over old+new gradients breaks
the induction; separate old/new local caps must actually preserve the original
old-bank law. Recalibrating old caps separately in the grown model also breaks
this exact claim. Teacher allocation/extra label exposure can break it too.
Proof is conditional on identical parent observations: in closed-loop action
systems changed actions change later data, so use matched-data internal replay
for this comparison and test actual control separately.

The final corrected prediction can still be worse. Parent preservation protects
available competence; a causal optional branch/selection policy must earn its
usefulness and charge discovery, audit and delivery work. Do not use DEV or
unrevealed targets to choose a per-event fallback. A serial eager implementation
is not a deadline-preserving asynchronous scheduler.

## 2. Fixed native features admit exact convex correction learning

Fix parent features z_i, new temporal features phi_i and the normalized readout.
The corrected feature is z_i+W phi_i. Our TokenReadout is either dense affine
softmax, or a hierarchical head plus selected tail, each affine in its input
(two consecutive tail linear maps are still affine). Hence target loss is a
sum of terms logsumexp(A(z_i+W phi_i)+b) minus a target affine logit.
Each term is convex in W; sums/averages preserve convexity. Its feature Hessian
is A^T(diag(p)-p p^T)A, positive semidefinite; hierarchical losses sum the head
and selected-tail contributions. This is exact normalized token NLL, not sampled
negative training or an approximation to full-vocabulary probabilities.

For lambda>0,

    J(W) = mean_i NLL_i(W) + lambda/2 ||W||_F^2

is strongly convex and coercive in finite W, with unique minimizer when the
features/readout are finite. Since W=0 is feasible, J(W*)<=J(0). An approximate
accepted solution with J(W_hat)<=J(0)+epsilon satisfies

    empirical_NLL(W_hat)<=empirical_NLL(0)+epsilon-lambda/2||W_hat||^2.

This certifies empirical fit on the declared TRAIN panel only. There is no
held-out guarantee, and no certificate for simultaneously adapting the readout,
parent, branch features, selection policy or random realization. Regularization,
panel coverage and temporal dependence affect generalization; select only on
TRAIN-side development and disclose all selection work.

The first useful discriminator is whether a fixed native branch's features
support a better correction at all when this output subproblem is solved, rather
than assuming deeper feature credit is the only failure. A useful frozen-feature
fit does not prove that end-to-end learning can acquire those features cheaply.
An unhelpful fit can mean no useful signal accessible through this interface,
insufficient panel support or regularization; not a family impossibility.

## 3. Implemented helper and boundaries

sleeping_machines/convex_feature_correction.py fits W with strong-Wolfe LBFGS
on an explicitly supplied frozen TRAIN panel, fixed lambda and iteration limit.
It keeps the best objective including feasible zero, returns tensors/certificate,
and deep-copies/freezes the readout. It does not mutate actor parameters, consume
RNG, fit hidden temporal state, select lambda, access data or admit a job. It
rejects readouts overriding the declared affine normalized loss or introducing
nonlinear tail layers. Tensor features must be finite, compatible CPU inputs.

Two numerical contracts cover dense and adaptive normalized losses, improvement
on a constructed supported feature panel, feasible-zero objective, zero-feature
fallback, no readout/RNG/gradient mutation and invalid regularization. These are
solver contracts, not corpus results. Every closure/head evaluation, feature
acquisition and inherited learning cost must be charged in a real comparison.
The implementation's best-evaluated objective certificate does not assert that
LBFGS found the unique global optimizer within its finite iteration budget.

## 4. Next decision, without multiplying fits

Use the completed harmful-correction audit before another long fit. One short
AWS private-memory intervention is registered in PROTECTED_DEPTH_ATTRIBUTION_20261015.md;
it adds the history-content contrast absent from the completed output audit.
Then select one repair: output-credit/conditioning if native features support
useful TRAIN-fitted corrections; parent-safe block learning if acquired learning
is being degraded; temporal feature/credit repair if no useful output response
is accessible. Test the complete intended mechanism on TRAIN with matched parent
continuation and retained state before a long acquisition. Do not simultaneously
add lane-noise, horizon, width and output changes. The independent-lane backend
is separately available; its unit contracts are not an admission to combine it.

Retain time-based computation, hard races, addressed private state, separate
keys/values, supported counterfactual credit and useful online adaptation. A
convex readout subproblem supports the native hierarchy; it does not replace
that hierarchy with a dense model. Reuse completed parents and feature panels;
no automatic transformer training or byte-level substitute for GPT2 language.
