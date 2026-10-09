# AWS — learning a backward update function across future examples

9 October 2026. B1/R1 shared learning enabler; AWS owner. This is a design and an admitted bounded development
pipeline, not a fitted performance claim. It complements notes 160–161 and optionality notes 11/155.

## Chosen path and failure addressed

A posterior-credit learner estimates which cause explains the outcome. That does not establish the best finite
parameter update, preserve optionality beyond its horizon, or teach an unused route's content. Exact differentiation
also does not choose its own learning objective. The next experiment isolates **learning the update function** from
learning attribution: retain exact inner gradients, learn a shared persistent rule that turns them into changes,
and train that rule on independent future predictive performance. Backprop supplies exact local derivatives; the
outer objective judges the learning algorithm. This is a learned optimizer construction, not a proof that gradients
are wrong or that a full trainable backward system has been built.

For forward parameters theta, backward parameters psi and local optimizer state h:

    (Delta theta_t, h_(t+1)) = U_psi(theta_t, g_t, h_t, block, step),
    theta_(t+1) = theta_t + Delta theta_t,
    J(psi) = E_task[L_independent_future(theta_H)].

Parameters psi generalize across independent task episodes. Optimizer state h carries history within each learner's
adaptation trajectory, and resets between independent tasks. The full meta derivative includes changed inner
parameter gradients and optimizer-state dynamics. Meta-training uses higher-order differentiation and charges it;
we do not claim this eliminates backprop or second-order training work. The initial forward parameters are fixed
and identical across arms. This prevents a better learned initialization from masquerading as a better update rule.

## Analytic witnesses, completed without fitting

`experiments/credit/reciprocal_update_math_v1.py` and
`results/credit/aws_reciprocal_update_math_20261009T2137Z.json`:

- Noisy support target, task variance 1 and independent observation noise variance 4: one step from zero minimizes
  independent future squared-error risk at gain **0.2**, versus **1.0** for same-example fitting. Future risks are
  **2.4 vs 4.0**. A gradient-imitation objective cannot express this preference by itself.
- Two quadratics with curvature 1 and 9: a curvature-conditioned reciprocal gain reaches their optima in one step.
  The best shared scalar for mean curvature-weighted loss is **0.112329**, leaving **0.197260** loss. This is an
  amortization opportunity if task structure reveals curvature; it is not evidence that our network learns it.
- Recruitment costing 0.1 now but improving two future uses by 0.2 each has immediate value -0.1 and continuation
  value +0.3. Complementary actions have information option value 1 when future evidence is observed, and no
  contingent-choice advantage when that evidence is unavailable.

Correction: the 2135Z evidence packet selected the scalar minimizing unweighted squared displacement and evaluated
curvature-weighted loss. Its scalar is not the optimum for that loss. The 2137Z packet uses the consistent objective;
the earlier result data stay preserved, source version in git. None of these witnesses are fitted model results.

## Native forward member and retained mechanisms

`experiments/credit/reciprocal_update_v1.py` implements four observed addressed key/value writes followed by queries
with replacement. Each slot carries complex temporal value modes, with elapsed-time decay and rotation. Separate
normalized keys/queries define a stochastic hard read route. Conditional on that route, one value drives two
exponential output clocks; the observed mark and wait are scored with the hazard and exact silence compensator.
The training likelihood marginalizes four hidden read causes and credits losers exactly. A selected-value forward
path has a numerical parity contract. Actual likelihood learning/evaluation scores every cause; it is not charged
as winner-only execution. Input key identities legitimately determine writes; target outcomes never enter causal
keys, values or pre-outcome rates.

This member has one persistent temporal memory, not deep persistent layers. Write selection is observed-addressed,
not a learned hidden write choice. It has no credit inference network, learned message head or recruitment policy.
Those are integration gaps, not mechanisms established by the new pilot. No existing benchmark model or R1 keyed
read is replaced. Task-specific value-to-output mappings vary across episodes; train, selection and assessment
streams use distinct seed ranges. The model and generated training data are independently seeded and hashed.

The backward module is an 8-hidden-unit shared coordinate MLP with block identity, bounded parameter features,
current gradients and first/second moment memory. It outputs a gain in [0.02,0.5] on an RMS-normalized gradient.
The positive gain constrains the local direction; this first construction cannot take arbitrary non-descent steps
or discover discrete recruitment. It tests useful finite updates before adding those freedoms.

## Frozen bounded development protocol

Queue prefix `aws_reciprocal_update1_20261009T2140Z`, AWS slot 1 after benchmark work and prior admitted enablers:

1. Numerical contracts: full parameter posterior-gradient identity; full outer derivative vs finite differences;
   target independence; transport semigroup and temporal sensitivity; selected-value path parity; future evidence
   changes the meta-training signal.
2. Smoke: 3 outer steps, 3 inner steps, batch 2, 2 development tasks.
3. Three seed-0 pilots, each 80 outer steps, 4 inner steps, batch 8: **future** (independent continuation),
   **immediate** (same support examples), **scalar** (one learned global gain with the same moment dynamics).
4. Automatic matched-model/data analysis. Final meta step fixed in advance, no assessment-based epoch selection.

Assessment: 16 unseen synthetic DEV tasks, distinct from 16 step-size selection tasks. Baselines are the same native
model adapted by SGD or Adam; learning rates {0.01,0.03,0.1,0.3} selected on the selection pool. Each pilot also
measures no update and backward-memory reset. This is a matched diagnostic, not a language or public benchmark test.

Predeclared three-seed confirmation screen: future-trained updates improve assessment NLL by at least 0.02 over the
selected ordinary baseline, and 0.01 over immediate/scalar pilots; at least 75% of paired task gains over each
objective control are positive. Failure yields development/error analysis, not a family loss. Confirmation and
integration are not automatically admitted by this packet. All PyTorch contracts and fits remain pending.

Resource accounting: whole meta-training targets and wall time, per-task adaptation targets/wall time, all-cause
likelihood scoring, backward/forward parameters, RSS and source hashes. Meta cost must be amortized over a stated
number of later adaptations before an efficiency claim. Target counts are not FLOPs; no total-compute superiority
is claimed. One thread, 3 GB RSS/8 GB virtual-memory caps, 8 GiB availability floor, contract/smoke dependencies and
measured smoke RSS margin, 20-minute pilot timeouts through the existing run_safe slot scheduler.

## Benchmark-facing next decision

B1/B2/B4/B5 jobs retain priority. On a passed diagnostic, compare the learned rule on existing R1 recall DEV with
the same forward model/initialization and equal complete resources before a longer language fit. A subsequent
learning-allocation module must optimize causal continuation including future weight changes, overwrite cost and
recruitment, not train on posterior responsibility alone. Paired future data and paid counterfactual outcomes train
it; observed future labels are never inputs to an earlier inference decision.

Known precedents: [Andrychowicz et al., learned optimizers](https://arxiv.org/abs/1606.04474),
[Jaderberg et al., synthetic gradients](https://proceedings.mlr.press/v70/jaderberg17a.html), and
[Metz et al., learned-optimizer resource tradeoffs](https://proceedings.mlr.press/v199/metz22a.html).
This is a substrate-specific engineering path, not a general novelty or convergence claim.
