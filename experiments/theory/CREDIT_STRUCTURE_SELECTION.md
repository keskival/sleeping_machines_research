# Credit structure is a hypothesis: the next integrated selection

12 October 2026, B1/R1. This is a construction and selection decision, not a new experimental result. Curie owns native credit implementation; AWS owns the admitted common-state language confirmations. Preserve both pipelines.

## What the current construction gives us

`causal_induction/race_credit_sensor.py` uses a shared native temporal model with separate numerical state at each public `(target, source)` address. Its inputs include forward feature tokens, elapsed-time/horizon summaries and observational relationship statistics; feedback tokens carry revealed route loss. Actor and credit races have independent random streams. Predictions support a factual-plus-sampled-alternative route contrast with positive-probability residual audits.

`causal_induction/shared_bank_sensor_v2.py` makes versioned credit records available to later forward reads. Actual later task losses can teach earlier credit writes and the independent credit read policy within the retained graph window. Numerical records persist beyond that window. Its fixed relational addresses, 16-visible-record discovery rule, two-layer/two-head/two-receiver critic, residual scale and teaching window are engineering hypotheses, not defining properties of learned credit.

`causal_induction/task_taught_shared_bank.py` retains the executable forward function and parameters but detaches features at the auxiliary loss readout. Auxiliary regression consequently teaches the readout; actual later task losses teach the native memory/read path. This gives a useful implementation of forward task teaching without another predictor hierarchy. It does not yet differentiate the complete future optimizer trajectory.

Completed shared-bank evidence: reciprocal NLL 0.616480 versus untaught 0.616819, but separate 0.615585 and retained raw-credit 0.613521 lead. Reciprocal costs 346.113 seconds versus separate 237.261. The task-taught replacement is pending; these cells must not be filled from expectations. Centered teaching reduced contrast error/magnitude from 91.7 to 1.154 but worsened task NLL to 0.621341. These results select task utility rather than regression fidelity as the adoption objective.

## Identify the object before choosing its representation

Let X contain the issued actor parameters, numerical memory, optimizer state, credit state and causal event history. Fix a continuation protocol and future exogenous randomness xi. An intervention a produces learner trajectory U_a(X,xi) and measured future loss

    Q_a(X,xi) = L_future(U_a(X,xi)).

The relative credit is D_ab = Q_a - Q_b. Every comparison starts from an independent copy of the same X. Optimizer moments and state changes are part of the intervention, not erased after it. Stopping, replay and fresh learning have different costs, so the policy target can be Q_a + lambda C_a with C_a measured and lambda declared before selection.

For a common continuation, D_ab = -D_ba and D_ab + D_bc + D_ca = 0, pointwise in xi. A learned potential q_psi(X,a) enforces those laws through d_ab = q_psi(X,a)-q_psi(X,b). A direct pair predictor must enforce them or explain why its comparisons use different continuations. Predicting potentials is therefore not intrinsically mistaken. The failure to investigate is whether a restricted representation spends its capacity on common outcome variation while missing action effects.

If Z is the issued feature packet, the irreducible squared contrast-prediction error is E[Var(D_ab | Z)]. Capacity cannot recover information absent from Z. Current operator information and learner state matter whenever identical feature packets can yield different intervention consequences. The explicit current-weight witness is in WEIGHT_CONDITIONED_CREDIT_IDENTIFIABILITY.md; it motivates an input repair, not a blanket verdict on temporal memory.

## The strongest next member

Use a current-operator local anchor plus a learned temporal residual:

    d_ab = k_ab(X) + r_psi(Z_X,a,b).

The anchor k_ab is the known local effect for the declared intervention. An exact local adjoint is appropriate for an infinitesimal differentiable change; an exact finite local response is appropriate for a discrete route substitution. Neither is automatically the full future-learning effect. State which anchor is used, retain its issue version, and charge its computation. The residual learns the difference between the anchor and actual future loss contrast, including memory effects and adaptation. It can use Curie's native shared records; shared access is optional and must earn its quality/work cost.

Prefer a difference of residual potentials when comparisons share a continuation. Reuse operator summaries when weights change rather than scanning every parameter for every query. Operator freshness, recurrence through depth, causal feedback and optimizer conditioning are separate axes; changing all of them at once would conceal the reason for any improvement.

## Finite teaching closure and honest gradients

Suppose the actor receives update theta_plus = theta - eta P g_psi and P is fixed for this one update. For a differentiable retained continuation,

    grad_psi L_future(theta_plus)
      = -eta (D_psi g_psi)^T P^T grad_theta_plus L_future.

This directly teaches the estimator through the intervention it caused. With Adam or a multi-step continuation, differentiate the actual update map, including moments and state, or declare the frozen/truncated approximation. Learned hard choices additionally require the appropriate score-function term or a labelled surrogate. Later task losses flowing through shared memory are one genuine path in this derivative, not its entirety.

Closure is actual forward outcomes -> finite retained update/record paths and paid paired audits -> estimator updates -> later interventions. It does not require a second learned trainer to score a third trainer. Local auxiliary targets can maintain useful prediction, but their coefficient must not silently replace the future task objective.

For an audit I with issued probability p>0 and known conditional contrast mean D, the corrected observation is

    d + I/p (D_observed - d).

If the audit observation is conditionally unbiased and the coin is independent of its unrevealed residual given the issued packet, its conditional mean is D. Retain the propensity and issue version. This identity preserves the contrast mean; a nonlinear optimizer can still change expected learning effects, so task adoption must be evaluated directly.

## One decision, not a broad architecture grid

After the owner's current task-taught fit completes, compare the retained native estimator with one current-operator-anchored residual member under the same actor, acquisition data, continuation and audit budget. Keep temporal memory, keys/values, sparse selected writes and causal arrivals. Do not substitute a dense critic. A matched anchor-only arm determines whether learned residual memory adds value; the completed original remains the practical incumbent. Distinguish current-state reuse from transfer across changed actors and dependencies.

Select using prospective future task loss and total work to reach common quality. Charge source acquisition, failed development arms, record discovery/transport, local anchors, all audits, validation and optimizer work. Use more than one acquisition horizon to expose amortization; an expensive source fit is not free after reuse. No standalone variance grid or new external neural control is admitted here.

AWS common-state language evidence supplies an actual future-learning target: at the larger seed-6 point, fresh loss 7.263107 versus replay 7.903868 and stopping 7.734948, from an identical acquired learner. The larger seeds 7/8 are active. These examples presently favor one action; they cannot identify when replay is preferable. Policy acquisition requires changing regimes and competing useful actions. Prepared nonlinear/rewired tasks on Curie's side are useful for that selection only after the existing transfer gate is passed or replaced by a documented diagnosis and corrected integrated test. Do not reopen deferred fits merely because this note exists.
