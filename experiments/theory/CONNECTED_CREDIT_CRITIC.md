# A connected credit critic for future-directed learning

10 October 2026; founder-directed B1/R1 enabler, with causal B10 integration. This formalizes a shared persistent credit model and learning policy. It extends RECIPROCAL_EVENT_LEARNING.md and CREDIT_LEARNING_CERTIFICATES.md. The construction uses familiar conditional value functions, Bellman recursion and adjoints; it is not a priority claim or a completed performance result.

## 1. Learning is an intervention process

Let H_t contain everything legitimately observed by the learning decision time t, including outcomes only when revealed. Define augmented internal state

    X_t = (s_t, c_t, o_t, theta_t, psi_t, omega_t, pending_t, versions_t).

s is inference memory, c shared credit memory, o optimizer memory, theta forward parameters, psi credit parameters, and omega update/scheduling parameters. pending stores bounded versioned records for delayed observations. History H_t is retained in the definition so a compact X_t is not silently assumed sufficient or Markov. The event timestamp is part of H_t; t denotes decision order and does not impose a synchronous clock.

A learning action a_t can modify addressed parameters, state, route/update timing, audit selection or optimizer state. A specified action map U(X_t,a_t) is followed by the actual forward/event transition. A continuation learning policy pi specifies subsequent actions. Those transitions include how changed theta and c change predictions and subsequent learning. The reference policy, state reuse/invalidation rules and parameter versions are fixed for each comparison.

For H future events, define

    Q_t^{pi,H}(H_t,X_t,a)
      = E[ sum_{k=1}^H gamma^{k-1} loss_{t+k}
           + lambda sum_{k=0}^{H-1} gamma^k work_{t+k}
           | H_t,X_t, do(a_t=a), continuation pi ].

The future distribution is the actual conditional distribution at t, not the empirical distribution of all past samples. It can change with history/regime. The discount/horizon, resource price and loss units are declared. Event-count and physical-time horizons differ; for time discount use exp(-beta elapsed_time) consistently in every arm.

Define intervention advantage against a stated baseline a_0:

    A_t^{pi,H}(a) = Q_t^{pi,H}(a) - Q_t^{pi,H}(a_0).

Smaller is better because this critic predicts costs. Past-loss descent and future advantage need not have the same sign. At H=1 with fixed parameters after the action and a parameter step delta, the first-order future target is E[grad_theta loss_{t+1} | H_t]^T delta, not necessarily grad_theta loss_t^T delta. Longer horizons additionally differentiate subsequent state and learning transitions.

The do notation declares a counterfactual target. It does not make that target identifiable from ordinary observational data. Controlled branches, legitimate randomized interventions, or a validated structural simulator provide evidence; action/outcome correlations without sufficient control do not.

## 2. The RL connection is precise

For a sufficient augmented state, or on full histories, the finite-horizon recursion is

    Q_t^{pi,H}(h,x,a)
      = lambda work_t(x,a)
        + E[ loss_{t+1} + gamma V_{t+1}^{pi,H-1}(H_{t+1},X_{t+1}) | h,x,do(a) ],
    V_t^{pi,H}(h,x) = E_{a drawn from pi(.|h,x)} Q_t^{pi,H}(h,x,a),
    V_t^{pi,0} = 0.

Time/regime dependence is explicit. The finite-horizon target is well defined without a stationary transition law. Infinite-horizon stationary fixed-point arguments must not be imported unless their assumptions hold. In an externally generated data stream, actions change the learner and its future predictions; they need not change the external event distribution. In embodiment or trading execution they may change the environment as well, which belongs in the transition model.

Thus the learning policy acts as an actor and the shared future-cost model as its critic. This is a connection to meta-learning/control, not an equivalence between value prediction and gradient estimation. A conventional critic often scores environment actions; this critic also scores interventions in the computing/learning substrate. It must represent the direction and location of those interventions.

For a stochastic learning policy pi_omega, when future psi/other direct omega paths are frozen for this policy-gradient witness,

    grad_omega J = E[ sum_t grad_omega log pi_omega(a_t|H_t,X_t)
                          (return_cost_t - baseline_t) ].

The baseline does not depend on the sampled action. Replacing returns with an approximate critic introduces approximation error unless corrected. Additional direct differentiable omega paths require their own derivatives. Deterministic differentiable interventions use grad_a Q and the derivative of the action map under the corresponding regularity assumptions. Hard routes and race schedules require discrete/counterfactual or score-function treatment; pathwise gradients through a hard winner alone omit alternatives.

## 3. Connected credit and exact adjoints meet at intervention derivatives

For a differentiable intervention family a, define

    b_t(a) = grad_a Q_t^{pi,H}(a).

This is future-directed intervention credit. It is a derivative of expected continuation cost, rather than necessarily the realized derivative of past loss. At a fixed differentiable transition X_next = T(X,a,xi), with differentiable continuation value and conditions allowing differentiation through expectation,

    grad_a Q = grad_a immediate_cost
                 + E[(partial_a T)^T grad_Xnext continuation_cost].

All direct observation/cost paths are included in immediate_cost. The Jacobian transports a cotangent; the continuation critic supplies the boundary cotangent. Exact derivatives of known local operations remain useful inside a learned future evaluator. If the continuation boundary is exact, this recovers exact differentiated continuation. If it predicts a conditional expectation, it supplies conditional expected credit. Approximation, truncation and simulator mismatch are different errors.

Construct one shared critic with persistent addressed state. Its local messages carry forward context, proposed intervention, error/outcome information already observed, elapsed time, uncertainty and version. Its event graph connects layers and temporal records; learned races choose service order and selected recipients. Local modules share parameters and communicate; locality does not require statistical isolation. Available critic capacity, selected messages, scored candidates and full learning work are measured separately.

One possible local parameterization is

    A_psi(h,x,a) = b_psi(h,x)^T a + (1/2) a^T M_psi(h,x) a + R_psi(h,x,a).

b predicts directional future credit, M models interactions, and R handles nonlinear finite interventions. M is symmetric for a scalar quadratic value; low-rank plus local blocks is an economical hypothesis, not a required limit. Large action vectors use sparse addresses and shared functions rather than evaluating a dense coefficient for every parameter. Exact known elapsed-time transport is retained. A scalar critic evaluated only at a=0 cannot identify intervention directions.

If b is obtained by differentiating a shared scalar critic, its smooth action-space field is locally integrable. Separate unconstrained vector predictors need not satisfy that property; whether enforcing it helps is a test. Hard routes retain finite intervention values rather than a fictitious smooth derivative.

## 4. What approximation accuracy guarantees

Finite actions: suppose |A_hat(a)-A(a)| <= epsilon on the admitted action set. The selected minimizer a_hat obeys

    A(a_hat) <= min_a A(a) + 2 epsilon.

Proof: compare true and predicted advantage at a_hat and at the true minimizer. Including no-op ensures the minimizer is at most zero. A stricter acceptance rule A_hat(a_hat) < -epsilon guarantees true improvement relative to baseline under this bound. Without a calibrated uniform or appropriate probabilistic bound this is a criterion to test, not a deployment guarantee. The evaluated horizon and continuation policy must agree with the enacted ones.

Continuous actions: let Q be L-smooth near zero and ||b_hat-grad_a Q(0)|| <= epsilon. For a=-eta b_hat,

    Q(a)-Q(0) <= -eta(1-L eta/2)||b_hat||^2 + eta epsilon ||b_hat||.

Therefore epsilon < (1-L eta/2)||b_hat||, with eta < 2/L, is sufficient for improvement of the conditional expected continuation objective. State/action constraints require a separate feasible-action analysis. This does not guarantee every future sample improves.

Small value error alone does not guarantee small derivative error: epsilon sin(a/epsilon^2) is uniformly bounded by epsilon but has derivative 1/epsilon at zero. Train/test directional derivatives or controlled finite interventions explicitly; value calibration cannot silently certify gradient quality. Producer-weighted derivative error and correction variance remain the metrics for exact-gradient calibration arms.

## 5. Two complementary training targets

Primary policy target: future cost/advantage under causally available information. Use held-out continuations, paired admissible branches and prequential outcomes. Pair branches with identical revealed prefixes and controlled/common future randomness where the simulator permits it. In an exogenous offline stream, replay the same subsequent observed inputs in both branches; all altered internal states and subsequent learning are recomputed. This is a realized continuation comparison, not access to those inputs at the original decision time.

Calibration target: exact derivatives/counterfactuals of specified completed computations. They teach known transport and expose missing directions. Backpropagation is the correct derivative reference for its stated objective; it is not the definition of the best future-learning policy. Audit correction preserves the declared raw-gradient mean only for that exact objective, not automatically for a learned future value or nonlinear optimizer.

A proposed training loss combines masked delayed future-value/advantage targets, audited directional derivatives for specified horizons, and an explicit work objective. Weights and horizon are registered on development data. Delayed targets train the current shared critic using prediction-time packets; they never change a previous prediction. Parameter/state staleness is modelled or invalidated. Training samples drawn from old regimes teach prediction only insofar as their conditional context transfers to current/future regimes.

Bootstrapped TD targets can reduce long-rollout demand but introduce target approximation and possible instability under function approximation, drift and off-policy data. Validate against small fully enumerated continuations. A forward simulator can generate counterfactual supervision, but simulator error must be audited against real outcomes. The critic does not validate its own invented future.

## 6. Small-scale experiment that distinguishes the proposal

Use one native temporal actor and exact tiny continuation branches. Keep its initial state, parameters, data, forward architecture and admissible action set fixed. Compare isolated predictors, a connected shared critic with reset state, and the same critic with persistent state; match budget or show the cost-quality curve. Include derivative-calibrated and future-advantage-trained arms with the same critic representation, so a changed target cannot masquerade as a connectivity gain.

First require fixed-packet learning, then disjoint continuation/action generalization. Measure advantage calibration, intervention ranking, directional derivative error, positive/negative transfer, future loss and full work. On controlled regime changes test whether a past-loss-descent action becomes inferior to a forward-looking action. Compare state adaptation, parameter adaptation and both. Assessment remains prequential: each action precedes its outcome; no hidden regime label enters unless legitimately supplied as input. Small teacher/audit work is charged.

The mathematical witnesses establish what accuracy would make the approach useful. They do not establish learnability of a particular feature interface, global stability, grokking or superiority at scale. Current Curie coarse-coordinate probes have not shown the required small-scale learning signal. The next already-admitted exact state-key contract stays unchanged; the subsequent shared-critic design and comparison are admitted as the bounded probe in section 8, without substituting the native forward architecture.

## 7. Collaborative candidate pools

A forward proposal policy builds an admitted intervention/route set R_phi(H_t,X_t). The critic ranks its members; audits reveal consequences for selected and sampled unselected alternatives. Both proposal parameters phi and critic parameters psi learn from future outcomes. Condition on a common feasible action space and a fixed budget. If

    delta_pool = min_{a in R_phi} A(a) - min_{a feasible} A(a),

and the critic has uniform error at most epsilon on the pool, the selected action satisfies

    A(a_hat) - min_{a feasible} A(a) <= delta_pool + 2 epsilon.

This follows from the finite-action bound. Proposal learning addresses coverage error; critic learning addresses ranking error. Charge pool construction and all candidate scoring, and include no-op. Exploration/audit support and diverse retained alternatives prevent unsupported self-confirming elimination. Optimizing a proposal through an uncalibrated critic can exploit critic errors: held-out realized interventions test that failure explicitly. Continuous proposals can differentiate a frozen critic; discrete proposals require controlled counterfactual or score-function signals. Predicted value alone does not certify pool improvement.

## 8. Admitted first native dynamics probe

`experiments/queue/enabler/curie_future_credit_v1_20261010T1140Z/manifest.json` admits exactly five sequential jobs: new intervention/causality/derivative contract, one two-step smoke, then fixed 128-step pilots at seeds 170/171/172. Each pilot fits isolated and connected shared critics on identical packets and future branches, followed by 32 proposal steps through the frozen connected critic. Dense supervised future loss supplies targets; DEV branch outcomes are withheld from selection. One bounded learned key-write candidate supplements nine retained basis/no-op candidates. Held-out evaluation separates realized proposal quality, pool coverage improvement and critic selection regret. Teacher, proposal and candidate work are charged.

Scope: initialized frozen depth-two native actor, 16 TRAIN and 16 DEV contexts per seed; current-state key-write interventions, not forward-weight adaptation, full deep counterfactual routing, regime adaptation or asynchronous execution. Critic parameter counts differ, so this first probe is a fit/transfer/dynamics gate, not an isolated connectivity win. Fixed checkpoints, no TEST, no efficiency claim. Zero/frozen predictors, uniform selection and enumerated within-pool oracle are references. Candidate target scaling uses TRAIN only. The depth-2/4/8 contract verifies native loss parity, causal feature packets, unchanged earlier losses and directional finite differences.

Decision: first establish acquired small-scale structure and held-out beneficial interventions. If TRAIN fails, diagnose representation/optimization. If only TRAIN succeeds, diagnose generalization. If proposals improve predicted values but worsen realized future costs, diagnose critic exploitation. Only a selected learner with a replicated signal proceeds to fully coupled forward-weight learning, controlled regime changes and depth/work scaling. The prior state-key contract must complete first. These jobs join Curie's guarded continuation after PAM, without changing AWS or the sealed benchmark fits.
