# Reciprocal event learning: prediction, credit and update dynamics

Curie, 10 October 2026; B1/R1 shared enabler. User-directed design combining forward side-information, persistent sparse asynchronous credit functions and learned optimizer parameters. This is a proposed formalism and proof programme, not a new performance or novelty claim. It extends notes 160–163 and the AWS reciprocal-update pilot without replacing admitted models or jobs.

## 1. Define the whole causal machine

For each addressed module i retain forward state s_i, credit state c_i, update state o_i and unresolved evidence records R_i. Parameters theta govern prediction; psi govern credit inference/transport; omega govern the update function. These parameter sets may share a basis, but their objectives and derivative paths must remain explicit. Let X = (s,c,o,R,theta,psi,omega) be the augmented machine state.

Between events, numerical states evolve by a specified transport map Phi(dt). At an event, only addressed modules perform transitions. For a linear decaying/rotating component, Phi(dt) = exp(A dt); stable dissipative generators bound homogeneous growth. Transporting a cotangent through a forward interval uses Phi(dt)^T, not Phi(dt)^(-1). Nonlinear, gated or state-dependent transport needs its actual Jacobian. Stability of individual decay components does not prove stability of the coupled learner.

An observation cycle is:

    (prediction, s_new, packet) = F_theta(s, c, input, elapsed_time)
    loss = scoring_rule(prediction, newly_observed_outcome)
    (predicted_credit, c_new) = C_psi(c, packet, outcome, elapsed_time)
    credit = residual_correct(predicted_credit, selected_audit_evidence)
    (delta_theta, o_new, learning_record) = U_omega(o, credit, packet, elapsed_time)
    theta_new = theta + delta_theta

Subsequent forward events train psi and omega from the consequences of these recorded decisions. Every record binds parameter/state versions and an information boundary. Outcome feedback may affect subsequent predictions, not the prediction scored against that outcome. The operational process includes learning itself, so exact differentiation of a fixed inference graph and exact differentiation through adaptation are different contracts.

No compulsory global forward/backward barrier is introduced. A credit event may arrive long after its originating write. An update event may be deferred or aggregated. Once-only record consumption, ordering at ties, overlap of pending records and overwrite semantics are part of the transition law. Parameters used by old records must be reconstructible or their drift must be treated as an approximation.

## 2. Forward evidence is structured, not just an activation

A packet can contain: causal address and time; parameter/state versions; selected cause and its probability; runner-up identities and margins already obtained by discovery; local messages/activations; local Jacobian or eligibility information; outcome-independent forecasts of future credit; and retained-state identifiers sufficient for an audited continuation.

Three targets are distinct:

1. Attribution: which hidden cause explains an outcome under the current generative model? Generated trajectories supply known causes; real-data posterior calibration needs additional correction.
2. Continuous sensitivity: how does an observed or future loss change with a producer activation/parameter?
3. Update utility: how does a finite parameter/state change affect subsequent predictive performance?

A route margin is side-information, not a measured counterfactual. A forced alternative with the same causal prefix and defined future randomness supplies a counterfactual. Losing-route utility does not alone supply its continuous content gradient. Every executed alternative and every discovered candidate is charged.

If exogenous future observations are independent of the action, paired rollouts may share those observations and random draws. In an embodied/action-dependent environment, holding future observations fixed can be the wrong intervention; use a specified environment transition model or justified off-policy estimator instead. Counterfactual correctness depends on the causal setting.

## 3. What the credit and optimizer activations must retain

Distinguish persistent numerical memory, retained differentiation evidence and slow parameters. Carrying c and o forward does not mean retaining an unbounded autograd tape.

When C or U executes, its learning record stores addressed inputs, pre-transition state, output, elapsed time, version, inclusion probability, and either the local derivative information or a reconstructible replay reference. It is retained until a later loss/continuation supplies supervision. Credit for its own parameters is temporal too: for a differentiable credit transition,

    E_next = D_c C E + partial_psi C.

Exact E can grow as credit-state dimension times credit-parameter count. The formalism does not remove that price by naming it an activation. Initial implementations use small shared local C/U functions and bounded declared differentiation horizons, with randomized residual audits where exact continuation evidence is reconstructible. Compression, discarded records, nonlinear thresholding and shortened horizons require explicit approximation contracts. Selective retention needs actual inclusion probabilities and support; deterministic expiry cannot be called full-horizon unbiased learning.

Credit memory may track uncertainty, unresolved delayed effects, model drift and time-dependent influence. Optimizer memory may track moments, curvature features, noise, age since update and previous update outcomes. These can share event storage/transport, but their meaning remains testable separately. Numerical time transport can be evaluated lazily on a read without touching every dormant module, when the transition law permits it.

## 4. A correctness anchor for sparse predicted credit

Choose the predictive objective and horizon first. Partition its exact fixed-version gradient into parameter blocks g_b, or use one explicitly valid graph-cut decomposition from note 163. Let h_b predict g_b from retained causal evidence. Freeze the predictor/propensities for the current audit, and draw I_b with recorded conditional inclusion probability p_b > 0. Define:

    g_hat_b = h_b + (I_b / p_b) (g_b - h_b).

For an exact audited g_b, conditional on graph, versions, predictor and p_b:

    E[g_hat_b] = g_b,
    E[||g_hat_b - g_b||^2] = (1/p_b - 1) ||g_b - h_b||^2.

Independence gives additive covariance across blocks; correlated audit scheduling requires inclusion covariance. If the audit is itself an estimator, its conditional mean and additional variance must be carried through the argument. If records were sampled earlier, their joint inclusion law matters too. Adapt p using information available before the draw; do not silently substitute an estimated probability for the probability actually used.

This is a control-variate/importance-correction identity, not a new theorem about convergence or efficiency. Sparse audits save work only if prediction, baseline application, discovery and state transport are also economical. Computing h_b and updating optimizer moments for every dormant parameter is a dense algorithm even when I is sparse. Cheap addressed baselines, exact lazy updates where possible, or separately analysed stochastic update policies are required.

Train h with detached audited targets using, for example,

    L_credit = sum_b (I_b/p_b) ||h_b - stop_gradient(g_b)||^2.

Its conditional expectation is the full block calibration loss. Treat probabilities as fixed in this supervised update; a learnable allocation policy has its own variance/work objective. Frozen predictions are corrected first; audit evidence trains the next predictor version. In parameter influence coordinates, this reduces audit variance when calibration improves. Cotangent-space error alone may underweight directions with large producer Jacobians.

## 5. Subsume optimizer parameters in the learning dynamics

U_omega can learn step scale, momentum/decay, preconditioning, update timing, block sharing and uncertainty-sensitive action. Discrete update selection/recruitment has score-function or counterfactual credit; continuous update transforms have actual local derivatives. Establish low-dimensional shared functions before unconstrained per-parameter meta-parameters.

The learning-system objective evaluates later predictions:

    J(psi,omega) = E[sum_(k=t+1)^(t+H) w_k loss_k(X_k)]
                  + lambda * E[measured learning work over the same interval].

The resource coefficient and loss weights are declared, not tuned against sealed assessment. Differentiation follows the actual augmented transitions, including how credit affects U, how U changes theta/o, how those affect later inference, and how reused c affects predictions. If psi is updated inside the horizon and that path is detached, the result is a specified stop-gradient approximation, not the exact meta-gradient. Counterfactual evaluations use the same information boundary and initial augmented state.

An unbiased raw g_hat does not imply E[U(g_hat)] = U(g): clipping, normalization, Adam-like moments and learned nonlinear update rules change the distribution. Either analyse the resulting stochastic algorithm directly, constrain U to a compatible linear map for an exact mean-step witness, or audit finite update utility. Sparse updates also change regularization/moment dynamics unless their equivalence or revised objective is stated.

Calibration and future utility are complementary objectives. A useful update can differ from the instantaneous steepest-descent step. Conversely, good short-horizon adaptation can conceal poor long-term attribution. Report both, using held-out continuations and delayed-credit stress cases. Meta-training cost must be charged and amortized over a stated deployment/adaptation workload.

## 6. Close the recursion without hiding another dense learner

Begin with a finite hierarchy: a small shared C/U model, local exact derivatives for short operations, explicit continuation horizons and sparse audits of extended influence. Those local operations are permitted engineering primitives; their costs are counted. We do not presume a recursively trained infinite tower of credit systems, or a free exact derivative at the top.

For a one-step witness, theta_next = theta + U_omega(...). If later loss depends on omega only through this update, its derivative is (partial_omega U)^T times the later theta cotangent. Additional direct/state paths must be added when present. The later forward computation supplies the consequence; a derivative estimator or a controlled counterfactual still supplies its sensitivity. Simply observing that later loss improved cannot identify every contributing optimizer parameter.

Generalization to deeper C/U models requires the same temporal-credit and work tests as the forward model. Full differentiation through small augmented trajectories is the correctness reference. A larger sparse execution is admitted only after gradient/update/causal/resource contracts pass.

## 7. Batch and sequence boundaries

Online sequences preserve elapsed time, state and pending causal records within the declared stream. Independent episodes reset stream-specific s/c and unresolved records; shared psi/omega persist. Optimizer state may legitimately persist across shuffled minibatches; activation/causal memory must not invent temporal dependence between independent examples. If a stateful minibatch learner is intended, its order-dependent process is the algorithm and must be evaluated across orderings. Frozen deployment, state-only adaptation and weight adaptation are different protocols.

## 8. Mathematical and empirical contribution to earn

Known ingredients: synthetic gradients/decoupled interfaces, e-prop/eligibility traces, recurrent stochastic sensitivity estimators, wake-sleep/reweighted recognition learning, learned optimizers and hypergradient adaptation. Our proposed contribution is their event-native integration with explicit causal/versioned evidence, factual and counterfactual credit, temporal credit/update states, sparse residual correction and complete resource bounds. This document does not establish uniqueness; a broader literature audit and a precise distinction from each closest construction remain required.

Required proofs/witnesses:

- Causal admissibility of all packet reads and delayed updates, including ties and overwrite.
- Exact augmented-machine gradients for tiny depth-2/4 models, including continuous producers, discrete routes and optimizer parameters; no overlapping-cut double counting.
- Conditional mean/variance of residual estimators and audited retention; separate nonlinear optimizer-step bias.
- Semigroup/Jacobian transport identities and explicit truncation/version-drift errors.
- Coupled-learning stability under stated assumptions; two-time-scale schedules alone are not a global convergence proof.
- Complete storage/operations as functions of depth, active routes, pool capacity, credit horizon and pending-record count. Fixed packet size alone does not bound outstanding packets under unlimited delay.

Required experiments: same forward architecture, initial parameters, data/order and validation budget; ordinary backprop plus Adam, exact online reference where affordable, frozen predictor, learned predictor without correction, corrected reciprocal learner, fixed versus learned U, and c/o reset controls. Depth 2/4/8 first, separate width/pool/horizon axes, delayed tasks that actually require intermediate layers. Primary evidence is complete work to matched quality, including credit/optimizer training and failed steps. No new external-architecture controls are admitted. Preserve AWS online3 and reciprocal-update ownership; Curie integrates the existing native producer repair first.

Native note-163 contract completed 10 Oct: plain and normalized two-layer R1 gradients decompose to machine precision, and enumerated residual audit means agree with full gradients. This establishes the correction identity in a small native graph; it does not establish learned credit calibration, sparse runtime, persistent credit/update feedback or depth scaling.

Primary precedents:

- Jaderberg et al., [Decoupled Neural Interfaces using Synthetic Gradients](https://arxiv.org/abs/1608.05343).
- Bellec et al., [A solution to the learning dilemma for recurrent networks of spiking neurons](https://www.nature.com/articles/s41467-020-17236-y).
- Tallec and Ollivier, [Unbiased Online Recurrent Optimization](https://arxiv.org/abs/1702.05043).
- Bornschein and Bengio, [Reweighted Wake-Sleep](https://arxiv.org/abs/1406.2751).
- Andrychowicz et al., [Learning to learn by gradient descent by gradient descent](https://arxiv.org/abs/1606.04474).
- Baydin et al., [Online Learning Rate Adaptation with Hypergradient Descent](https://arxiv.org/abs/1703.04782).
- Grathwohl et al., [Backpropagation through the Void](https://arxiv.org/abs/1711.00123), as cited in note 163.

## Connected future-directed credit target

[CONNECTED_CREDIT_CRITIC.md](CONNECTED_CREDIT_CRITIC.md) formalizes the founder clarification: the primary learning-policy objective is future conditional performance, including changing distributions and inductive structure. A connected persistent intervention critic supplies continuation values/cotangents; exact derivatives of completed computations supply calibration and transport targets. Finite-horizon Bellman recursion applies on sufficient augmented state or full causal histories, with explicit time dependence. No stationary regime, tiny disconnected critic, free simulator or automatic value-to-derivative accuracy is assumed.
