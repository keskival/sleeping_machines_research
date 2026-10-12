# Market policy classes, shared history and adaptive credit

Curie, 12 October 2026. B10; formal motivation for the native test-time comparison. This is a modelling construction and an experimental hypothesis, not a claim to identify traders or beat market participants.

## Shared history couples the clocks

Write the observed marked trade history H_t={(t_i,k_i):t_i<t}. Participant j has policy pi_j, private predictive state s_j and marked conditional intensity lambda_jk(t|H_t,s_j). Its next event changes H, so other clocks change through their policy's response to the updated common record. Nonlinear history-dependent clocks are therefore a natural representation of this interacting stream; independent unconditioned traces are not required.

When event sources can be defined without simultaneous collisions, their conditional intensities add: lambda_k=sum_j lambda_jk. A competing-clock construction with conditionally independent latent clocks has survival S(tau|H)=product_j S_j(tau|H) and aggregate hazard sum_j h_j(tau|H). That is a valid model construction, not evidence that real participants become independent after conditioning on the observed trace. Private states and unobserved information can leave residual dependence.

Recorded simultaneous trades remain distinct sequence events. Score the observed recording cell with its integrated mass, rather than rewarding an unbounded density at a rounded zero gap. This is the existing B10 six-type/1us protocol.

## Predictive equivalence, not deanonymization

Participants following the same policy can share its parameters. Interchangeability additionally requires matching predictive state or marginalizing their state distribution. For a class c, the observable intensity is Lambda_ck=sum_{j in c} lambda_jk. If all members share state-conditioned response f_ck(H,s), this can be written as the integral of f_ck against the class's time-varying state/occupancy measure rho_c(ds,t). A model can learn the response family and the aggregate occupancy without one network per actor.

An observational equivalence relation over latent explanations is equality of the conditional law of future observable traces, for the declared histories and prediction horizon. Two identical clocks and one clock with their summed intensity may be observationally indistinguishable. A unique recovery of trader identities, their count or causal ownership is neither necessary for forecasting nor implied by good forecasts.

This relates to FAS as interleaved latent mechanisms and persistent state. FAS's independent hidden generators and markets' common-history coupling are different assumptions. FAS oracle identities cannot be used to teach a fair anonymous model. Public market marks may address memory; they are event types, not inferred trader identities.

## Changing strategies and learning the credit rule

The class occupancy rho_c and response parameters may evolve. Even when a policy family is reusable, its useful features, predictive dependencies, time scales and update geometry may change. The forward model theta predicts the marked stream; a credit model phi maps issued state/response information and arrived feedback into a learning update g_phi. Causal updates have the form

    theta_{n+1}=U(theta_n, optimizer_state_n, g_phi_n(available_history_n)).

Future observed losses can teach phi through the actual response of U. This targets the utility of an update under subsequent data, rather than only imitation of a past exact derivative. Outcome-law estimation and prospective utility are different objects: a mean-changing utility field does not inherit unbiased-gradient claims from a calibrated residual field.

Online ordinary backprop already adapts theta. The distinctive hypothesis is that adapting phi changes how effectively theta responds to the new process. The experiment must compare online phi with frozen phi and with ordinary online native backprop, from matched acquired states, and must charge the future teacher and optimizer. A frozen predictor alone is insufficient to attribute an advantage to learned credit.

## Current experiment and investor interpretation

NATIVE_TEST_TIME_PROTOCOL.md restores the completed18164parameter market model rather than a weakened predictor; the new native credit model has34728parameters. Native hard-race memory, causal activation-distribution summaries, actual producer-response inputs and prospective one-step Adam teaching are combined. The actor still uses the earlier dense temporal implementation, and actor/critic shared-bank access is not integrated here. This comparison isolates a concrete learning-system change on the existing full forecasting task.

All Aug25 forecasts are scored before the corresponding outcomes are used for updates. Aug25 was part of earlier validation, so this is test-time-training behaviour on development data, not a sealed test claim. Better time/type forecasts would support probabilistic market modelling; price direction, execution utility, cost-inclusive profit and superiority to other participants are separate quantities.

Known antecedent: history-coupled point-process modelling of market activity is surveyed by Bacry, Mastromatteo and Muzy, "Hawkes processes in finance" (2015), https://arxiv.org/abs/1502.04592. The present contribution being tested is the integration of our native temporal credit machinery with future-taught online updates, not the invention of point-process market models.
