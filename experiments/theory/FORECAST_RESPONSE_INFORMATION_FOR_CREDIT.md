# Forecast-response information for economical credit

12 October 2026. B10 diagnosis after completed native optimizer-control and regularized-utility comparisons. This is a conditional analytical result and an interface hypothesis, not a measured quality gain or a claim of field novelty.

Let H be the history available at update time, U_theta(a,H) the actual parameter update controlled by learned credit a, and Y future outcomes. Define the utility derivative

    T = D_a U_theta(a,H)^T grad_theta L_future(theta+U_theta(a,H), Y).

The critic receives causal information Z. Under squared teacher error, its optimal prediction is E[T|Z]. Its irreducible error is E[Var(T|Z)]. Enlarging its information with a causal forward response R reduces optimal error by exactly

    E ||E[T|Z,R] - E[T|Z]||^2.

This follows from orthogonal conditional projections for square-integrable T. A bigger network cannot recover information absent from Z. Conversely, extra responses help only when they change the conditional expectation; input dimension alone proves nothing. This projection identity is a known probability result applied here to the actual-update teacher.

The implemented producer P_theta exposes persistent numerical state. The completed reachability contract finds 3756/18164 actor parameters with zero current producer VJP and nonzero observed-loss gradients. Ordinary backprop still trains these parameters. A state-only counterfactual response P_(theta+delta)-P_theta cannot directly reveal a perturbation confined to these heads. Optimizer-step control reaches them, so its utility can depend on responses not explicitly represented by that counterfactual input. Other critic inputs may indirectly contain this information; insufficiency has not been proved for their complete joint distribution.

A diagnostic extension should expose a causal predictive-law response as well as the persistent-state response: evaluate or reuse the forward forecast at the last observed history under factual parameters and an already charged candidate update. Record a compact typed response for clock rates, mixture weights and event-mark probabilities, or directional predictive-law features. No future outcome may select the probes. Bind each packet to the actor version and issue history. Prefer already computed forecasts; charge additional replay, discovery and storage where required.

This retains temporal clocks, hard-race native credit, causal activation history and next-window teaching. It does not replace the model with a dense recurrent critic. Raw parameter coordinates are not route similarity: useful generalization concerns effects on future predictive laws conditional on history. Two routes can be equivalent under the current probes yet differ under another history; probe coverage and identifiable effects must be measured.

Required development decision: compare state-only and state-plus-forecast responses with the same source acquisition, full actor, stable teacher, six controls and paid work. First verify a head-only perturbation has zero producer response but nonzero predictive-law response and that packets contain only observed history. Do not scale until this information diagnosis changes a measured decision.


Completed TRAIN-only interface contract `curie_b10_forecast_response_20261012_v1_contract`: clock.bias +0.01 gives persistent-state response 0.0, predictive-clock response 306.605; state/forecast widths 88/80. This confirms a missing explicit response channel, not insufficiency of the complete existing critic information or a quality gain. Source and input hashes preserved.
