# When learned credit actually replaces backprop

12 October 2026. B10 and shared B1/R1 learning enabler. User asks whether standard backprop plus learned credit necessarily disadvantages efficiency. In the current market quality experiment, yes: complete exact within-window boundary credit is retained and additional learned field, predictive probes and future teaching are paid. The actor gradients are truncated at32-event window boundaries, not differentiated through the whole day. No market efficiency claim is supported.

## Accounting before architecture

Let C_local be unavoidable current readout/local update work, C_transport exact long-range credit work, C_predict learned credit inference and its state updates, C_probe candidate discovery/response work, C_teach future-utility teaching, and C_opt additional critic optimizer work. With transport audits admitted at fraction p and teaching at fraction q, expected complete update cost is

    C_learned = C_local + C_predict + C_probe + p*C_transport
                         + q*(C_teach+C_opt).

For the same number of learning opportunities, exact learning costs C_local+C_transport. Replacement saves work only if

    C_predict+C_probe+q*(C_teach+C_opt) < (1-p)*C_transport.

This is an accounting identity, not a quality guarantee. Include acquisition A and compare total work to matched quality: A+N_learned*C_learned versus N_exact*(C_local+C_transport). Saved operations are not necessarily saved CPU time or hardware energy. Current small-market boundary transport is a small share of exact learning, so an elaborate per-event critic may lose even when most boundary audits are removed. Measure components rather than assuming that depth or capacity automatically repairs this inequality.

## What can stay exact

Small selected local derivatives and observed readout error can remain exact. The work to replace is repeated propagation/replay through persistent history, depth and losing routes. If all dense long-range gradients are still computed for every update, the critic is an addition. Computing exact gradients and then discarding them is not replacement. Future teacher differentiation can also quietly recreate that cost; its admission rate and actual replay must be charged.

## Audited replacement and its limits

At a producer boundary, let predicted factual cotangent be c_hat, exact target c, and audit A drawn after c_hat is fixed with positive probability p conditional on causal information. The residual estimator

    c_tilde = c_hat + A/p * (c-c_hat)

has conditional mean c, and residual variance proportional to (1/p-1)||c-c_hat||². Through a fixed producer Jacobian this preserves the mean parameter gradient; the relevant error norm is ||J^T(c-c_hat)||². Adaptive p is valid when known and sampled after prediction with positive support. Gradient clipping, Adam normalization and finite steps do not preserve this unbiasedness in general. They need actual-update and future-quality tests. Prospective corrections deliberately change the target and must be separated from factual residual calibration.

Never calculate c on an unaudited step merely to feed the critic; use observed forward response and local error. Rare exact receipts can teach factual calibration, while rarer actual future-utility receipts teach prospective direction and optimizer controls. Score every forecast before outcome-dependent updates. Carry numerical critic state with versioned receipts; truncate or replay gradient history explicitly.

## Admission requirement for a replacement experiment

First measure exact local/transport, critic inference, forward response, future replay and both optimizers under the actual integrated model. Select one bounded budget from those measurements, not a blind audit grid. Require learned credit to preserve useful forecast quality against frozen credit and ordinary online learning, then compare complete work to matched quality. Retain the model's temporal races and addressed state; admit fewer informative critic events and delayed teacher receipts if they earn their cost. A sparse event-native architecture alone does not make its current Python execution economical.

The running forecast-response comparison answers an information/quality question. Its exact anchors are explicit. It cannot pass the replacement-efficiency requirement, even if its quality gate passes.


**Native credit replacement budget (12 October, measured):** Same2048TRAIN acquisition and512DEV cost profile. BPTT boundary transport/proposals0.780s; native response/inference2.944s, future teacher4.015s. Even deleting all exact transport/proposal work cannot pay the current response/inference phase; fewer audits alone cannot yield a same-opportunity wall advantage. Local differentiation2.025/2.325s for BPTT/online. Arm wall3.585/10.362s includes setup/verification0.729/0.022s. Shared acquisition43.002s and original5750.687s predictor fitting are additional. Phase residuals<0.0002s; proposal parity passes. Small profile is cost diagnosis, not a new quality confirmation. Prioritize shared-leg credit reuse, sparse event/teacher admission and cheaper native execution before another replacement fit. V1 smoke timing guard stopped; v2 charges omitted setup/verification.
