# Forward route consequences beyond factual gradient windows

13 October 2026 · R1 engineering candidate for large-scale language.

The concrete limitation is in the active implementation: numerical memory persists, but both factual gradients and the existing route teacher stop at the next 16-token boundary. A write can influence prediction beyond that boundary without receiving credit for that influence. This is a restriction of the driver, not the family. The current deep acquisition is preserved as the comparator.

## The change and its retained mechanisms

`horizon_allocated_depth_token_language.py` separates factual gradient width W from forward consequence window count K. For a site at token t in an optimizer chunk of length n, the consequence endpoint is

    e(t,K) = min(n, (floor(t/W)+K) W).

K=1 is the existing endpoint exactly. K=4 permits effects across three subsequent detach boundaries, capped by the same optimizer chunk. Factual state carry, gradient detach points, hard temporal races, private addressed banks, separate keys/values, final-stage prediction, TRAIN site allocation, proposal support, clipping and Adam are retained. Only the scored causal continuation and its allocation proxy change. This is a bounded longer-horizon route teacher, not a learned compact critic or full document-history learning.

The shadow forward pass reuses the saved pre-window state and issued race tape. Subsequent observed tokens enter as causal continuation inputs; targets enter only loss scoring. The forced receiver retains the existing teacher's first-arrival time convention. Its loss contrast is conditional on that intervention and replay rule; it is not the derivative of the complete stochastic routing distribution or a differentiable alternative-write gradient. Downstream races and stored values evolve on the actual altered continuation.

## Estimator accounting

Let Delta_s be alternative-minus-factual token loss for a lane at suffix position s. Uniform outcome-independent position sampling estimates the suffix mean. Multiplying by (e-t)/n estimates its suffix contribution to chunk-average loss. The existing inverse site probability, nonwinner proposal weight and inverse teacher admission probability then preserve the declared raw conditional route-credit surrogate mean under the extended horizon. These are before global gradient clipping and Adam; their nonlinear updates need not preserve this mean.

The allocation proxy previously used the short suffix weight. The new allocator rescales the label supplied to that proxy by (e(t,K)-t)/(e(t,1)-t), leaving the actor's original extended-horizon contrast unchanged. Positive site support remains. Longer consequences can raise both usefulness and variance; the allocation proxy is still local score-space demand, not full parameter-update variance.

Replay reaches the last sampled loss position. Scoring four sampled targets keeps readout scoring bounded but does not make intermediate state evolution free. Charge all replayed tokens, key discovery, memory updates, factual backpropagation, clipping and dense Adam. The instrumented reserved fits separate core, readout, backward and optimizer CPU time; CPU timing is not physical energy.

## Native integration and acquisition gate

The pending deep diagnostics run a frozen TRAIN-only null-winner replay check: exact features, full numerical state and race-tape consumption; an alternative intervention must preserve the prefix before its issued site. They record finite post-boundary loss differences and feature effects on the native D8/D16 member. These are dependencies under intervention, not quality gains. Diagnostic forward and scoring targets are recorded separately from acquisition.

After the current D8/8M, D16/8M and D8/32M ladder finishes successfully, the continuation checks these contracts and whether post-boundary feature effects are measurable. It then runs one 64-update integrated resource smoke using the full reserved 8M acquisition settings, verifies finite losses/every-layer input gradients and measured RSS/throughput, and resumes the exact checkpoint to 8M. No small-data quality veto, architecture grid or external-model fit. The candidate is compared with the completed same-depth, same-seed, same-data baseline, reporting final and short-DEV-selected long-history scores plus all work. The extra teacher horizon is explicitly a learning-rule difference.

If the contracts or resource envelope fail, stop and diagnose. If the completed quality comparison loses, retain the comparator and diagnose horizon variance, relevance and actual update geometry before admitting more fitting. Public validation stays sealed. A development gain would support the next larger acquisition; it would not by itself establish a frontier-language advantage.
