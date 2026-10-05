## Appendix. Tokenized language: integrated learning development

Sleeping Machines pursues a general-purpose substrate for language and reasoning, multimodal world models, embodiment, event-native analytics, continual learning, communication, self-design and hardware. These language diagnostics test constructions within that family.

Seed6; GPT-2 FineWeb; 2,048 training tokens, 8,160 presented targets over64updates, 1,024 disjoint development targets. P16/D2/H2/U4:256persistent memory scalars per lane, four selected writes and16scored keys per target. Public validation remains reserved. Loss is natural-log NLL; lower is better. Throughput includes training work, excludes evaluation.

| Member | Initial dev | Best trained dev | Final dev | Targets/s |
| --- | --- | --- | --- | --- |
| Local value | 8.906910 | 9.131615 | 9.888922 | 404.09 |
| First-site full suffix | 8.906910 | 9.130419 | 9.865284 | 342.83 |
| First-site K4 suffix | 8.906910 | 9.130868 | 9.844923 | 351.84 |
| Uniform-site K4 suffix | 8.906910 | 9.063122 | 10.354817 | 386.82 |

Uniform-site K4 wins the trained development comparison against first-site K4 by0.067745NLL, single seed at equal data/presentations. It combines broader site exposure, utility weighting and removal of immediate-only route credit.

All trained members lose to initialization. Uniform-site late dev10.354817 loses to first-site K4late9.844873. Initial weights now remain eligible for selection; historical trained-only fields stay preserved with the correction beside them.

Mechanisms: persistent state, race clocks/transport, sparse hard writes, separate selection/value roles, small messages, depth and actual counterfactual writes. Credit remains chunk-bounded; all keys are scored. The uniform-site learner uses winner-only factual/replay proposals. Memory occupancy alone is not predictive capacity.

## Appendix. Tokenized complete work and scaling prerequisites

Synthetic P16/D2/H2/U4/B8/T16, vocabulary50,257; all adaptive tails exercised, first AdamW update. Arithmetic includes factual computation, readout, alternative sampling, replay, backward, clipping and optimizer. Special functions have a separate ledger; random sampling work is unquantified. These are executed steps, not whole-fit extrapolations or hardware energy.

| Scoring | Targets | Step MFLOPs | MFLOPs/target |
| --- | --- | --- | --- |
| Full scoring | 128 | 253.930317 | 1.983831 |
| K4 scoring | 128 | 218.255413 | 1.705120 |

K4 arithmetic is14.05%lower in this realization. Replay readout falls47.286432to12.713512MFLOPs. Complete fitting and per-target fitting FLOPs on the actual corpus, and winner-only inference FLOPs, are not yet traced; no whole-fit resource win is claimed.

Exact actual-driver interruption/resume passed for integrated, sampled-position and uniform-site learners, including all relevant RNGs. Full-score sampled learning equals the original trajectory. Wider decoder tails preserve normalization and initial token priors; default parameterized learning parity and full-width resume passed.

The public reference target reuses the modded-nanoGPT2025-01-26log:3.2774NLL,695,992,320training presentations,10,485,760reserved validation targets. That protocol/context differs from these development rows. The exact attention/context audit precedes benchmark scoring; no dense reference is retrained.

Prepared AWS8K comparisons test broader credit and full-width tails before selecting a scaling member. They are unrun here and have no predicted scores. Rough scaling laws follow a selected promising member and actual resource measurements. Delivery/admission remains unobserved.
