## Appendix. Tokenized8K: replicated learning and memory-use decisions

Sleeping Machines pursues a general-purpose substrate for language and reasoning, multimodal world models, embodiment, event-native analytics, continual learning, communication, self-design and hardware. This integrated stage advances the Transformer-leading language program; tiny fits are engineering diagnostics.

Both horizons beat initialization in both seeds. GPT-2 FineWeb:8192admitted training tokens,16368fitting target presentations/two passes,32AdamWupdates,2040development targets. P16/D2/H2/U4,batch64; credit16/64 share data/updates/cadence. Initial-inclusive development selection every8updates; public validation untouched. Four selected writes,16scored keys and256persistent memory scalars per lane. The bounded local fits preserve the immutable AWSqueues.

| Seed | Credit | Initial NLL | Selected NLL | Gain | Context gain | Memory gain |
| --- | --- | --- | --- | --- | --- | --- |
| 6 | 16 | 8.340313 | 8.297491 | 0.042822 | 0.017759 | 0.000381 |
| 6 | 64 | 8.340313 | 8.300131 | 0.040182 | 0.001898 | -0.000268 |
| 7 | 16 | 8.340313 | 8.286427 | 0.053886 | 0.027515 | 0.000980 |
| 7 | 64 | 8.340313 | 8.290236 | 0.050078 | 0.017711 | 0.000449 |

Credit16mean selected8.291959 beats credit64mean8.295183, and is better in both paired seeds; the0.003224mean gap is smaller than seed spread. Ordinary measured fitting throughput is395–422targets/s, peakRSS452456–453252KiB. Retain both completed records. Numeric state carries across credit boundaries; truncating derivatives does not reset state.

Context gain uses the same frozen readout with a constant mean of8184causal TRAINfeatures. Memory gain is NLL after per-token addressed-state/arrival/seen erasure minus intact NLL; negative means the intervention improves loss. Recurrent-message erasure costs0.017843–0.021955NLL. Matched route RNG, source binding and token partition parity pass. These are frozen interventions, not retrained ablations; the constant-feature control is not an optimally refitted unigram.

| Alternative | Seed | Selected NLL | Gain vs credit16 | Parameters |
| --- | --- | --- | --- | --- |
| Full-width tails | 6 | 8.302969 | -0.005478 | 3292836 |
| Memory gain2 | 6 | 8.310055 | -0.012563 | 2115464 |
| Full-width tails | 7 | 8.286345 | 0.000082 | 3292836 |

Full-width decoder tails lose at seed6 and are effectively tied at seed7. Retain the narrower decoder. First-tail training exposure is695targets/data pass, so this comparison exercises learned tail projections. The memory-gain intervention doubles the existing unit output contribution while preserving all temporal/state/routing/learning mechanisms and parameter count; it is selected only from completed fits. See TOKEN_MEMORY_COUPLING_20261005.md for derivation and contracts.

| Native member | Fit targets | Whole fit GFLOPs | Fit MFLOPs/target | Eval MFLOPs/target |
| --- | --- | --- | --- | --- |
| Credit16seed6 | 16368 | 13.534669 | 0.826898 | 0.272633 |
| Credit64seed6 | 16368 | pending | pending | pending |

Actual whole-fit traces must reproduce each control trajectory. Arithmetic includes factual computation, all-key discovery, suffix replay, exact target readout, backward, clipping, optimizer and in-step diagnostics; initialization/frequency counting,evaluation and serialization excluded. Special functions separate; random-sampling work unquantified. Evaluation includes scorer reductions. Pending cells contain no predicted work or quality. Equal updates/data here are not an iso-FLOP claim. Selected Transformer-reference quality and resource protocol remains the larger benchmark target.

## Appendix. Language strengths: measured decisions and complete work

The target data region deliberately favors strong Transformer references over LSTMs; tiny native fits diagnose learning before that larger comparison. The native construction chooses temporal computation, persistent memory lifetime and sparse addressed activity independently of Transformer topology or fixed windows. Matched-history and longer-native-history comparisons are labelled separately; causal inputs and scored targets stay explicit. Capacity beyond activity earns promotion through better held-out prediction with discovery and learning fully charged.

Controlled learning-rate test: seed6, GPT-2 FineWeb2K, P16/D2/H2/U4, batch64/credit64, uniform K4 actual alternative-write utility,16updates and1,016development targets. Initial weights eligible. Lower learning rate loses selected quality by0.046604NLL; retain .003. Its better final loss does not replace the better selected incumbent. These are development comparisons.

| Recipe | Selected NLL | Final NLL | Fit targets | Targets/s | RSS KiB |
| --- | --- | --- | --- | --- | --- |
| Retained lr .003 | 8.819470 | 9.111889 | 8160 | 404.6 | 451728 |
| Alternative lr .001 | 8.866074 | 8.866074 | 8160 | 423.2 | 451308 |

Decoder exposure uses train-only frequency rank and eight contiguous lanes. At2K every target reaches the head: identical default/full-width development curves do not test learned tail rank. The existing8Kcomparison reaches the first tail. Branch counts below are targets per data pass; head cutoff2,000, tail boundaries10,000/30,000/50,257.

| Train tokens | Distinct | Head | Tail1 | Tail2 | Tail3 |
| --- | --- | --- | --- | --- | --- |
| 2048 | 796 | 2040 | 0 | 0 | 0 |
| 8192 | 2696 | 7489 | 695 | 0 | 0 |
| 32768 | 6778 | 26915 | 5845 | 0 | 0 |
| 65536 | 10914 | 51272 | 13343 | 913 | 0 |
| 262144 | 23144 | 196953 | 47809 | 17374 | 0 |

Actual native work: multiply-add=2FLOPs; arithmetic excludes separately counted special functions and unquantified random sampling. Whole fitting includes factual core, all-key scoring, alternative discovery, suffix replay, exact readout, backward, clipping, optimizer and in-step diagnostics. Initialization/frequency counts, evaluation and serialization are outside that fitting column. Selected evaluation includes scorer reductions, with no gradient/optimizer. Audit instrumentation wall time is not ordinary throughput. Public Transformer work remains in its separate audited protocol; no matched-compute win is inferred here.

| Boundary | Targets | Total GFLOPs | MFLOPs/target | Coverage |
| --- | --- | --- | --- | --- |
| Whole fit | 8160 | 6.183337 | 0.757762 | complete |
| Selected evaluation | 1016 | 0.296939 | 0.292263 | complete |

Next: existing AWS8Kpaired seeds/horizons, trained memory utility, informative decoder comparison, then measured width/data and fixed-activity pool scaling. Public validation remains reserved. See experiments/TOKEN_STRENGTH_EXECUTION_20261005.md for the ordered strength tests.

## Appendix. Tokenized language: confirmed small-fit learning

Sleeping Machines pursues a general-purpose substrate for language and reasoning, multimodal world models, embodiment, event-native analytics, continual learning, communication, self-design and hardware. These diagnostics select constructions within that family.

Both integrated members beat initialization in both seeds. GPT-2 FineWeb:2,048 admitted training tokens,8,160 training targets over four passes and16updates,1,016 scored development targets. Optimizer batch64 per lane; credit16/64 share data, updates and evaluation cadence. Natural-log NLL, lower is better; initialization is eligible for selection. Public validation is reserved.

| Seed | Credit | Initial dev | Selected dev | Gain | Context gain |
| --- | --- | --- | --- | --- | --- |
| 6 | 16 | 8.906910 | 8.812527 | 0.094383 | 0.005029 |
| 6 | 64 | 8.906910 | 8.819470 | 0.087440 | 0.015265 |
| 7 | 16 | 8.906910 | 8.813838 | 0.093072 | 0.011278 |
| 7 | 64 | 8.906910 | 8.812853 | 0.094057 | 0.011398 |

Context gain compares actual features with constant mean causal TRAIN features through the same frozen learned readout. All four contributions are positive. This intervention does not isolate persistent memory from current-token features or refit an optimal unigram control. Much of the total gain is marginal recalibration.

Frozen history interventions: erasing the recurrent event message worsens NLL0.04618–0.06734 across all four checkpoints; erasing addressed memory changes it by−0.001080 to+0.001388. Matched RNG and intact partition parity pass. Messages can carry multi-event history. These out-of-distribution interventions target addressed-memory utilization for diagnosis; they are not retrained ablations.

Credit16 wins selected quality at seed6; credit64 wins at seed7 and has better final loss in both seeds. Retain both for the matched AWS8K comparison. Changing optimizer batch16 to64 also reduced updates64 to16 at equal targets: a recipe improvement, not an isolated horizon effect.

P16/D2/H2/U4 supplies256 persistent memory scalars per lane, four selected writes and16 scored keys per target. Mechanisms include race clocks, addressed persistent state, separate keys/values, small messages, depth and actual counterfactual write credit. Decoder rank and credit boundaries are selectable design choices. These are confirmed development learning gains; the public benchmark comparison is the next larger test.

## Appendix. Tokenized language: integrated learning development

Sleeping Machines pursues a general-purpose substrate for language and reasoning, multimodal world models, embodiment, event-native analytics, continual learning, communication, self-design and hardware. These language diagnostics test constructions within that family.

Historical optimizer-batch16 recipe, seed6; GPT-2 FineWeb; 2,048 admitted training tokens, 8,160 presented targets over64updates, 1,024 development tokens yielding1,016 scored targets after8lane starts. Earlier notes called these1,024targets; saved losses are unchanged. P16/D2/H2/U4:256persistent memory scalars per lane, four selected writes and16scored keys per target. Public validation remains reserved. Loss is natural-log NLL; lower is better. Throughput includes training work, excludes evaluation.

| Member | Initial dev | Best trained dev | Final dev | Targets/s |
| --- | --- | --- | --- | --- |
| Local value | 8.906910 | 9.131615 | 9.888922 | 404.09 |
| First-site full suffix | 8.906910 | 9.130419 | 9.865284 | 342.83 |
| First-site K4 suffix | 8.906910 | 9.130868 | 9.844923 | 351.84 |
| Uniform-site K4 suffix | 8.906910 | 9.063122 | 10.354817 | 386.82 |

Uniform-site K4 wins the trained development comparison against first-site K4 by0.067745NLL, single seed at equal data/presentations. It combines broader site exposure, utility weighting and removal of immediate-only route credit.

All trained members in this historical batch16 recipe lose to initialization. Uniform-site late dev10.354817 loses to first-site K4late9.844873. Initial weights now remain eligible for selection; historical trained-only fields stay preserved with the correction beside them.

Mechanisms: persistent state, race clocks/transport, sparse hard writes, separate selection/value roles, small messages, depth and actual counterfactual writes. Credit remains chunk-bounded; all keys are scored. The uniform-site learner uses winner-only factual/replay proposals. Memory occupancy alone is not predictive capacity.

## Appendix. Tokenized complete work and scaling prerequisites

Synthetic P16/D2/H2/U4/B8/T16, vocabulary50,257; all adaptive tails exercised, first AdamW update. Arithmetic includes factual computation, readout, alternative sampling, replay, backward, clipping and optimizer. Special functions have a separate ledger; random sampling work is unquantified. These are executed steps, not whole-fit extrapolations or hardware energy.

| Scoring | Targets | Step MFLOPs | MFLOPs/target |
| --- | --- | --- | --- |
| Full scoring | 128 | 253.930317 | 1.983831 |
| K4 scoring | 128 | 218.255413 | 1.705120 |

K4 arithmetic is14.05%lower in this realization. Replay readout falls47.286432to12.713512MFLOPs. Complete fitting and per-target fitting FLOPs on the actual corpus, and winner-only inference FLOPs, are not yet traced; no whole-fit resource win is claimed.

Exact actual-driver interruption/resume passed for integrated, sampled-position and uniform-site learners, including all relevant RNGs. Full-score sampled learning equals the original trajectory. Wider decoder tails preserve normalization and initial token priors; default parameterized learning parity and full-width resume passed.

The public reference target reuses the modded-nanoGPT2025-01-26log:3.2774NLL,695,992,320training presentations,10,485,760reserved validation targets. That protocol/context differs from these development rows. EOS/reset/target and layered attention boundaries are source-audited; historical data hashes are absent and no checkpoint is released by this record. No dense reference is retrained.

Prepared AWS8K comparisons test broader credit and full-width tails before selecting a scaling member. They are unrun here and have no predicted scores. Rough scaling laws follow a selected promising member and actual resource measurements. GitHub delivery is confirmed through the shared origin reference; AWS admission/results remain unobserved.
