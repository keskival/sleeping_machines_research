# Decoder capacity is a selectable coordinate

The current P16/H2 member exposes32features. Adaptive decoding divides tail
projection width by2per cluster, reaching4features for the final vocabulary
tail. Its contextual logit variation there has rank at most4 before the shared
cluster probability is added. This is an imposed decoder bottleneck, not a
limit of persistent event representations. Overfitting at2K does not show that
raising this rank will generalize; nevertheless it must be selectable before
claiming the chosen model maximally uses its representational capacity.

Keep the frequency-initialized, exactly normalized head/cluster decomposition
and choose a minimum tail projection width up to the full input width. Existing
low-rank projections stay the default control. Wider tails retain every core
mechanism and only change supporting output capacity. They add learned weights,
selected tail matrix work, backward, optimizer and memory cost. They do not
change observed information, causal state, race clocks or route credit.

Required contracts: exact initial frequency distribution, probability
normalization, declared tail widths and gradients into formerly excluded
feature directions. Select width by completed quality/work evidence, not by
allocated parameter count alone. No baseline retraining or long fit is required
to verify those contracts. The parameterized engine should retain local,
first-site and all-event credit choices and correct initialization selection;
historical drivers and queued sources remain preserved.

## Completed contracts and selected comparison

`capacity_token_readout.py` supplies selectable tail widths without changing
the event core. Guarded contracts completed18:36:19 UTC:3passed in1.45s.
The default factory preserves every parameter and RNG state. Full-width tails
preserve the exact initialized token prior and normalization. A learned-point
rank witness expands the last41-token fixture tail from2to11independent
contextual directions. This verifies available capacity, not learned quality.

`token_language.py` is the parameterized selected-checkpoint entry point for
new recipes; `token_language_engine.py` supports the completed credit choices,
sample counts and tail minimum. Actual default learning parity passed18:36:45
against the completed uniform-site driver. Actual full-width resume passed
18:38:18, matching optimizer/state/all RNGs and quality curves exactly across
the interruption. Completed diagnostics/results are preserved.

Immutable AWS packet `addenda/aws_capacity_tokens_20261005T184000Z.json` prepares
a full-width32tail8K fit after the matched uniform-site default-tail control.
Same core, seed, data and128-update budget; source/data pins and1000000KiB RSS
cap are explicit. No wider-tail quality is claimed before that result. Historic
and already queued producers remain intact.
