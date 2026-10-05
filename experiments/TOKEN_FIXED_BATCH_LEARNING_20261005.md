# Fixed optimizer batch learns beyond initialization

GPT-2 FineWeb,2,048admitted training tokens,8,160fitting presentations,
1,024admitted development tokens and**1,016scored next-token targets**.
P16/D2/H2/U4; batch64per lane,16AdamWupdates, four passes, evaluation every4
updates. Credit16and64share all data, update count and selection cadence.
Seed6/7fits use the guarded one-job runner,700000KiB RSS cap and8GiB reserve.

| Seed | Credit | Initial dev NLL | Selected dev NLL | Learning gain |
|---|---:|---:|---:|---:|
|6|16|8.906910|8.812527|0.094383|
|6|64|8.906910|8.819470|0.087440|
|7|16|8.906910|8.813838|0.093072|
|7|64|8.906910|8.812853|0.094057|

**Both members beat their initialization in both seeds.** Credit16wins selected
quality at seed6by0.006943NLL; credit64wins at seed7by0.000985. The mean ranking
gap is below the credit64seed spread; retain both for the prepared8Kcomparison.
Credit64has better final development loss in both seeds. All selected artifacts
exist and initialization remains eligible. This establishes small-development
learning, not a public Transformer win or iso-FLOP advantage.

Relative to the earlier batch16recipe, changing to batch64also reduces updates
from64to16at the same8,160targets. That is a recipe improvement, not an isolated
horizon effect. It makes concrete progress on the learning setup without
declaring the earlier failed recipe a model-family ceiling.

## What part uses context?

Frozen selected models are evaluated with actual features versus the same
learned readout driven by a constant mean of2,040causal TRAINfeatures. No dev
targets fit the mean. Actual context improves NLL by0.005029/0.015265 for
seed6credit16/64, and0.011278/0.011398for seed7. All four context contributions
are positive. This constant-feature intervention is not an optimally refitted
unigram baseline and does not isolate persistent memory from current-token
features. Much of the gain over the original initialized distribution is
marginal recalibration; contextual capacity has a smaller positive contribution.
Evidence: `results/diagnostics/curie_token_context_utility_audit_20261005_v1.json`.

## Denominator correction, preserving original losses

Earlier notes/tables called the1,024-token development interval1,024targets.
`interval_tensor` admits128tokens to each of8lanes; evaluate scores127next-token
targets per lane. The true count is1,016. Similarly2,048-token dev intervals
score2,040targets;8,192-token intervals score8,184. All paired rows use the
same targets and their saved loss means are unchanged. Frequency counts use
all admitted TRAINtokens; passes score2,040targets each in this2Kstudy.
The reported train metric is a frozen first128-targets-per-lane probe, not
necessarily the entire training population. Public benchmark scoring must
load its required lookahead token per sequence and score the exact10,485,760
reference targets; it must not inherit this interval/target naming ambiguity.

## Reference protocol now audited

`references/MODDED_NANOGPT_CONTEXT_AUDIT_20261005.md` records the source-derived
reset/document/target/attention protocol and published-data provenance. Reuse
the final3.2774NLLlog; the record has no released checkpoint. Intermediate
scores follow changing attention and optimization schedules, so they are not
independently tuned small-budget baselines. No public validation was scored
in this work. Exact reference boundary precedes the eventual benchmark run.
