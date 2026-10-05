# Small-first development of the integrated token model

User direction, 5 October: keep working toward a model that scales and wins;
reconsider old choices, use small runs to debug learning and capacity first.
CPU only. No new dense baseline training. Public validation is reserved.

## Current construction

`token_episodes.py` retains hard races and computational delays, sparse writes,
persistent deep memories, separate keys/values, temporal transport and the
zero-forward local counterfactual value teacher. Token IDs address the content
projection directly, exactly replacing one-hot multiplication. Numeric state
persists across chunks; credit truncation is explicit. EOS clears its own lane
before its observed token is consumed. Stochastic chunking uses a persistent
generator. Current candidate scoring and proposals are all-candidate work.

`token_readout.py` supplies an exactly normalized adaptive vocabulary head.
Its frequency ordering uses only the run's training interval. Adaptive head
cost, core cost, backward and optimizer work all belong in the resource bill.

## First failure and controlled repair

The character driver's content initialization scales with vocabulary size.
At GPT-2 vocabulary size its token-dependent variance is about 1/(3*50257),
while the source embedding starts with unit variance. This is an avoidable
input-path imbalance, not a model-family limit. The v2 balanced-input arm
initializes content at std .1, source bias at zero and contextual gate weights
at zero/bias -2. All temporal/routing/learning mechanisms remain. This changes
initial information balance and contextual exposure, not the inference graph
or asymptotic work. Compare to the saved identical-budget legacy-input arm;
separate the initialization components if the bundle improves learning.

## Admission ladder

1. Numerical contracts: one-hot/lookup logits and all gradients; stochastic
   partition invariance; numeric-state persistence and detach; lane-local EOS;
   normalized output likelihood; exact model/optimizer/state/RNG continuation.
2. Seconds-long real-token smoke, then a bounded small-corpus fit. Log train/dev
   curves, hard write shares/effective receiver counts, probability entropy,
   private receiver gradients and memory effective rank.
3. Tiny-corpus memorization: establish the model can fit its own training data.
   Diagnose input paths, readout bottlenecks and credit before widening pools.
4. Capacity interventions at fixed data/presentations: input balance, carried
   vs reset state, pool growth with tied/untied maps, explicit loser/future-write
   credit and credit horizon. Uniform activity alone is not useful capacity.
   A change advances only on the relevant quality/work or diagnosis endpoint.
5. CPU kernel throughput and memory smoke at the winning small-fit shape;
   then progressive data scales with checkpoints, before a benchmark-sized fit.

Grokking is measured on held-out modular-addition pairs with a fixed split;
reuse completed curves first. Track training mastery and delayed held-out
thresholds independently. Natural language loss improvement is not itself
proof of grokking. Existing tied/untied pool-2/pool-8 fits reach >99% held-out
accuracy; use their generalization timing to motivate new integrated tests.

## Data and public reference

Pinned FineWeb classic GPT-2 train shard 000001 and validation shard 000000
are local under ignored `data/fineweb_gpt2`. Development uses validation token
positions >=20,971,520, disjoint from the public first 10,485,760 targets.
No pilot score is compared to the published full validation number.
The historical January 2025 modded-nanoGPT source/log is preserved in
`references/`; its reported 3.2774 NLL is a public quality target, not a
matched-compute claim or a measured LSTM crossover on this exact corpus.
Audit its long-context/masking policy before final scoring.

## Reproducibility and resource rules

Each job has a unique one-job queue, uses run_safe.sh, one CPU thread, an RSS
watchdog and an 8 GiB available-memory floor. Checkpoint restores model, AdamW,
data cursor, numeric state, random generators and accumulated curve. Completed
results include source/data hashes, presentations, full measured fitting wall
and RSS. FLOPs are pending an operator-complete trace; no projected win is
reported as measured. Preserve producer versions before changing their source.

## Completed first fits

Single seed 6; GPT-2/FineWeb classic. 8,192 training tokens, 16,368
presentations, separate 2,048-token development sample; no public validation.

| Integrated variant | Initial dev NLL | Best dev NLL | Final train probe NLL |
|---|---:|---:|---:|
| Legacy input/adaptive readout | 11.2723 | 9.3313 | 6.4615 |
| Balanced input/adaptive readout | 11.2453 | 9.3949 | 5.6405 |
| Balanced input/frequency-initialized adaptive readout | 8.3403 | **8.2376** | 5.5456 |

The frequency-initialized readout wins this development comparison: **8.2376
vs 9.3313 NLL**, same token presentations. It also improves on its own
train-only frequency starting point by .1027 NLL. Balanced input alone improves
fitting but loses this development comparison by .0636 NLL. Train/dev divergence
starts after the useful early stage; enlarge the development/training sample
and check contextual learning before investing in a long fit.

The 512-token dense-readout memorization fit reaches **.02436 train NLL** from
11.1399 after 256 updates. This is an integrated learnability contract; its
held-out loss increases, as expected for a memorization probe. Dense head
measured 222 tokens/s; adaptive head measured 516–517 tokens/s for these shapes.
Peak RSS is 385–435 MB. Whole-fit FLOPs have not yet been traced.

Six eager numerical contracts pass, including exact next optimizer update
following state/model/Adam/RNG checkpoint restoration. The first compiled
contract failed because this image lacked Python C headers. Preserve its
failed queue/log; install the missing development headers and retry under a
new job name. This is a compiler environment failure, not model evidence.

Compiled-core parity passes after the header repair. The identical compiled
pilot has bestdev8.23759864 versus eager8.23759855; measured full fitting
throughput **838 vs 516 tokens/s (1.62×)**. Cold compilation costs69.6s in the
contract and is preserved separately. Next admitted small fit:
`curie_token_64k_20261005_v5.txt`:64K training tokens,64 fitting lanes,8 eval
lanes,128 updates. The watchdog stopped it when host MemAvailable fell to8154MB below8192MB; no quality result exists. Its failed runner log is retained. No larger
benchmark fit is admitted yet.


## Sparse capacity integration

Concrete engineering failure: all-candidate value proposals make pool growth
increase fitting work in proportion to dormant capacity. Reuse the established
§416 winner-plus-one-alternative kernel, with direct token lookup and persistent
state. This retains race clocks, sparse writes, temporal transport, separate
keys/values and an unbiased sampled estimate of the local value teacher. It
removes unnecessary losing proposal work, not available receivers. All keys
are still scored; optimizer work and the initial cache refresh are charged.
It does not supply full future-write credit.

` sparse_token_episodes.py ` rebuilds differentiable key caches from current
parameters at every chunk, then refreshes only the winning key within that
chunk. EOS resets the appropriate cache lane too. No cached parameter
projection is carried across Adam updates. The factual and alternative RNGs
are independent and persist; evaluation does not consume the training
alternative RNG.

Three new contracts pass: full-proposal vs sparse features/writes/all gradients
with persistent state after parameter changes; sampled credit changes gradients
without changing factual state; chunk partition preserves both generators and
state. The eight-receiver smoke evaluates two proposals per race, uses all
receivers, effective counts7.68–7.87/8 and memory rank9.96–11.49/16.

Fixed-budget pool4 sampled fit bestdev **8.2393** vs full local teacher8.2376;
eager450tokens/s vs full517tokens/s. The small-pool sampled path loses CPU
throughput here; compare the large-pool case and compiled execution before
claiming a speed advantage. Its value is constant proposal count during
capacity growth. Next admitted small fit is pool32 with shared input/output/
gate/control maps, private keys, key-read maps, clocks and memories, same
8K data/128-update budget. A capacity-use score alone is not a quality win.

## Current parameterized entry point —5 October18:40 UTC

Use `token_language.py` for new recipes, through unique run_safe queues.
It includes corrected initial/trained checkpoint selection and delegates to
`token_language_engine.py`. Select local credit (`--future-site first
--future-every 0`), first-site actual suffix credit, or uniform event-site
credit. Select suffix scoring count and adaptive minimum tail width explicitly.
Frequency initialization and balanced inputs are supporting choices in the
prepared queues; they are not model-family restrictions. Historical drivers
and source-pinned pending AWS queues remain preserved.

Default-capacity learning matches the completed event driver exactly. Wide-tail
normalization, prior and available contextual-rank contracts pass; full-width
actual resume passes. P16/H2 has32features; old tail projections16/8/4 are a
selectable readout bottleneck. Full-width32quality is queued on AWS, unmeasured.
No improvement is inferred from parameter count or the rank contract alone.

Completed later studies and work:
`TOKEN_2K_CREDIT_FINDINGS_20261005.md`; uniform-site repair and its0.067745NLL
trained gain are in `theory/TOKEN_EVENT_CREDIT_COVERAGE_20261005.md`.
All2Ktrained variants lose to initialization. Complete synthetic work audits
and their scope are in `theory/TOKEN_INTEGRATED_WORK_FINDINGS_20261005.md`.
The completed pool32tied-map arm gives8.267295 versus pool4 8.239298, a0.027997
loss at the same8Ktarget budget. More capacity by itself did not improve it.
The report's new tokenized appendix derives tables from completed files only.
