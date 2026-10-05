# CPU language evidence: reuse open references, train the integrated model

5 October 2026 · User-directed · Research selection, no new training admitted.

The objective is a useful quality/work advantage against competent language Transformers, as evidence toward the general-purpose temporal/event substrate. Use published runs to choose a regime where Transformers are competitive; spend our training budget on the integrated model. CPU-only and properly tokenized language are the current execution constraints. Byte/character benchmarks remain historical mechanism evidence rather than the main language program. Do not substitute byte prediction for proper tokenization to accommodate current code: fix the token interface and learning implementation.

## Decision

Stop making discovery of an exact LSTM/Transformer crossover a new training campaign. There is no corpus-size threshold independent of model capacity, optimization, context and training work. Published comparisons already identify useful Transformer-leading regimes, including full text8 and WikiText-103. Our smaller text8 controls identify their own configuration ranking, not a universal property of the corpus.

**The smallest well-supported tokenized candidate found is WikiText-103: about 103M training word tokens.** It supplies historical recurrent/Transformer comparisons and released Transformer weights. **The modern GPT-style candidate is the FineWeb replication ecosystem**, with GPT-2 BPE data shards, logs and separate released reproduction checkpoints. Select a particular run, not the entire changing ecosystem. No claim that a 100M-token arbitrary FineWeb subset defeats a tuned LSTM is supported by this review.

Use the selected reference's exact tokenization. WikiText's historical word vocabulary and GPT-2 BPE are separate protocols: do not compare their perplexities. For a direct GPT-style comparison, retain GPT-2's existing tokenizer and exact source token stream. A new train-only tokenizer is unnecessary when an immutable reference tokenizer exists.

## Verified candidates

| Candidate | Data and existing evidence | Reuse | CPU decision |
| --- | --- | --- | --- |
| WikiText-103, historical word-level protocol | About 103M TRAIN tokens. Published Transformer-XL table: LSTM 48.7 PPL, LSTM+cache 40.8, QRNN 33.0, Transformer-XL standard 24.0 at 151M parameters. Different recipes; this is historical quality evidence, not equal-compute crossover evidence. | Official splits and vocabulary; authors' released Transformer-XL weights and evaluation code; fairseq additionally links a 247M adaptive-input Transformer checkpoint. | First candidate for the smallest literature-backed tokenized comparison. Freeze the exact word protocol and reference evaluation context. Large vocabulary needs an efficient normalized head and measured CPU smoke. |
| FineWeb classic / llm.c GPT-2 124M | Published full reproduction uses 10B tokens and context 1024. Training recipe and logs are public. | Pre-tokenized GPT-2 shards and published reproduction record; original GPT-2 weights are NOT weights trained on the identical open reproduction corpus. | Reference training is unnecessary. A full native 10B fit is deferred until CPU throughput supports it. A smaller native pilot is development, not a matched-data replication claim. |
| FineWeb / modded-NanoGPT | Public 3.28 validation-NLL target, fixed GPT-2 tokens and reproducible record logs. Individual records differ in data presentations, context, architecture and optimizer. | Pin a particular completed record; read actual training-token count, validation policy and complete resource boundary from its source/logs. | Best source to investigate a smaller GPT-style target. Latest records introduce huge sparse hashed tables and specialized hardware; do not assume the latest record is a small ordinary GPT or translate its GPU seconds into native CPU cost. Checkpoint availability is not verified. |
| FineWeb-Edu / bardiaegz GPT-2 124M | Publisher reports 7.29B pretraining tokens and validation loss 3.0771, with a released base checkpoint. This is FineWeb-Edu, not FineWeb classic. | Base weights, model definition and publisher recipe; API file listing confirms base and SFT files. | Secondary checkpoint candidate. Exact held-out split, preprocessing, presentation order and baseline work still need audit before admitting it. SFT weights excluded from a pretraining comparison. |
| TinyStories | Authors release small decoder-only models and synthetic stories. | Useful cheap tokenized implementation/learning pilot. | Not selected as strategic evidence: the reviewed sources do not establish the required strong-Transformer-versus-LSTM regime. Do not turn it into an endless side campaign. |
| Pythia | Released small models and ordered data, but each full run sees approximately 300B tokens. | Checkpoints are useful evaluation assets; early checkpoints require their precise seen-data prefix. | Small parameter count does not make the training corpus small. Full replication deferred. |

## Primary sources checked

- [Transformer-XL paper, tables 1–3 and dataset protocol](https://arxiv.org/html/1901.02860v3): historical recurrent/Transformer evidence and context conventions.
- [Transformer-XL official pretrained-model instructions](https://github.com/kimiyoung/transformer-xl/tree/master/tf): released weights and preprocessed vocabularies; downloads are linked, not fetched/validated here.
- [Salesforce WikiText dataset](https://huggingface.co/datasets/Salesforce/wikitext): raw and vocabulary-normalized variants; preserve the one used by the reference.
- [fairseq released language models](https://github.com/facebookresearch/fairseq/tree/main/examples/language_model): alternative pretrained Transformer and scoring conventions.
- [llm.c exact 124M run recipe](https://github.com/karpathy/llm.c/blob/master/scripts/run_gpt2_124M.sh): training configuration and token presentations.
- [llm.c data preprocessing](https://github.com/karpathy/llm.c/blob/master/dev/data/fineweb.py): distinguish classic/education data, tokenizer, EOS and shard split.
- [modded-NanoGPT records and rules](https://github.com/KellerJordan/modded-nanogpt): use a pinned record and its exact scoring policy.
- [pre-tokenized FineWeb shards](https://huggingface.co/datasets/kjj0/fineweb10B-gpt2): download only selected shards.
- [released FineWeb-Edu base checkpoint and recipe](https://huggingface.co/bardiaegz/gpt2-124m-fineweb): publisher-reported quality; not independently rescored.
- [TinyStories paper](https://arxiv.org/abs/2305.07759) and [Pythia repository](https://github.com/EleutherAI/pythia): pilot alternative and corpus-size exclusion respectively.

## CPU feasibility and the token interface

The completed native p64/D4/H2/U2 90M four-pass record reports 359,997,440 fitting characters, 7,978.64 training characters/s, 45,775.58 s total wall time (12.72 h), and 1,295,196 KiB peak RSS. At that unchanged throughput, 100M steps take 3.48 h, 1B take 34.82 h and 10B take 14.51 days, before new token/head/context overhead. These are scheduling extrapolations, not tokenized throughput measurements.

The current driver hardcodes 27-symbol one-hot content and a 27-way output head. Do not construct 50K-dimensional one-hot arrays for tokenized training. Gather the input projection's token column (equivalently an embedding lookup) to preserve its mathematical operation. A normalized larger-vocabulary head is supporting language infrastructure, not a replacement for temporal computation.

With the current 128-wide combined payload, a 50,257-way dense output projection alone requires approximately 6 × 128 × 50,257 = 38.6M FLOPs per training target for forward/backward matrix products, excluding normalization and optimizer work. The saved entire character fit is about 2.68M counted operations/target. Thus GPU Transformer timings cannot answer whether our tokenized CPU model is affordable. This proxy excludes input processing, all core work and protocol effects; trace the actual implementation before setting a fit budget.

A normalized hierarchical/adaptive output head is a candidate if dense output dominates. Record its failure target (vocabulary cost), retained core mechanisms, exact probability factorization, full inference/training work and a small integrated comparison before a long fit. Sampled training losses alone are not exact full-vocabulary evaluation. Do not remove races, selected writes, key/value separation, persistent memory or counterfactual credit to fit a budget.

## Bounded work that creates value

1. Select one published tokenized reference, first checking WikiText-103 and one moderate-size FineWeb record. Freeze source/data/tokenizer revisions, reference score, split, scored targets, history/reset policy, unique data and actual presentations. Record whether weights and exact scoring are available. No baseline retraining.
2. Implement efficient token IDs through the native input/output path and verify causal predictions, exact normalized probabilities, state resets, gradient/update behavior and 27-symbol equivalence where the interface is mathematically unchanged. First perform a short integrated fit and one CPU throughput/RSS smoke on a provisioned owner host through run_safe.sh.
3. Set a bounded native fit from measured throughput: one initial configuration, one diagnosis-driven alternative at most. Budget initial training in one host-day; report if the chosen benchmark requires more rather than silently reducing data and calling it a replication. Preserve checkpoints and milestone trajectories.
4. Train the selected native model on the reference data. Primary value: held-out quality plus complete native work against an existing strong reference. When a native fit uses less data than the reference, label the data difference explicitly; quality success is still useful, but not identical-data matched-compute evidence.
5. Reuse reference weights for inference-only scoring when it materially improves comparability. Published numbers can supply a literature comparison without rerunning training. Unknown reference fitting cost remains unknown; it does not block an accuracy result or a measured inference-work comparison.
6. Confirm a promising selected comparison with another native seed; only then spend on a larger point. Additional modern hybrid/MoE controls are expansion work, not prerequisites to the first scoped result.

Stop CPU baseline grids, duplicate public GPT training, speculative crossover brackets and benchmark expansion without an answer-changing target. Existing running jobs and completed evidence remain intact; owners implement priority changes at safe boundaries. CPU-only is an instruction, not a measured theorem that this model family cannot benefit from a GPU.

## Diagnose and repair scaling in the model family

Latest user direction: use properly tokenized language to find and fix the family's routing and learning bottlenecks, not merely populate a comparison table. The chosen published reference supplies an external quality target. Development trajectories and targeted integrated comparisons identify the internal failure.

Start with the current integrated D4/H2 event model; compare another family member only when it tests a specified information path or learning hypothesis. Dense/carrier-only members remain diagnostic controls. Fixing the tokenizer interface is prerequisite engineering; it is not the scaling contribution.

| Failure to measure | Instrumentation on development only | Bounded repair/comparison |
| --- | --- | --- |
| Receivers never become useful | Slot occupancy and exposure, winner entropy, state/read specialization, held-out loss as available pool grows at fixed writes | Tied/shared processing, recruitment or sampled credit, one change at a time; charge candidate scoring, all loser teaching and optimizer work |
| Routing selects poor content | Local alternative replay utility, teacher variance/direction, downstream loss after route intervention, clock precision | Compare the current teacher to one derived credit repair with the same factual inference and a capped teaching budget |
| Writes help now but harm future prediction | Credit horizon, state ablation by age, delayed downstream utility and truncation boundaries | One longer-credit or write-utility comparison at the same complete resource cap; do not infer useful memory from occupancy alone |
| Added depth consumes work without improving loss | Per-depth gradient support, route survival, intermediate readouts and marginal quality/work | Diagnose bypasses or failed credit before changing topology; verify retained mechanisms in a small integrated fit |
| State cannot access available context | Early/middle/late document loss, retrieval/state erasure, document resets, timing precision and time-shift contracts | Fix the measured access/retention path; retain equivalent available causal history in primary comparisons |
| Token prediction overhead consumes the CPU budget | Separate input lookup, output normalization, core, teacher and optimizer work plus RSS | Equivalent token gathers first; normalized adaptive/hierarchical head only after a documented cost failure and integrated comparison |

Record measurements on a small fixed TRAIN/development slice from the selected corpus during each bounded fit. Compare paired configurations where possible. Diagnose one concrete failure, derive the proposed repair, state which mechanisms and costs it changes, verify numerical contracts, run a small integrated comparison, then promote the successful recipe to more data/capacity. A failed repair remains evidence about that member and intervention.

Every admitted fit must answer a stated question and produce a reusable checkpoint, development loss trajectory, mechanism measurements and full work ledger. Select on development, not repeated public test checks. Save the previously successful recipe as the comparison. Reserve a larger data/budget point before selecting repairs so that we test actual scaling rather than more tuning of the same point. Confirm the selected improvement with a second native seed; use additional seeds when uncertainty changes the verdict.

The target is better prediction per complete learning/inference resource with useful capacity beyond active work. A faster carrier-only model or a dense language gain cannot stand in for that target. No arbitrary performance ceiling is inferred from a restricted variant.

## Relation to earlier protocols

This plan supersedes new discretionary baseline-training and GPU-admission requirements in LANGUAGE_CROSSOVER_PLAN.md and FRONTIER_COMPUTE_PROTOCOL.md. Their theoretical mechanisms, historical results and future expansion remain available. No active queue, source pin or frozen execution command is rewritten. The research register contains checked remote revisions; it is not an executed training manifest.

## Source-bound implementation audit

[LANGUAGE_IMPLEMENTATION_AUDIT_20261005.md](LANGUAGE_IMPLEMENTATION_AUDIT_20261005.md) identifies restrictions in the actual completed leading fit (all seven saved source hashes match): short state-reset segments, local delivery credit without alternative-write utility, dense loser proposal work, rare-route exposure, final checkpoint selection and fixed execution structure. It distinguishes demonstrated restrictions from unmeasured penalties and lists existing alternatives to reuse. Repair order: token interface, persistent fitting/selection, useful routing/credit, larger sparse-learning capacity; clock/noise/depth changes follow measured failures.

## Governing direction: implementation serves the objective

User direction: there can be no implementation restrictions; we make the implementation. Treat the audit findings as engineering work to complete. Proper tokenization, useful persistent memory and scalable routing/learning are requirements to implement, not optional capabilities vetoed by the current driver or backend. Choose an economical experiment and build the implementation it needs.
