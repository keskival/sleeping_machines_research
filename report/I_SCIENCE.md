# Sleeping Machines — Part I: The science

A trainable computing substrate that computes through time

Tero Keski-Valkama and Karoliina Salminen · Research report, Part I · 6 October 2026

This is the reader-facing report: the ambition, the model family and its theory, what the evidence shows on each front, and the experiments that decide the next step. Two companion parts hold the rest. [Part II — Methods and machinery](II_METHODS.md) collects protocols, metric definitions, compute accounting, numerical contracts and implementation checks. [Part III — Experiment record](III_RECORD.md) preserves every experiment entry and the earlier narrative chapters verbatim. `REPORT.md` remains the complete combined record; Parts II and III are generated from it by `report/split_report.py`.

## 1. The ambition

Sleeping Machines aims to make intelligence composable within **one trainable computing substrate**. Language and reasoning, multimodal world models, embodied action, event-native analytics over interleaved process streams, and typed tables become different forms of experience available to the same learner. The ambition extends through continual and on-device learning, learned communication and self-designing models to the hardware that executes them: globally clockless, memory-local event hardware that learns on chip.

An observation enters through an interface that respects its meaning: a token, a sensory event, a typed comparison or an action. It becomes an addressed message that interacts with persistent memory. Learned delays and temporal races decide which evidence meets and which computation happens. Small messages carry new evidence into deeper state; separate keys and values distinguish where evidence goes from what it says. Counterfactual credit teaches the routes and writes that could have happened. Available memory and skill can grow beyond the activity recruited for any one observation.

Each benchmark below is evidence for one front of this program. None of them defines the project.

## 2. The model family

Every message carries **content, an arrival time and an address**. Receivers hold persistent memories that decay and rotate with the real time elapsed between events. Candidate routes **race through learned delays**: the winner updates state and sends the next message, and the routes that did not win still receive **counterfactual credit**. Arrival order decides which memories meet and which computation happens, so delays perform computation rather than label it. Dense synchronous layers are a special case; one model can compute densely where a task needs it and selectively elsewhere.

![Position among model families](figures/architecture_landscape.svg)

The four architectural bets, and the mechanisms each one keeps explicit:

| Bet | Mechanisms | Why it matters |
| --- | --- | --- |
| **Time performs computation** | Learned delays, temporal races including race attention, decay and rotation by elapsed time | Selection and combination happen in the timing itself; waiting is part of the computation |
| **Hard routes learn through counterfactual credit** | Winner-only forward execution; credit to unrealized alternatives and optional routes | Discrete sparse routing is a long-standing training problem; crediting alternatives trains it without dense inference |
| **Deep persistent event representations** | Sparse addressed state updates; small messages mixing incoming content with persistent memory; separate keys and values | Memory is addressed and retained, not a buffer rescanned at every step |
| **Capacity beyond activity** | Large receiver pools, few selected writes per event | Stored skill can grow faster than the work spent per observation |

Supervision is silence-aware where the objective depends on waiting: the absence of an event until a real clock deadline is an observation, not missing data. Work is accounted separately for available capacity, scored keys, selected state updates, value deliveries and counterfactual learning work, so a dormant receiver is never reported as free to score or train. Part II gives the accounting conventions.

## 3. The theory

### 3.1 Foundational principles

The foundational calculus ([THEORY.md](../experiments/THEORY.md), §§1–25) reduces to five principles. Later notes extend each.

| | Principle | Consequence |
| --- | --- | --- |
| P1 | **Inference is tropical; unrealized futures are its dequantization.** A race computes minimums over event times (min-plus); at temperature σ the possible histories form a recombining forest weighted by e^(−cost/σ). | Near-miss weighting is the derivative of the soft minimum; credit is conserved at each collapse; holistic backpropagation is inside–outside. |
| P2 | **Weaving closes the past.** Collapses are stopping times; later inputs cannot affect them. | When to commit is an optimal-stopping problem; woven and unwoven counterfactuals differ. |
| P3 | **A race neuron is a weighted mean in time.** Within a piece, T = θ/ρ + Σ uᵢtᵢ. | Exact time-shift equivariance; urgency (ρ) and evidence (u) separate; learning must recruit absent evidence additively. |
| P4 | **Credit must reach what did not happen.** Along the realized history credit reaches only nodes that fired; the blind spot compounds with depth. | Counterfactual credit matters from depth 2; routing networks share the same boundary term. |
| P5 | **Thresholds are prices.** Homeostasis is the dual update of a capacity constraint; a race layer with homeostasis is an online entropic optimal-transport solver. | Target rates are capacities; log-ratio updates balance faster. |

### 3.2 Temporal algebra and attention

**An exponential race selects route i with probability softmax(score)ᵢ exactly, and delay-coded aggregation reproduces softmax attention exactly over the delivered keys** (theory notes [08](../experiments/theory/08_vector_memory_and_deep_stacks.md) and [151](../experiments/theory/151_normalized_clock_noise_and_precision_credit.md)). The family therefore contains a Transformer-capable function class. **Selective state-space models (Mamba) are also exact members:** each token is an event delivered after a learned, content-dependent delay, memories decay over the elapsed time, and writes and reads are content-addressed; this reproduces Mamba-1 and Mamba-2 selective scans to 4e−13, and rotating memories extend it ([note](../experiments/theory/MAMBA_CONTAINMENT_20261007.md)). A winner-only race delivers the sampled winner's value instead of the weighted average and skips the value-aggregation half of attention arithmetic. One winner is a sample, not the average: averaging m independent winners has mean-square error variance/m.

With width d, depth L, N historical keys, feed-forward expansion r and shared projection work B = (8 + 4r)d² per layer, S extra evolving-state work per layer, P updated parameters and U targets per Adam update (two FLOPs per multiply-add):

| Per token / target | Transformer | Race substitution |
| --- | --- | --- |
| Inference | L(B + 4Nd) | L(B + 2Nd + S) |
| Training, approximate | 3LB + 12LNd + 19P/U | 3L(B + S) + 10LNd + 20P/U |
| Logical value reads (FP32) | 4LNd bytes | 4Ld bytes |
| Logical key + value reads | 8LNd bytes | 4L(N + 1)d bytes |

The race training term charges admitted losing-value credit; it is not winner-only training. At fixed width the attention-only arithmetic limit is about 2× at inference and 1.2× during counterfactual training, and total key/value access improves by at most about 2×. Larger gains require richer temporal computation to reach the same quality at smaller width αd and depth βL: projection work scales by βα² and context work by βα. These are logical counts, not measured off-chip traffic or joules.

### 3.3 Keys, values, credit and depth

- **Key/value separation** ([note 22](../experiments/theory/22_key_value_separation_and_race_boundaries.md), §§194–196) isolates harmful winner changes, preserves actual routing during smooth value learning, and derives joint key/value/clock counterfactual policy credit with its reachability requirements.
- **Statistic-valued race memory** ([note 59](../experiments/theory/59_statistic_valued_race_memory.md), §§382–392): learned keys with sufficient-statistic values give zero-variance counterfactual route credit and closed-form write credit. Hierarchical count backoff is an exact cascade of escape races ([note 58](../experiments/theory/58_sufficient_statistic_state_and_count_references.md)).
- **Depth** ([notes 04, 08c](../experiments/theory/08c_sparse_attention_and_depth_bounds.md)): residual depth and temporal gauges, sparse-attention depth bounds, and the conditional depth bounds of causal context bridges. Counterfactual credit matters from depth 2 because the realized-history blind spot compounds with depth (P4).
- **Compute allocation**: the useful quantity is prediction quality at a fully counted budget of fitting, inference and learning work, including candidate discovery and losing-value credit. Capacity, activity and teaching work are separate axes (§§299–302, [note 46](../experiments/theory/46_integrated_sparse_temporal_language.md)).
- **Typed semantics** ([TYPED_SEMANTICS_AND_RACE_COMPOSITION](../experiments/theory/TYPED_SEMANTICS_AND_RACE_COMPOSITION_20261006.md)): type-respecting comparisons (numeric, ordinal, categorical, missingness) enter before neural mixing, with the equivariance conditions learning must satisfy. 39 mathematical/interface contracts pass.

## 4. Evidence, front by front

Conventions for every table in this section: single seed unless a seed count is stated; native fitting work is traced (and extrapolated from traced windows for long fits), dense references use shape estimates; "train compute" counts the complete fit including counterfactual credit and optimizer work. Every number comes from a completed result file; pending cells say pending.

### 4.0 Public benchmark: temporal point processes (EasyTPP)

Marked event streams with continuous times, on the official EasyTPP splits and the per-event log-likelihood protocol of the published state of the art (S2P2, NeurIPS 2025; [B1 dossier](../experiments/B1_EASYTPP.md)). Our model is a race of delayed clocks over persistent temporal memory: after each event, exponential clocks and delayed clocks that may stay silent race, and the first to fire sets the next event's time and mark. The likelihood is exact; losing clocks are credited through the survival term. Clocks may not resolve time below the data's recording resolution. Five seeds; TEST scored once per seed at the best DEV checkpoint.

| Taxi (nats/event, higher is better) | Total | Time | Mark | Parameters | Inference MACs/event |
| --- | --- | --- | --- | --- | --- |
| Best published (S2P2) | 0.522 ± 0.004 | 0.733 | −0.211 | 251,850 | 249,856 |
| **Ours, single model** | **0.5250 ± 0.0010** | 0.733 | **−0.208** | **20,504** | **20,708** |
| **Ours, 5-seed predictive mixture** | **0.5363** | **0.740** | **−0.204** | 102,520 | 103,540 |

| Taobao (nats/event) | Total | Time | Mark | Parameters | Inference MACs/event |
| --- | --- | --- | --- | --- | --- |
| Best published (IFTPP) | 1.318 ± 0.017 | 2.708 | −1.391 | — | — |
| S2P2 | 1.304 ± 0.039 | 2.719 | −1.415 | 26,801 | 26,016 |
| **Ours, single model** | **1.3991 ± 0.0025** | **2.753** | **−1.353** | **24,046** | **24,362** |

| StackOverflow (nats/event) | Total | Time | Mark | Parameters | Inference MACs/event |
| --- | --- | --- | --- | --- | --- |
| Best published (S2P2) | −2.163 ± 0.009 | −0.641 | −1.521 | 30,166 | 29,216 |
| **Ours, single model (with continuous-time state clock)** | **−2.1444 ± 0.0037** | −0.643 | **−1.501** | 36,183 | 36,732 |
| **Ours, matched size (2 layers)** | **−2.1525 ± 0.0045** | −0.646 | **−1.506** | **29,191** | **29,660** |

| Amazon (nats/event) | Total | Time | Mark | Parameters | Inference MACs/event |
| --- | --- | --- | --- | --- | --- |
| Best published (S2P2) | 0.781 ± 0.011 | 2.652 | −1.871 | 127,056 | 125,568 |
| **Ours, single model (anchored delay windows)** | **0.8028 ± 0.0007** | **2.657** | **−1.854** | **35,000** | **35,936** |

| Retweet (nats/event) | Total | Time | Mark | Parameters | Inference MACs/event |
| --- | --- | --- | --- | --- | --- |
| Best published (NHP) | −6.348 | −5.584 | −0.764 | — | — |
| S2P2 | −6.365 | — | — | 298,627 | 297,600 |
| **Ours, single model (every clock held below the 1 s recording cell)** | **−6.3262 ± 0.0009** | −5.558 | −0.768 | **19,654** | **19,850** |

**Win on StackOverflow, confirmed over 5 seeds:** +0.019 nats per event over the best published model, with the best mark log-likelihood and time level with the best, at 1.26× S2P2's per-event work; **a matched-size model (fewer parameters than S2P2, 1.015× its per-event work) also wins, with all five seeds above S2P2**. The gain comes from a state clock: a persistent memory that keeps evolving through the silent interval drives part of the hazard.

**Win on Taobao, confirmed over 5 seeds:** +0.081 nats per event over the best published model, with the best time and mark log-likelihoods, at 0.92× S2P2's per-event work.

**Win on Amazon, confirmed over 5 seeds:** +0.022 nats per event over S2P2, every seed ahead (0.8021 to 0.8036), with the best published time and mark log-likelihoods, at 0.29× S2P2's per-event work and 3.6× fewer parameters, one run per seed and no restarts. The earlier protocol (0.784 ± 0.027 with three restarts per seed) had one seed in a low optimization basin; free delay windows drifted beyond every observed gap, and anchoring each window to its gap component removed the basin. EasyTPP's Monte Carlo estimator gives 0.8037 to 0.8060. Amazon's times have no recording grid (281K distinct gaps, no zero gaps).

**Win on Retweet, confirmed over 5 seeds:** +0.022 nats per event over the best published model (NHP) and +0.039 over S2P2, every seed ahead (−6.3250 to −6.3272), with the best time log-likelihood, at 1/15 of S2P2's parameters and per-event work. Retweet times are recorded in whole seconds; every clock holds its one-cell hazard below the cell, and the recording-grid audit moves the score by 0.004, a fifth of the margin. Under EasyTPP's own Monte Carlo estimator the seeds score −6.3252 to −6.3277.

**Win on Taxi, confirmed over 5 seeds:** the 5-seed mixture beats every published model on total, time (best published 0.735) and mark log-likelihood at 0.41× S2P2's per-event inference work; a single model leads S2P2 on the mean at 1/12 of its parameters and per-event work. The earlier StackOverflow model (v5, without the state clock) was 0.018 behind S2P2 (−2.181 vs −2.163), its 5-seed mixture ahead (−2.154) at 4.6× compute; the state-clock model above supersedes it.  The Taxi win reproduces on independent hardware (curie desktop CPU, separately downloaded data, frozen driver, five seeds): 0.5252 ± 0.0007 vs 0.5250 ± 0.0010 on AWS, every seed above S2P2's mean. All winning results pass a recording-grid audit (scores unchanged when event times are dequantized within their recording resolution; Retweet within 0.004 nats). S2P2 parameters and MACs are counted from its released layer definitions at its published configuration.

### 4.0b Public benchmark: irregular clinical time series (P19 sepsis)

PhysioNet 2019 sepsis prediction on the five official Raindrop splits (38,803 ICU stays, 34 irregularly sampled channels, 4.2% positive). Our event-native classifier treats each time step as an event; addressed channel memories carry sufficient statistics (count, mean, min, max, first, last, trend, staleness) and decay with elapsed time; values enter through typed comparisons; silence (time since a channel was last measured) is an input ([B2 dossier](../experiments/B2_IRREGULAR_TS.md)).

| P19, 5 official splits | TEST AUROC | TEST AUPRC |
| --- | --- | --- |
| Best published (MTM, 2025) | 0.903 ± 0.020 | 0.583 ± 0.053 |
| **Ours (62,681 parameters; MTM reports 873K for its PAM configuration)** | **0.916 ± 0.022** | **0.639 ± 0.039** |

**Win on P19 over the best published model**, with AUPRC — the clinically relevant metric at 4% prevalence — ahead by more than either model's split spread. **Where the margin comes from:** gradient-boosted trees on the same per-channel statistics score 0.914 ± 0.020 / 0.618 ± 0.041 on the same splits (reverse-ablation diagnostic, 7 Oct), so most of the gain over MTM is the statistic-valued channel memory; our full model is level with those trees on AUROC (+0.002) and ahead on AUPRC (+0.021), within split spread. On P12 (in-hospital mortality, five official splits) the same configuration reaches TEST AUROC 0.871 ± 0.011 and AUPRC 0.585: behind MTM on AUROC (0.880), level on AUPRC (0.586), ahead of every other published model. PAM is in development.

### 4.0c One family across data types

The same construction — persistent memories that decay and rotate with elapsed time, addressed memory written sparsely, typed comparisons, silence as information, and temporal races — wins public benchmarks of different kinds. The P19 classifier imports the very temporal memory layer of the EasyTPP model, at the same size (32 wide, 16 modes, 2 layers).

| Data type | Benchmark | Result |
| --- | --- | --- |
| Generative event streams (when and what happens next) | EasyTPP, 5 datasets | **Wins on all 5** (Taxi, Taobao, StackOverflow, Retweet, Amazon) |
| Irregular clinical records (classification) | P19 sepsis, official splits | **Win**, AUPRC 0.639 vs 0.583 |
| Anonymous interleaved process logs | FAS v1, 3 seeds | **Win** vs six generic controls, 0.592 vs 0.559 AUROC |
| Character language | text8, 10M characters | **Win** vs tuned Transformers at ≤ equal compute; **loss** to tuned LSTMs; **loss** at 90M |
| Temporal reasoning from few examples | synthetic, 5 runs | 99.7% vs Transformers 33–41% |
| Mixed-type tables | synthetic; banknote | learns the synthetic interaction; **trees lead** on banknote |

**One shared core across four event domains (G2, development data, one seed).** A single temporal memory (14,080 parameters) trained on Taxi, Taobao, StackOverflow and Amazon at once, with dataset-specific mark embeddings and clock heads (68,910 parameters in total, against about 111,000 for four separate models), matches or beats the separate models on three of four: Taxi DEV 0.4899 vs 0.4872, Taobao 1.2895 vs 1.284, StackOverflow −2.186 vs −2.177 (same design without the state clock). Amazon trails (0.683 vs 0.69–0.77 across seeds) and was still improving at the last epoch, while the other three peaked by epoch 40; per-dataset schedules in the joint run are the next test. Mobility, shopping and Q&A timing share one learned temporal dynamics.

**Positioning.** The EasyTPP leader S2P2 (NeurIPS 2025) is demonstrated on one task type, event likelihood. Its parent family, deep state-space models, is general on dense sequences and language. Our claim is specific: one family, with one shared core, wins public benchmarks on both generative event modelling and sparse clinical classification, with further evidence on process logs, language and reasoning. The tests that make this a transfer claim — self-supervised event pretraining for clinical prediction, one model across five event datasets — are track G ([plan](../experiments/GENERALITY_PLAN.md)).

### 4.1 Learned temporal computation (synthetic, multi-run)

| Task | Ours | Reference | Verdict |
| --- | --- | --- | --- |
| Depth-3 event chains, 2,000 examples seen once (5 runs) | **99.73–99.93%** | Transformers 33.25–40.80% (same distinct examples, repeated fitting) | **Win** |
| Race retrieval at 4× training context (5 runs) | **100%** within 4,000 examples | Best of seven Transformer configurations, lower | **Win** |
| Unseen mod-17 triples via a learned phase rule | **100%** of 3,440 | — | Solved |
| Timing-only discrimination (identical marks, addresses, order) | **95.31%** | Order-only control 50.00% (exact ceiling) | **Win** |
| 16-source order, shared rules with private memories | **75.39%**, 14,180 parameters | Private rules 44.14%, 157,940 parameters | **Win** (11.1× fewer parameters) |

![accomplishments](figures/accomplishments.png)

These are structured tasks with declared priors; they establish that the mechanisms learn order, timing and retrieval from few examples. Elapsed time is used as information: clearing persistent state drops the timing task to 50%.

### 4.2 Hard-route credit and capacity beyond activity (text8, 10M characters, one pass)

| Change | Test bpc | Cost |
| --- | --- | --- |
| Value-informed alternative credit, p32/D4 | 2.507 → **2.371** | +0.3% fitting work; forward unchanged |
| Same credit at depth 8 | 2.456 → **2.326** | |
| Double the receiver pool at 8 selected writes/char | 2.371 → **2.345** | 1.65× fitting work; inference arithmetic flat (0.163 → 0.164 MFLOPs/position) |

Counterfactual credit to unrealized alternatives works, at depth, for almost no extra work, and stored capacity improves quality without raising inference arithmetic.

### 4.3 Character language (text8)

Same test interval text8[95M:96M]; native and Transformers score reset T256 windows, LSTMs carry state; budgets are complete training compute.

| Data | Budget | Ours | Tuned LSTM | Tuned Transformer | Verdict |
| --- | --- | --- | --- | --- | --- |
| 10M | ≤ 352 TF | **1.888** (p96/d4, 6 passes) | 1.826 (LSTM-512) | 1.996 (TF128×4) | Win vs Transformers; **loss** vs LSTM |
| 10M | ≤ 107 TF | **1.955** (p64/d4, 4 passes) | 1.915 (LSTM-384) | 2.215 (TF128×4) | Win vs Transformers; **loss** vs LSTM |
| 10M | 888.8 TF (TF) | 1.888 at 352 TF (0.40×) and 0.18× inference | — | 1.908 (TF256×4, 4 passes) | **Win**: better quality at 0.40× training, 0.18× inference compute |
| 90M | C, ≈ 0.96 PF | 1.800 (p64/d4, 0.97 PF) | 1.729 (LSTM-512, 0.91 PF) | 1.780 (TF192×4, 0.95 PF) | **Loss** to both |
| 90M | D, ≈ 2.1 PF | 1.783 (p96/d4, 2.11 PF) | — | 1.704 (TF192×4, 2.07 PF); 1.811 (TF256×4, 2.00 PF) | **Win** vs TF256×4 at near-equal compute (1.06×); **loss** to TF192×4 |

![One-pass 10M native variants against saved dense controls (3 October): quality and whole-fit work in one unit](figures/current_native_language_status.png)

At 10M characters the native model beats tuned Transformers at equal or lower compute; tuned LSTMs lead. At 90M both dense families lead, and the gap to the best dense reference is wider than at 10M (0.07–0.08 bpc versus 0.04–0.06). The direction of that trend is the central problem for the language front.

### 4.4 Properly tokenized language (FineWeb, GPT-2 BPE)

Fixed P24 integrated model, 3.16M parameters, two passes; DEV is 2,040 next-token targets (8 lanes × 255) from the FineWeb validation shard; count references are fitted on the identical TRAIN prefix and scored on the identical targets ([result](../experiments/results/token_language/aws_token_ngram_reference_20261006T143000Z.json), `experiments/token_ngram_reference.py`).

| TRAIN tokens | Ours (DEV NLL, nats) | Kneser–Ney bigram | Kneser–Ney trigram | Add-one unigram |
| --- | --- | --- | --- | --- |
| 64K | **8.033** | 8.139 | 8.155 | 8.162 |
| 256K | 7.742 | **7.647** | 7.648 | 7.972 |
| 1M | 7.252 / 7.264 (seeds 6 / 7) | 7.251 | **7.217** | 7.950 |

![token n-gram calibration](figures/token_ngram_calibration_20261006.png)

**The native tokenized model currently performs at bigram level.** It beats the count models at 64K tokens, trails them at 256K, and ties the bigram at 1M while the trigram leads by 0.03–0.05 nats. The count models fit in seconds; the native 1M fit took 73 minutes at 480 training tokens/s on one CPU thread. Erasing persistent memory raises native loss by only 0.010 / 0.017 nats (seeds 6 / 7): memory is used, but carries little.

The 2,040-target DEV slice has a standard error of about 0.09 nats for an unpaired mean and is harder than average: on 65,528 targets from the same offset the bigram scores 6.584 at 1M rather than 7.251. Differences of 0.02 nats on this slice, such as the 256K time-scale pair (+0.018 on seed 6, −0.022 on seed 7), do not separate variants. Native models are next scored on the larger slice.

Two further completed results: credit window 64 lost to credit window 16 by 0.129 nats at 64K tokens (credit 16 retained), and slower memory evolution increased stored-content dependence in both seeds (payload erasure 0.043 / 0.032 versus 0.011 / 0.014) without a replicated quality gain.

### 4.5 Anonymous interleaved processes (FAS v1)

Early fault detection from anonymous interleaved event logs; test 2,000 clean and 2,000 faulty runs; references are six generic controls fitted on the same anonymous inputs.

| Native seed | AUROC at 256 events | AUROC at 512 events |
| --- | --- | --- |
| 6 | **0.5998** | **0.7422** |
| 7 | **0.5910** | **0.7360** |
| 8 | **0.5864** | **0.7329** |
| Best generic control | 0.5587 | 0.7272 |

**Replicated win: all three seeds beat the best generic control at both horizons**, by +0.028 to +0.041 at 256 events and +0.006 to +0.015 at 512. FIFO and timing-aware de-interleaving learn their routes from hidden training identities; they are oracle-assisted diagnostics and are excluded from the comparison.

**FAS v2 (B3, in development).** The confirmatory benchmark merges two production lines into one anonymous log with 2% of events dropped. The setting was chosen on validation data before any learned model saw it. At 512 events per line, the identity-aware oracle reaches AUROC 0.821 and the best anonymous classical detector 0.685: 0.136 of signal that only correct binding of events to processes can reach. Neural reference families are not trained (6 October direction). The sealed comparison is against the information-matched classical detectors, and the public release invites outside submissions.

The model being developed for it follows from the theory of interleaving:
- A merged log of timed processes is a superposition. Its next event is the earliest pending event of the hidden processes, which is a race.
- A race readout gives each memory slot its own clock and own-duration law, and scores the next event by the exact superposition likelihood.
- Writing each event to the slot whose race most likely produced it is a particle filter: it trains a lower bound on the likelihood marginalised over all interleavings, with no learned write credit (theory §§432–436).
- In a small integrated fit on synthetic interleaved processes, binding purity rose from chance (0.27) to 0.81 against a Bayes-greedy ceiling of 0.91.

### 4.6 Typed tables

| Task | Ours | Reference | Verdict |
| --- | --- | --- | --- |
| Synthetic mixed-type interaction, 64 FIT rows | **100%** on 256 DEV rows, 6,370 parameters | — | Mechanism learns (fixed predicates, one seed) |
| Banknote, reserved test, 3 seeds | 91.8% | Boosted trees 94.0%; logistic regression 94.7% | Behind; first-pass variant |

![banknote reserved test](figures/banknote_reserved_test.png)

### 4.7 Public benchmarks, speech and adaptation

Attempt levels follow [WIN_CRITERIA.md](../experiments/WIN_CRITERIA.md). A first-pass variant is an existing member with at most knob-level tuning and no task-specific design; its result describes that variant, not the family. These benchmarks are not current battles.

| Benchmark | Ours | Reference | Verdict |
| --- | --- | --- | --- |
| NeuroBench Mackey-Glass, τ 17, 30 repeats (sMAPE) | 14.84, 57.6 KB | LSTM 13.37 (490 KB), ESN 14.79 | Behind; first-pass variant; smallest footprint |
| NeuroBench primate reaching, 5 of 6 official sessions (R²) | 0.619 paired mean | tinyRSNN 0.643, bigRSNN 0.683; leaderboard 0.71 | Behind; first-pass variant |
| Spiking speech, 512 private held-speaker utterances | 79.69% | Published SHD official test: 94–96% (different partition) | Official comparison pending |
| Online adaptation to a new character stream | 3.191 → 3.096 bpc | — | Adapts during use |

### 4.8 Learning dynamics: grokking

On a harder modular task (train fraction 0.25), larger pools grok later (untied pool 2: 4,750 steps; pool 8: 8,000), and **asymmetric memory decay — private decay 10× shared — groks 1.7× sooner** (4,750 versus 8,250 steps for tied pool 8), as theory §427.3 predicted. Single seed.

## 5. What the evidence says

**What works.** The temporal mechanisms learn order, timing and retrieval from few examples where Transformers fail; time is used as information. Counterfactual route credit improves language at depth for 0.3% extra work. Stored capacity improves quality at flat inference arithmetic. On anonymous interleaved processes the native model beats every generic control across three seeds. At 10M characters it beats tuned Transformers at equal or lower compute.

**What does not work yet.** Three measured problems explain most of the losses.

1. **Memory carries little context in language.** Tokenized quality is at bigram level, and erasing persistent memory costs 0.01–0.02 nats. The model is mostly using the current token.
2. **The implemented route credit is nearly blind on long horizons.** An exact audit on FAS (one race forced to each alternative, all other noise shared) finds correlation −0.08 and 0.29 between the implemented credit and the true consequence of each routing choice, with 60–64% sign agreement. Most of a choice's effect lies after the next prediction, and with long memories beyond the training segment. The proposed transported write credit scored worse (0.18 on R8) and was not promoted. This is why weight decay, extra write bandwidth, larger tied pools and a longer credit window all failed to move the language gap: the signal that would teach binding barely exists.
3. **Throughput.** Training runs at 480–860 tokens/s on one CPU thread. A 100M-token fit takes days; GPT-2-scale data is out of reach for the current implementation. This is an implementation limit, not a property of the family.

The gap to dense models widening from 10M to 90M characters is consistent with problems 1 and 2: dense models convert additional data into context use, and ours does not yet.

## 6. The plan: owned battles where the mathematics favours us

A race of exponential clocks over memory that decays with elapsed time is a temporal point process. The first clock to finish gives the next event's type and time; the waiting time without events enters the likelihood as a survival term, which is our silence-aware supervision. Timed, irregular event data is therefore the family's home field, and it has public leaderboards. Each battle is an owned development project: study the task and the leading methods, design the model for the task within the family, iterate on development data with error analysis, then run the sealed protocol ([PRODUCT_ORDERS.md](../experiments/PRODUCT_ORDERS.md)).

| Battle | Benchmark | Pass criterion | What it settles |
| --- | --- | --- | --- |
| **B1 (lead)** | EasyTPP: Retweet, Taxi, StackOverflow, Amazon, Taobao | Best published log-likelihood on ≥ 2 of 5 datasets, type and time accuracy no worse; 5 seeds; measured inference work | **Taxi, Taobao, StackOverflow and Retweet won** (§4.0); pass criterion exceeded; Amazon continues |
| **B2** | Irregular clinical and sensor series: P12, P19, PAM (Raindrop protocol) | Best published AUROC / accuracy on official splits | **P19 won** (§4.0b); P12 and PAM in development |
| **B3** | FAS v2 sealed confirmation, then public release of FAS | As pre-registered: native seed mean ≥ best information-matched classical + 0.02 AUROC at 512 events per line, paired bootstrap lower bound > 0 | A confirmed event-native analytics win and a benchmark we define. Setting selected (oracle 0.821 vs classical 0.685); race-readout binding model in development |
| **R1** | Language research (one slot) | Solve associative recall/induction with irregular gaps; beat KN trigram on the large DEV slice | Whether persistent memory binds context; gates any language scaling |

High-fidelity route credit (exact forced-lane credit at small pools, then a low-variance multi-step estimator) is developed inside B1, B3 and R1, where model sizes make it affordable. Parallel-scan training of the linear decay/rotation core follows when a battle's fitting time requires it. Every result reports measured inference work beside quality. New Transformer and LSTM training is retired: comparisons use published scores under the exact matching protocol and the dense results already completed.

## 7. Where to read further

- [Part II — Methods and machinery](II_METHODS.md): metric definitions, evaluation protocols, compute and work accounting, numerical contracts, admission gates, replay drivers and the hardware cost model.
- [Part III — Experiment record](III_RECORD.md): every experiment entry, wins and losses, and the earlier narrative chapters as published.
- [Model family overview](model_family_overview.md) → [formal core](model_family_specification.md) → [design rationale](model_family_design.md) → [composition rules](model_family_composition.md); [visual atlas](architecture_atlas.html); [claims and evidence map](architecture_evidence.md).
- [THEORY.md](../experiments/THEORY.md) and its numbered notes; [FINDINGS.md](../experiments/FINDINGS.md) for the dated results log; [WIN_CRITERIA.md](../experiments/WIN_CRITERIA.md) for win definitions.
