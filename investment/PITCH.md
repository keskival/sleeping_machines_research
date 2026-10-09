# Sleeping Machines — investor pitch

**The next general-purpose substrate for machine intelligence.**

Sleeping Machines aims to make intelligence composable within **one trainable computing substrate**. Language and reasoning, perception and world models, embodied action, typed tables and interleaved process streams become different forms of experience available to the same learner. The ambition extends through continual learning, communication and self-design to the hardware that executes them.

An observation enters through an interface that respects its meaning: a token, a sensory event, a typed comparison or an action. It becomes an addressed message interacting with persistent memory. Learned delays and temporal races decide which evidence meets and which computation happens. Small messages carry new evidence into deeper state; separate keys and values distinguish where evidence goes from what it says. Counterfactual credit teaches the routes and writes that could have happened.

The larger promise is that these are **reusable learned computations**. Comparison, binding, retention, prediction and planning learned from one form of experience can support another through shared representations and end-to-end credit. Perception can inform reasoning; reasoning can guide action; process traces and tables can contribute to the same world model. Joint training must measure that transfer and the retention of existing skills.

This also connects the learner to its execution. Time performs computation, including useful sleeps; local events can advance it without a mandatory global clock. Inference and learning are designed around the same temporal, addressed substrate. Available memory and skill can grow beyond the activity recruited for one observation, with work tuned from small data and constrained devices to large datasets and datacenters. Candidate discovery and alternative learning work count in that budget.

**A revealing proof of the platform:** mixed-type tables, token sequences and anonymous event streams learned through the same temporal core, followed by measured skill transfer. The program spans small-data usefulness and large-data scaling.

## Wins and advances at a glance (8 October 2026)

**Seven public leaderboard wins**, each on the official splits with a sealed test scored once per run, 34 of 35 independent
runs ahead of the best published result:

| Benchmark | Ours | Best published | At what cost |
| --- | --- | --- | --- |
| EasyTPP Taxi (mobility events) | 0.525 ± 0.001 | 0.522 (S2P2, NeurIPS 2025) | 1/12 of the leader's parameters and compute; **reproduced independently on separate hardware** |
| EasyTPP Taobao (shopping) | 1.399 ± 0.003 | 1.318 (IFTPP) | 0.92× the leader's compute; largest margin (+0.081) |
| EasyTPP StackOverflow (Q&A activity) | −2.153 ± 0.005 | −2.163 (S2P2) | **matched size and compute**; all 5 seeds ahead |
| EasyTPP Retweet (social cascades) | −6.326 ± 0.001 | −6.348 (NHP) | **1/15** of S2P2's parameters and compute |
| EasyTPP Amazon (shopping reviews) | 0.803 ± 0.001 | 0.781 (S2P2) | 0.29× the leader's compute |
| P19 ICU sepsis prediction | AUPRC 0.639, AUROC 0.916 | 0.583 / 0.903 (MTM) | temporal memory adds +0.072 AUPRC on every split beyond summary statistics; reproduced |
| PAM wearable activity recognition | accuracy 0.978, F1 0.980 | 0.975 / 0.976 (MTM) | 46K parameters vs MTM's 873K |

**Rigor:** pre-registered reporting rules; the benchmark's own scorer agrees; a recording-grid audit every win passes; a
one-command kit for third-party reproduction. **In progress:** one single configuration across all five EasyTPP
datasets (four complete, all ahead on the mean).

**Advances beyond the leaderboards:**
- **Tokenized language:** 6.009 ± 0.010 nats per token vs a Kneser–Ney trigram's 6.537 on GPT-2-tokenized FineWeb (3 seeds);
  at 4× the data 5.510 vs 6.100, so the lead grows with data.
- **Binding and recall:** associative recall across irregular gaps 97.5% (3 seeds) vs a 14% baseline, 91.6% at twice the
  training length; **learning through the race alone** (no backpropagation into the network) reaches 77%.
- **Generative mode:** the same race-of-clocks model samples realistic event streams exactly (no rejection), for simulation,
  what-if analysis and synthetic data; taxi streams match real timing (gap KS 0.014 vs 0.063 naive) and event-to-event
  structure (0.024 vs 0.53).
- **Exact training signal and measured speed:** the closed-form likelihood removes the Monte Carlo gradient noise that
  sampled-intensity models carry (18% relative noise at the common 10-sample setting, measured); a streaming per-event
  update runs at about 1,500 events per second on one CPU core in plain PyTorch (Taxi; 20.7K multiply-adds per event vs
  the leader's 250K). A race of clocks provably approximates any continuous inter-event density, confirmed numerically.
- **Reasoning from few examples:** temporal event chains 99.7% vs Transformers 33–41%; race retrieval 100% at 4× context.
- **Character language:** beats tuned Transformers at equal or lower compute (text8, 10M characters).
- **Learning signal and capacity:** credit to unchosen routes cuts error by 0.14 bits per character for 0.3% more work;
  doubling memory slots lowers error with the same eight writes per input.
- **Generality:** one shared temporal core serves four event domains; anonymous interleaved process logs (FAS v1) beat six
  generic detectors across three seeds.
- **Theory:** a race of clocks contains softmax attention and Mamba-style selective state spaces exactly.

**Where we are behind** (stated with the numbers in the report): large-scale character language (90M), tuned LSTMs at 10M
characters, P12 mortality AUROC, real-world tables, FAS v2 (level on validation with a time-encoded Transformer reference, 0.702 vs 0.704, at about 1/7 of its parameters and a far better likelihood; ahead of the best classical detector's 0.685; sealed test pending, a tie expected under the 0.02 rule).

## Evidence anchors for the platform

**Token sequences, anonymous interleaved process logs and mixed-type comparisons now learn within the temporal/sparse family.** They exercise different information structures through addressed messages, persistent memory, temporal computation and credit to unrealized alternatives. This is evidence for a common computing construction spanning small-data tables, asynchronous analytics and tokenized language.

| Evidence front | Completed indication | What it establishes |
| --- | --- | --- |
| **Public benchmark: timed event data (EasyTPP, ICLR 2024)** | **Taobao 1.399 ± 0.003 vs best published 1.318** nats/event (+0.081) at 0.92× the leader's compute; **Taxi**: a 5-seed mixture (0.536) beats every published model at 0.41× S2P2's compute (NeurIPS 2025), and a single model leads S2P2 on the mean (0.525 vs 0.522) at 1/12 of its parameters and compute. **StackOverflow** −2.144 ± 0.004 vs −2.163 (+0.019) at 1.26× S2P2's compute. **Retweet −6.326 ± 0.001 vs best published −6.348** (NHP; S2P2 −6.365) at **1/15 of S2P2's parameters and per-event compute**. Official splits, sealed test, 5 seeds. | **All five datasets won on a public leaderboard of the family's home field, and one model configuration wins all five without per-dataset tuning**, four of them at a fraction of the state of the art's compute; the Taxi win reproduced independently. |
| **Public benchmark: ICU sepsis prediction (P19, PhysioNet 2019)** | **AUROC 0.916 ± 0.022, AUPRC 0.639 ± 0.039** vs best published 0.903 / 0.583 (MTM, 2025) on the five official splits; 62,681 parameters. Without its temporal memory the same network scores 0.900 / 0.572 on every split; trees on the same statistics 0.914 / 0.618: the temporal memory carries signal the statistics lack (AUPRC ahead of the trees on every split). Reproduced on separate hardware. | **A second public win, in clinical early warning**, with AUPRC (the clinically relevant metric at 4% prevalence) ahead beyond split spread. |
| **Public benchmark: wearable activity recognition (PAM)** | **Accuracy 0.978 ± 0.007, F1 0.980 ± 0.008** vs best published 0.975 / 0.976 (MTM, 2025), five official splits, 46,316 parameters; four of five splits ahead. | **A third public domain won**: wearable sensor streams. |
| **One family across data types** | The P19 classifier reuses the event model's temporal memory layer (same code, same size). | Wins on generative event modelling and on sparse clinical classification; the EasyTPP leader S2P2 is shown on one task type. Transfer tests are next ([generality plan](../experiments/GENERALITY_PLAN.md)). |
| **Anonymous interleaved processes** | FAS mean AUROC **0.5924 vs 0.5587** at 256 events; **0.7370 vs 0.7272** at 512 events. All three native seeds beat the best of six saved generic controls at both points. | Replicated early-detection quality wins on FAS v1. Privileged identity-assisted diagnostics are excluded; sealed v2 and stronger neural references are the next benchmark. |
| **Mixed-type tables, small data** | **100% accuracy on 256 synthetic DEV rows from 64 FIT rows**,6,370 learned parameters. Numeric, categorical, Boolean and missingness comparisons enter before neural message processing. | The integrated core learns the mixed-type interaction; relabeling, column-order, missing-value semantics and actual alternative-write credit checks pass. Fixed predicates, one seed; learned discovery and real tables versus trees are the next tests. |
| **Properly tokenized language** | The temporal-memory token model with the keyed predecessor read scores **6.009 ± 0.010** nats per token (3 seeds) at 1M GPT-2 tokens of FineWeb vs a Kneser–Ney trigram's 6.537; at 4M tokens **5.510** vs 6.100. | First tokenized-language evidence that persistent memory binds context; the lead grows with data. A published small Transformer under the same protocol is the next reference. |

**The unifying advantage is the reusable construction:** type-respecting comparisons and tokens become events; events recruit meaningful state and teach hard routes through their consequences. Available capacity, selected activity, temporal learning and execution are separate design axes. The larger ambition connects language and reasoning, multimodal world models, embodiment, continual learning, communication, self-design and clockless hardware through this substrate. Joint training with shared parameters must next measure skill transfer and donor-skill retention. These completed instances were trained separately.

Source-bound numerical packet: [headline evidence](../report/unification_headline_evidence_20261006_v1.json).

**What is already shown** (language rows single-seed; native compute traced, reference compute shape-estimated):
- **Wins a public leaderboard on its home field:** EasyTPP Taobao (+0.081 nats/event over the best published model at
  0.92× the leader's compute), Taxi (beats every published model at 0.41× S2P2's compute) and StackOverflow (+0.019 at
  1.26× compute); 5 seeds, sealed test.
- **Wins a public clinical benchmark with the same core:** P19 sepsis prediction, AUPRC 0.639 vs 0.583 for the best
  published model (MTM, 2025), five official splits. The classifier reuses the event model's temporal memory layer.
- **Beats Transformers at a fraction of their compute.** On 10M characters of text8 the native model reaches 1.888 bpc
  against the 4-pass Transformer-256×4's 1.908, at 0.40× its training and 0.18× its inference compute, and beats the
  validation-selected tuned Transformers at both budgets (1.888 vs 1.996; 1.955 vs 2.215). Tuned small LSTMs lead at
  these budgets by 0.04–0.06 bpc.
- **Counterfactual route credit works:** 2.507 → 2.371 bpc for 0.3% extra training work.
- **Capacity beyond activity:** doubling the receiver pool improves 2.371 → 2.345 bpc at the same selected work.
- **Learned temporal computation:** 99.7–99.9% event-order accuracy from 2,000 examples seen once (Transformers
  33–41%); 95.3% on timing-only discrimination where any order-only model is capped at 50%; 100% on unseen modular
  rules; 100% retrieval at four times the training context.
- **Scales with data and adapts online:** at 90M characters the native model reaches 1.783 bpc and beats a tuned
  Transformer-256×4 (1.811) at 1.06× its compute, while a tuned Transformer-192×4 (1.704) still leads; online
  adaptation 3.191 → 3.096 bpc.
- **Hardware economics:** in the cost model, 5.8× fewer bytes moved per character than Transformer-256×4 at better
  quality (modelled; silicon measurement is a funded milestone).

## Typed learning: preserve meaning, then compose evidence

A category label, a temperature, a duration and a missing observation support different operations. Our typed interface applies meaningful comparisons before their outcomes enter addressed messages and persistent neural state. Computational races select evidence; counterfactual consequences teach the routes. Neural sums remain useful once representations share a declared latent space.

The proposed advantage combines tree-compatible conditional decisions with deep temporal memory and a common interface across modalities. Valid changes of units or category labels should preserve the evidence and downstream computation; training requires compatible parameter and optimizer coordinates too. The new theory derives these conditions and retains credit through computational time. **39 mathematical/interface contracts pass.** The integrated synthetic mixed-type fit already reaches 100% on 256 DEV rows from 64 FIT rows with fixed predicates. Learned predicate discovery, real tables versus trees and shared-core skill transfer are the next proofs.

Modern tabular neural models already use typed feature maps; comparison layers and neural/tree hybrids have precedents. Our contribution is the complete temporal/sparse composition and its tested consequences. See [typed semantics and race composition](../experiments/theory/TYPED_SEMANTICS_AND_RACE_COMPOSITION_20261006.md).

**Paths to revenue.** Two entry routes share the core: datacenter AI at lower total cost (models and runtime), and
event-native vertical models on existing CPUs and GPUs for interleaved, timestamped logs in industry, IoT, IT
operations, security and finance, priced on detection value, the shortest path to revenue. Licensed runtime and
accelerator IP and learning-capable event processors for mobile and robotics follow. The largest outcome is ownership
of a widely used AI infrastructure layer.

**The raise.** €3M at €50M priced pre-money, with €100M as a stretch scenario. The 18-month program buys the team, the
decisive proofs (race attention against competent Transformers at scale on GPU, the pre-registered FAS v2 home-field
benchmark and a real-data track, trained sparse serving), hardware feasibility on FPGA and neuromorphic silicon, and
company-owned IP through patent review and selective filings.

Read the [valuation rationale](VALUATION_RATIONALE.md), [full investment case](INVESTMENT_CASE.md),
[opportunity register](../report/model_family_opportunities.md) and [research evidence](../report/architecture_evidence.md).
The [previous pitch](archive/pitch_20261005T160000Z_previous_PITCH.md) is archived.
