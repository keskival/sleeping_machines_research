# Sleeping Machines — investor pitch

**The next general-purpose substrate for machine intelligence.**

Sleeping Machines aims to make intelligence composable within **one trainable computing substrate**. Language and reasoning, perception and world models, embodied action, typed tables and interleaved process streams become different forms of experience available to the same learner. The ambition extends through continual learning, communication and self-design to the hardware that executes them.

An observation enters through an interface that respects its meaning: a token, a sensory event, a typed comparison or an action. It becomes an addressed message interacting with persistent memory. Learned delays and temporal races decide which evidence meets and which computation happens. Small messages carry new evidence into deeper state; separate keys and values distinguish where evidence goes from what it says. Counterfactual credit teaches the routes and writes that could have happened.

The larger promise is that these are **reusable learned computations**. Comparison, binding, retention, prediction and planning learned from one form of experience can support another through shared representations and end-to-end credit. Perception can inform reasoning; reasoning can guide action; process traces and tables can contribute to the same world model. Joint training must measure that transfer and the retention of existing skills.

This also connects the learner to its execution. Time performs computation, including useful sleeps; local events can advance it without a mandatory global clock. Inference and learning are designed around the same temporal, addressed substrate. Available memory and skill can grow beyond the activity recruited for one observation, with work tuned from small data and constrained devices to large datasets and datacenters. Candidate discovery and alternative learning work count in that budget.

**A revealing proof of the platform:** mixed-type tables, token sequences and anonymous event streams learned through the same temporal core, followed by measured skill transfer. The program spans small-data usefulness and large-data scaling.

## Evidence anchors for the platform

**Token sequences, anonymous interleaved process logs and mixed-type comparisons now learn within the temporal/sparse family.** They exercise different information structures through addressed messages, persistent memory, temporal computation and credit to unrealized alternatives. This is evidence for a common computing construction spanning small-data tables, asynchronous analytics and tokenized language.

| Evidence front | Completed indication | What it establishes |
| --- | --- | --- |
| **Public benchmark: timed event data (EasyTPP, ICLR 2024)** | **Taobao 1.399 ± 0.003 vs best published 1.318** nats/event (+0.081) at 0.92× the leader's compute; **Taxi**: a 5-seed mixture (0.536) beats every published model at 0.41× S2P2's compute (NeurIPS 2025), and a single model leads S2P2 on the mean (0.525 vs 0.522) at 1/12 of its parameters and compute. **StackOverflow** −2.144 ± 0.004 vs −2.163 (+0.019) at 1.26× S2P2's compute. **Retweet −6.326 ± 0.001 vs best published −6.348** (NHP; S2P2 −6.365) at **1/15 of S2P2's parameters and per-event compute**. Official splits, sealed test, 5 seeds. | **All five datasets won on a public leaderboard of the family's home field**, four of them at a fraction of the state of the art's compute; the Taxi win reproduced independently. |
| **Public benchmark: ICU sepsis prediction (P19, PhysioNet 2019)** | **AUROC 0.916 ± 0.022, AUPRC 0.639 ± 0.039** vs best published 0.903 / 0.583 (MTM, 2025) on the five official splits; 62,681 parameters. Without its temporal memory the same network scores 0.900 / 0.572 on every split; trees on the same statistics 0.914 / 0.618: the temporal memory carries signal the statistics lack (AUPRC ahead of the trees on every split). Reproduced on separate hardware. | **A second public win, in clinical early warning**, with AUPRC (the clinically relevant metric at 4% prevalence) ahead beyond split spread. |
| **Public benchmark: wearable activity recognition (PAM)** | **Accuracy 0.978 ± 0.007, F1 0.980 ± 0.008** vs best published 0.975 / 0.976 (MTM, 2025), five official splits, 46,316 parameters; four of five splits ahead. | **A third public domain won**: wearable sensor streams. |
| **One family across data types** | The P19 classifier reuses the event model's temporal memory layer (same code, same size). | Wins on generative event modelling and on sparse clinical classification; the EasyTPP leader S2P2 is shown on one task type. Transfer tests are next ([generality plan](../experiments/GENERALITY_PLAN.md)). |
| **Anonymous interleaved processes** | FAS mean AUROC **0.5924 vs 0.5587** at 256 events; **0.7370 vs 0.7272** at 512 events. All three native seeds beat the best of six saved generic controls at both points. | Replicated early-detection quality wins on FAS v1. Privileged identity-assisted diagnostics are excluded; sealed v2 and stronger neural references are the next benchmark. |
| **Mixed-type tables, small data** | **100% accuracy on 256 synthetic DEV rows from 64 FIT rows**,6,370 learned parameters. Numeric, categorical, Boolean and missingness comparisons enter before neural message processing. | The integrated core learns the mixed-type interaction; relabeling, column-order, missing-value semantics and actual alternative-write credit checks pass. Fixed predicates, one seed; learned discovery and real tables versus trees are the next tests. |
| **Properly tokenized language, growing data** | P24 reaches **7.2515 / 7.2636 DEV NLL** at 1M TRAIN tokens across two seeds. Memory erasure raises loss **0.0104 / 0.0174**. Fixed-width seed6 quality improves **8.0333 → 7.7417 → 7.2515** from 64K → 256K → 1M TRAIN tokens. | Replicated language learning and useful persistent memory, with measured data-growth quality. Two passes; public Transformer quality, full larger fitting work and the reserved 4M point are the next comparisons. No scaling law is fitted. |

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
