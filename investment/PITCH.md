# Sleeping Machines — investor pitch

**The next general-purpose substrate for machine intelligence.**

Sleeping Machines aims to make intelligence composable within **one trainable computing substrate**. Language and reasoning, perception and world models, embodied action, typed tables and interleaved process streams become different forms of experience available to the same learner. The ambition extends through continual learning, communication and self-design to the hardware that executes them.

An observation enters through an interface that respects its meaning: a token, a sensory event, a typed comparison or an action. It becomes an addressed message interacting with persistent memory. Learned delays and temporal races decide which evidence meets and which computation happens. Small messages carry new evidence into deeper state; separate keys and values distinguish where evidence goes from what it says. Counterfactual credit teaches the routes and writes that could have happened.

The larger promise is that these are **reusable learned computations**. Comparison, binding, retention, prediction and planning learned from one form of experience can support another through shared representations and end-to-end credit. Perception can inform reasoning; reasoning can guide action; process traces and tables can contribute to the same world model. Joint training must measure that transfer and the retention of existing skills.

This also connects the learner to its execution. Time performs computation, including useful sleeps; local events can advance it without a mandatory global clock. Inference and learning are designed around the same temporal, addressed substrate. Available memory and skill can grow beyond the activity recruited for one observation, with work tuned from small data and constrained devices to large datasets and datacenters. Candidate discovery and alternative learning work count in that budget.

**A revealing proof of the platform:** mixed-type tables, token sequences and anonymous event streams learned through the same temporal core, followed by measured skill transfer. The program spans small-data usefulness and large-data scaling.

**What is already shown** (single seeds; native compute traced, reference compute shape-estimated):
- **Beats Transformers at a fraction of their compute.** On 10M characters of text8 the native model reaches 1.888 bpc
  against the 4-pass Transformer-256×4's 1.908, at 0.40× its training and 0.18× its inference compute, and beats the
  validation-selected tuned Transformers at both budgets (1.888 vs 1.996; 1.955 vs 2.215). Tuned small LSTMs lead at
  these budgets by 0.04–0.06 bpc.
- **Counterfactual route credit works:** 2.507 → 2.371 bpc for 0.3% extra training work.
- **Capacity beyond activity:** doubling the receiver pool improves 2.371 → 2.345 bpc at the same selected work.
- **Learned temporal computation:** 99.7–99.9% event-order accuracy from 2,000 examples seen once (Transformers
  33–41%); 95.3% on timing-only discrimination where any order-only model is capped at 50%; 100% on unseen modular
  rules; 100% retrieval at four times the training context.
- **Scales with data and adapts online:** 1.800 bpc after 90M characters; online adaptation 3.191 → 3.096 bpc.
- **Hardware economics:** in the cost model, 5.8× fewer bytes moved per character than Transformer-256×4 at better
  quality (modelled; silicon measurement is a funded milestone).

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
