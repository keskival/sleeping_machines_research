# Sleeping Machines — investor pitch

**The next general-purpose substrate for machine intelligence.**

Sleeping Machines is building one trainable architecture, learning rule and execution model for intelligence
everywhere: frontier language and reasoning, multimodal world models, embodied intelligence and robotics, event-native
analytics, continual on-device learning, adaptive communication and self-designing models, running economically from
datacenters to phones, robots and sensors, and ultimately on globally clockless, event-driven hardware that learns on
chip. General intelligence is the overarching aspiration.

Today's deep learning computes in lockstep: every layer runs at every step and memory is a buffer to rescan. Sleeping
Machines computes through time and sparse events. Messages carry content, an arrival time and an address; persistent
memories evolve with elapsed time; candidate routes race through learned delays; the winner acts, and the routes that
did not win still receive counterfactual credit. Delay-coded aggregation reproduces softmax attention exactly, so the
family contains Transformer-class computation, and dense synchronous layers are a special case. One model computes
densely where a task needs it and selectively everywhere else, with more stored capacity than selected work.

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
