# Sleeping Machines: the investment case

Sleeping Machines aims to be a **universal trainable computing substrate**: one composable architecture for language and reasoning, multimodal world models, embodiment, typed tabular data, synchronous samples, asynchronous events, anonymous interleaved process traces, token sequences, continual learning, communication, self-design and hardware.

The ambition combines distinct design axes: **computation through sleeps and temporal races; globally clockless event-driven execution; learning in the same temporal, addressed and asynchronous substrate as inference; capacity beyond selected activity; and tunable work for small-data, low-resource and large-scale settings**. Data modality is a separate axis from each of these. Scaling advantages and economical intelligence from sensors to datacenters are targets to measure. See [the full vision](../VISION.md).

**A learning architecture and computing substrate for intelligence everywhere.**

Prepared 3 October; revised 5 October 2026. Investment thesis and proposed commercial strategy;
completed research evidence is identified separately from product ambitions.

Current terms: **€3M raise at €50M priced pre-money**, with €100M as a separate
stretch scenario. Read the [current valuation rationale](VALUATION_RATIONALE.md)
and [application opportunity register](../report/model_family_opportunities.md).
The [previous memo](archive/valuation_20261004T171500Z_previous_INVESTMENT_CASE.md)
retains the earlier $10M discussion for historical review.

## Wins and advances at a glance (9 October 2026)

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
one-command kit for third-party reproduction. **One configuration wins all five:** a single configuration with no per-dataset tuning is ahead of the best
published result on all five EasyTPP datasets under a pre-registered protocol (24 of 25 seeds ahead): Taobao 1.397,
Taxi 0.526, StackOverflow −2.144, Retweet −6.324, Amazon 0.802.

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
  the leader's 250K). A race of clocks provably approximates any continuous inter-event density, confirmed numerically; likelihood saturates once the number of clocks covers the data's gap modes (predicted, then confirmed), so model size follows from the data.
- **Reasoning from few examples:** temporal event chains 99.7% vs Transformers 33–41%; race retrieval 100% at 4× context.
- **Character language:** beats tuned Transformers at equal or lower compute (text8, 10M characters).
- **Learning signal and capacity:** credit to unchosen routes cuts error by 0.14 bits per character for 0.3% more work;
  doubling memory slots lowers error with the same eight writes per input.
- **Generality:** one shared temporal core serves four event domains; anonymous interleaved process logs (FAS v1) beat six
  generic detectors across three seeds.
- **Theory:** a race of clocks contains softmax attention and Mamba-style selective state spaces exactly.

**New event domains, in development (9 October; development data, sealed tests not yet scored):**
- **Temporal Graph Benchmark, node affinity (tgbn-trade, world trade flows):** validation NDCG@10 **0.875** vs 0.860 for
  the leaderboard leader NAVIS (ICLR 2026) and for persistent forecast, ahead in every validation year, with 2,107
  parameters and 4 minutes of CPU fitting. The pre-registered sealed test (3 seeds; bar: NAVIS's 0.863) is queued.
- **Temporal Graph Benchmark, link prediction:** the official evaluation is reproduced exactly (EdgeBank 0.527 / 0.495,
  as published). tgbl-wiki in development: 0.776 vs TPNet's 0.842 (both validation; repeat interactions already above
  recency, new pairs are the gap being worked). tgbl-review: a training-free decayed event state reaches 0.344
  validation MRR, above TGN, TGAT and TNCN (0.313–0.325); GraphMixer leads at 0.428.
- **Second neural-TPP benchmark** (Bosser & Ben Taieb; 7 datasets × 5 splits): the frozen one-configuration EasyTPP
  model, with no tuning, is running all 35 pre-registered runs; verdicts on five-split means.

**Where we are behind** (stated with the numbers in the report): large-scale character language (90M), tuned LSTMs at 10M
characters, P12 mortality AUROC, real-world tables, FAS v2 (level on validation with a time-encoded Transformer reference, 0.702 vs 0.704, at about 1/7 of its parameters and a far better likelihood; ahead of the best classical detector's 0.685; sealed test pending, a tie expected under the 0.02 rule).

## The proposition

Sleeping Machines aims to make capable intelligence economical wherever it is
needed: in datacenters, on personal devices and inside machines that interact
with the world. We are developing models and their execution substrate together.
Time performs computation; messages race to select useful work; persistent
addressed memories retain context; counterfactual credit teaches alternatives
that did not win. The long-term implementation target is a globally clockless,
event-driven substrate supporting both inference and learning.

The investment opportunity is ownership of a useful new way to build, train and
execute AI. A successful platform could earn revenue through model/runtime
licensing, deployment software, accelerator IP and eventually hardware. Each
route draws on a common architecture and toolchain. The initial product must
solve a specific customer problem; the broader platform is the expansion path.

The ambition is the next general-purpose substrate for machine intelligence:
one architecture, learning rule and execution model spanning frontier language
and reasoning, multimodal world models, embodied intelligence and robotics,
event-native analytics, continual on-device learning, adaptive communication
and self-designing models, on hardware from datacenters to phones, robots,
sensors and clockless event processors that learn on chip. General intelligence
is the overarching aspiration. The case rests on a working research program
with concrete positive results and a staged way to turn them into customer
value; frontier capability, commercial serving savings and clockless chips are
the milestones this round funds.

## Why this could become a foundational platform

AI needs capability under constraints. A datacenter buyer cares about useful
output per dollar, rack watt and unit of resident memory. A mobile buyer cares
about capability under battery, heat, storage and connectivity limits. A robot
needs timely decisions, persistent context and adaptation as its environment
changes. These are different commercial requirements, with a shared technical
question: how much useful intelligence can a system obtain from the work it
actually performs?

Our architecture explicitly separates available capacity, scored keys,
selected state updates, delivered values and counterfactual learning work.
This makes capacity beyond selected activity a design objective. Deep memories
and learned routes allocate work to the current need while preserving a larger
store of skills and context. Scalable key discovery is the next engineering
step (the current implementation scores all keys in its pools).

If these mechanisms improve the quality/resource frontier, the benefit can be
spent on lower cost or greater capability at the same budget. For example,
a demonstrated 40% reduction in a relevant total cost would permit about
1.67 times the corresponding work at the same budget. The scaling study
measures whether the advantage widens with scale.

## The advantage portfolio

| Mechanism | Potential customer advantage | Present evidence and next boundary |
|---|---|---|
| Time as computation and race attention | Select and combine information through learned delays and arrival order; match irregular streams | Temporal/race implementations and integrated fits exist; precision, jitter and physical timing need hardware validation |
| Sparse addressed updates and separate keys/values | Deliver and update useful content without computing every candidate value during inference | Winner-only evaluator and arithmetic ledgers exist; actual-trained parity, rescore and total runtime remain pending |
| Persistent deep event representations | Maintain context across observations and compute from incoming messages plus stored state | Integrated language/event models learn; latest language scores use segment resets, not demonstrated indefinite memory |
| Capacity beyond selected activity | More useful state or skills at bounded selected work | Doubling p32 receiver slots improves quality at eight writes per input; key scoring, learning and storage grow |
| Counterfactual route credit | Train sparse hard choices and deepen useful computation without dense inference | Alternative-value credit improves completed depth-4 and depth-8 language fits; training alternatives still cost work |
| Event-driven, globally clockless, memory-local hardware | Energy from sparse local memory access (data movement dominates AI-chip energy) plus no clock tree and leakage-only idle capacity; average-case latency; thermal headroom for memory-on-logic stacking; a competitive model for the existing neuromorphic chip class ([hardware thesis](HARDWARE_THESIS.md)) | Mechanisms and energy accounting derived; no fabricated Sleeping Machines chip or measured joule advantage; local on-chip learning rule open |
| Online learning | Adapt to users, environments and drift near the point of use | Causal full-backbone CPU pilot improves predictions; stable continual learning and on-chip optimizer execution remain open |
| Computation during silence | Preserve relevant temporal evolution without periodic scans; reason about absence when needed | Temporal algebra and silence-aware supervision are part of the research; deadlines, readouts and physical retention are paid |
| Models and substrate developed together | Align learning, memory, execution and physical implementation rather than optimize one cost in isolation | Shared theory, implementations, contracts and audited results exist; an integrated commercial stack is still to be delivered |

Clockless circuits, sparse models, recurrent state and on-chip learning have
precedents. Intel's Loihi 2 describes asynchronous cores and programmable
learning. Our differentiation is the complete trainable construction:
deep content-bearing temporal computation, addressed persistent memory and
credit to unrealized hard routes, translated into a useful system. [Intel Loihi 2](https://www.intel.com/content/dam/www/central-libraries/us/en/documents/neuromorphic-computing-loihi-2-brief.pdf)

## What already makes the case tangible

**Public leaderboard wins on the family's home field (6–7 October 2026).** On EasyTPP (ICLR 2024), the standard benchmark
for marked event streams in continuous time, the model is a race of delayed clocks over persistent temporal memory.
Official splits, the published per-event log-likelihood protocol, 5 seeds, sealed test scored once per seed
([dossier](../experiments/B1_EASYTPP.md)):

| Dataset (nats/event) | Best published | Ours | Compute vs S2P2 |
|---|---|---|---|
| Taobao | 1.318 (IFTPP) | **1.399 ± 0.003** | 0.92× |
| Taxi | 0.522 (S2P2) | **0.525 ± 0.001** | 1/12 |
| Taxi, 5-seed mixture | 0.522 (S2P2) | **0.536** | 0.41× |
| StackOverflow | −2.163 (S2P2) | **−2.144 ± 0.004** | 1.26× |
| StackOverflow, matched | −2.163 (S2P2) | **−2.153 ± 0.005** | 1.015× |
| Retweet | −6.348 (NHP) | **−6.326 ± 0.001** | 1/15 |
| Amazon | 0.781 (S2P2) | **0.803 ± 0.001** | 0.29× |

Higher is better (log-likelihood, nats per event); compute is per-event inference work relative to S2P2. Every seed is
ahead of the best published result on every dataset.

**A second public win, with the same core, in clinical early warning (7 October 2026).** On P19 (PhysioNet 2019 sepsis
prediction, 38,803 ICU stays, 34 irregularly sampled channels, 4.2% positive), five official splits, TEST scored once per
split: **AUROC 0.916 ± 0.022 and AUPRC 0.639 ± 0.039** against the best published 0.903 / 0.583 (MTM, 2025), with
62,681 parameters ([dossier](../experiments/B2_IRREGULAR_TS.md)). The classifier reuses the EasyTPP model's temporal
memory layer — same code, same size — adding addressed channel memories with sufficient statistics, typed comparisons
and time-since-measurement as information. Paired five-split tests (8 Oct): removing the model's temporal memory (same network reading only its statistic slots) drops it to 0.900 / 0.572 on every split (paired ΔAUPRC +0.072, t = 31), and gradient-boosted trees on the same statistics score 0.914 / 0.618: the temporal memory carries signal the statistics lack, and the model leads the trees on AUPRC on every split (+0.026, paired t = 4.2; AUROC level). The result reproduces from scratch on separate hardware (AUROC within 0.0004 on all five splits).

**A third public win, in wearable sensors (8 October 2026).** On PAM (activity recognition from wearable inertial
sensors, eight classes), five official splits, TEST scored once per split under a pre-registered protocol: **accuracy
0.978 ± 0.007 and F1 0.980 ± 0.008** against the best published 0.975 / 0.976 (MTM, 2025), with 46,316 parameters.
Four of five splits are ahead; split 4 scores 0.964.

**Generality, stated precisely.** The EasyTPP leader S2P2 (NeurIPS 2025, GE HealthCare and UC Irvine) is shown on one task
type. One Sleeping Machines family now wins public benchmarks on generative event modelling and on sparse clinical
classification (P19) and wearable sensors (PAM), with further evidence on process logs, tokenized and character language
and few-shot temporal reasoning; large-scale language and real tables remain behind. One shared temporal core (14,080 parameters) trained jointly on Taxi, Taobao,
StackOverflow and Amazon matches or beats the separate per-dataset models on three of four (development data): mobility,
shopping and Q&A timing share one learned dynamics. Generative pretraining, on the same clinical cohort or on 31K
stays from another, gave no robust label-efficiency gain on mortality prediction (mean +0.004 AUROC at 10% labels)
([generality plan](../experiments/GENERALITY_PLAN.md)).

- **Taobao and Taxi are confirmed wins** at a fraction of the state of the art's compute.
- **StackOverflow is a confirmed win** in accuracy (+0.019 nats/event) at 1.26× compute, from a continuous-time state
  clock that passed the recording-grid audit.
- **Retweet is a confirmed win:** −6.326 ± 0.001 vs the best published −6.348 (NHP), all five seeds ahead, at 1/15 of
  S2P2's parameters and per-event compute.
- **Amazon is a confirmed win (7 October):** 0.803 ± 0.001 vs S2P2 0.781 ± 0.011, all five seeds ahead, with the best
  published time and type log-likelihoods, at 0.29× S2P2's per-event compute and 3.6× fewer parameters.
- **All five EasyTPP datasets are won**, against a pass criterion of two.
- **One configuration wins the whole benchmark (8 October):** a single model configuration, no per-dataset tuning,
  ahead of the best published result on all five datasets in a pre-registered protocol (24 of 25 seeds ahead), at 0.07×
  to 1.03× S2P2's per-event compute. It also beats our own dedicated models on Taxi, StackOverflow and Retweet.
- **How Amazon was won:** the previous protocol (0.784 ± 0.027) had one seed stuck in a weaker optimization basin; the
  diagnosis (delay windows drifting beyond all observed gaps) led to windows anchored to their gap component, which
  removed the basin: five seeds within 0.0015 of each other, one run each, no restarts.
- These are the first public-leaderboard results of the family. They show the core primitive is competitive on public
  terms where its mathematics fits: a race of clocks *is* a temporal point process.

These completed language comparisons use text8, 10M fitting characters, one
pass and the saved controls' T256 evaluation window. Each native row is a single
seed. Fitting work includes backward, route-credit and optimizer estimates.
Inference work is per evaluated input position; native winner-only values are
shape traces pending actual-trained backend admission, not measured serving.

| Model | Test bpc, lower better | Whole fitting TFLOPs est. | Fit MFLOPs/input position est. | Inference MFLOPs/input position est. |
|---|---|---|---|---|
| Native p32/D4, timing credit | 2.5064 | 7.22 | 0.72 | 0.163 |
| Native p32/D4, alternative-value credit | 2.3715 | 7.24 | 0.72 | 0.163 |
| Native p32/D4/pool4, alternative-value credit | 2.3452 | 11.95 | 1.20 | 0.164 |
| Native p64/D4, alternative-value credit | 2.1833 | 26.79 | 2.68 | 0.605 |
| Native p96/D4, alternative-value credit | 2.1625 | 58.65 | 5.87 | 1.324 |
| Saved LSTM-256 | 2.1706 | 20.31 | 2.03 | 0.677 |
| Saved Transformer-256x2 | 2.4269 | 111.26 | 11.13 | 3.710 |

Three observations are particularly relevant to investment:

- **A learning repair unlocks meaningful capability.** Value-informed route
  credit improves p32/D4 by 0.135 bpc with about 0.3% more counted fitting work
  and unchanged hard forward behavior. The improvement also appears in a
  completed depth-8 model. This is evidence that learning the alternatives
  matters, rather than evidence that additional activity alone explains gains.
- **The sparse native construction beats Transformers at lower compute.**
  Credited p32/D4 beats the saved one-pass Transformer by 0.0554 bpc with about
  15.4 times less estimated fitting work and 15.2 times fewer parameters. With
  more passes, the native model reaches 1.888 bpc against the 4-pass
  Transformer-256x4's 1.908 at 0.40x its training and 0.18x its inference
  compute.
- **Quality and useful capacity improve.** The p96 model slightly exceeds the
  saved LSTM's quality, while using about 2.9 times its fitting work. At p32,
  doubling slots from 16 to 32 improves 2.3715 to 2.3452 with eight selected
  writes unchanged, but 16 to 32 scored keys and about 1.65 times fitting work.
  Useful capacity grows beyond selected activity.
- **Tuned baselines at our budgets (5 October 2026).** Ten dense baselines retuned at the native compute budgets and
  selected by validation: the native model beats every tuned Transformer at both budgets (1.888 vs 1.996; 1.955 vs
  2.215 bpc, at about equal compute). Tuned small LSTMs lead at these budgets (1.825 at ≤ 352 TF; 1.915 at ≤ 107 TF),
  on local 2–4-character modelling. The strategic contest is against Transformers at scale.

A separate integrated full-backbone online CPU experiment scores 3.1909 bpc
with frozen weights versus 3.0957 with adaptation on 8,191 new development
targets. Predictions precede each 16-character block update; both arms have
persistent event state. Next: gains under drift with retained quality and
bounded update cost.

The program also spans event speech, event vision, tabular tasks, interleaved
industrial event logs and temporal prediction, each at its own maturity; the
report lists every win and loss per domain. Invalid target-dependent mixtures
are quarantined and excluded from this investment case.

Completed parents and exact values are indexed by the accompanying evidence
manifest. [Research status](../report/sleeping_machines_status.pdf)

The numerical table above remains the frozen 3 October 10M comparison.
At 90M characters, against tuned references at equal training compute, the results are mixed:
- **≈ 0.96 PF:** native 1.800 bpc loses to a tuned LSTM-512 (1.729) and a tuned Transformer-192×4 (1.780).
- **≈ 2.1 PF:** native 1.783 beats a tuned Transformer-256×4 (1.811) at 1.06× its compute, while a Transformer-192×4
  (1.704) leads.

The gap to the best dense model widens from 10M to 90M; that trend is the central language problem (report Part I
§4.3).

**Tokenized language (8 October 2026).** A temporal-memory token model with the keyed predecessor read scores
6.009 ± 0.010 nats per token (3 seeds) on the 65,528-target GPT-2-tokenized FineWeb slice at 1M training tokens, against a Kneser–Ney trigram's 6.537 and the trigram at 4× the data (6.100); at 4M tokens 5.510 vs 6.100, so the lead grows with data. The keyed predecessor read behind it also solves associative recall across irregular gaps (97.5%, 3 seeds) (report Part I §4.5). The count models are a floor, not the frontier; a published small Transformer under the same
protocol is the next reference. See the [current evidence map](../report/architecture_evidence.md).

## Expanded platform opportunities

The [opportunity register](../report/model_family_opportunities.md) specifies
irregular-stream ingestion, asynchronous neural codecs, task-oriented
communication, learned memory/update policies and joint embodied cognition.
These can exploit one event/state/learning interface across local and remote
regions. Variable-rate coding, irregular-input models and joint VLA learning
already have precedents. The proposed value is their economical temporal and
selective integration, with first proof conditions and current gaps explicit.

Joint motor/reasoning learning can teach shared physical abstractions and
planning skills; transfer in each direction is tested separately on held-out
tasks. General intelligence is the overarching aspiration.

## Markets: one foundation, distinct products

**Datacenters and frontier-model developers.** Start with a reproducible
quality/cost improvement on a defined workload. Buyers could license a model,
runtime or accelerator design that increases useful output within their power,
memory and latency budgets. Efficient inference is one entry; efficient
training and stronger scalable models are larger opportunities. A software
proof can precede the capital demands of custom silicon.

The IEA's updated outlook puts datacenter electricity at 485 TWh in 2025 and
projects 950 TWh in 2030. That is the scale of the constraint (not all of it
is addressable AI spend); validated quality per watt and dollar has
substantial economic value.
[IEA outlook](https://www.iea.org/reports/key-questions-on-energy-and-ai/executive-summary)

**Mobile, wearables and personal devices.** The desired product is an SDK/model
and later licensed compute IP for continuously available contextual AI:
speech, sensing, assistance and personalization under a device power budget.
Persistent state and local learning could reduce cloud dependence and feedback
latency. Keeping processing local can reduce raw-data transfer, but the product
must implement its privacy and update policy. Online learning must be shown
to improve usefulness without unacceptable forgetting or additional energy.

GSMA's 2026 report describes 5.8 billion unique mobile subscribers and 8.8
billion wireless connections. Those are ecosystem counts, not compatible
device shipments or paying customers. The opportunity is an OEM deployment
that becomes a repeatable design win, then expands across device families.
[GSMA Mobile Economy 2026](https://www.gsma.com/solutions-and-impact/connectivity-for-good/mobile-economy/wp-content/uploads/2026/02/The-Mobile-Economy-2026.pdf)

**Robotics and autonomous machines.** Continuous sensor fusion, memory,
low-latency response and adaptation offer a natural application for an event
substrate. The proposed product is a model/runtime or embedded accelerator
that improves a measured perception or adaptive-control workload. Learning
from deployment could eventually address changing payloads, environments and
sensor characteristics. Closed-loop transfer, latency, robustness and retained
skills are proven in robotic trials, the next stage after recorded-stream
models.

IFR reports five million industrial robots operating in 2025 and more than
600,000 new installations that year. This is a concrete industrial ecosystem,
not a forecast of our sales; service robots and future autonomous devices are
additional possibilities. [IFR World Robotics 2026](https://ifr.org/ifr-press-releases/news/five-million-robots-now-operate-in-factories-globally)

**Everything between edge and cloud.** Industrial monitoring, network/telecom
streams, connected vehicles, private enterprise AI, environmental sensing and
adaptive forecasting share aspects of the same opportunity. Some need sparse
continuous sensing; others need capable private inference or adaptation. They
are expansion options after a repeatable first product, with their own data,
quality and deployment protocols.

## Define the upside in economic terms

The following are transparent, independent scenarios, not forecasts, observed
prices or a computed market size. They show how a narrow deployment can become
a venture-scale business without requiring an immediate win in every domain.

| Route | Explicit hypothetical assumptions | Annual company revenue implied |
|---|---|---|
| Datacenter software/IP | Reach $1B/year of eligible customer execution spend; reduce total eligible cost 20%; capture 10-20% of the $200M value created | $20-40M |
| Mobile compute IP | Win 100M newly licensed devices/year at $0.25-$1.00 per device | $25-100M |
| Robotics/industrial runtime | Serve 1M paying active units at $20-$100 per unit/year | $20-100M |
| Scaled platform | Achieve $250M recurring annual revenue across a validated product portfolio | $250M; requires adoption and sustained delivery |

The datacenter scenario's $200M saving precedes our fee: a $20-40M fee leaves
$160-180M for customers before migration, support and update costs. Final
customer savings must include all those costs. Device royalties are paid on actual licensed units; existing installed
devices are not annual shipments. Robotics revenue assumes a recurring product,
not an upfront chip sale. The routes can overlap, so do not sum them into a TAM.
Chip revenue, margins and financing needs differ substantially from software/IP.

For scale intuition only, $250M recurring revenue at an assumed 8-15 times
revenue valuation would imply $2.0-3.75B enterprise value. Neither that revenue
nor those multiples is a prediction or a current comparable. A frontier-model
and substrate platform with durable billion-dollar annual revenue could
support much larger, potentially tens-of-billions outcomes under suitable
economics. The route to that upside is owning a widely adopted layer of AI
infrastructure, with defensible quality/cost benefits and meaningful value
capture.

This is why the upside can justify early investment: capital today buys an
opportunity to establish that layer before the complete platform is proven.
The opportunity has correlated technical risks across its products; several
applications do not constitute independent chances of success.

## Defensibility and capital efficiency

The moat combines a learning method, an execution contract,
model/runtime engineering, memory/routing design, hardware mappings and a
growing corpus of reproducible evidence. Successful deployments can add
integration know-how, task-specific learning policies and developer adoption.
The combination is harder to reproduce than any isolated primitive.

No patents are filed yet. The historical manifesto is public and cites a 2021
origin; the round funds a documented inventory of ownership, contributor
rights, dependencies, licenses and protectable implementations, and a
freedom-to-operate review.

Funding should convert this proposed moat into documented company assets:
ownership and disclosure review, patent counsel's assessment, selective priority
filings and trade-secret practice. These sit within the existing €200,000
legal/IP/operations allocation, subject to quotes and staged filing decisions.
The founder's earlier patents belong to his former employer and contribute
experience, not venture-owned rights. See the [IP protection plan](IP_PROTECTION_PLAN.md).

There are financing precedents for both parts of the thesis: Liquid AI
announced a $250M Series A for efficient general-purpose models in December
2024; Innatera reported a EUR15M Series A for neuromorphic edge technology in
March 2024. They demonstrate investor interest in these categories. Their
teams, maturity and funding amounts do not establish our valuation.
[Liquid AI announcement](https://www.liquid.ai/blog/we-raised-250m-to-scale-capable-and-efficient-general-purpose-ai), [Innatera announcement](https://www.innatera.com/newsroom/innatera-raises-e15m-for-neuromorphic-edge-ai/)

The proposed sequence is capital-efficient: establish the algorithm and
software resource frontier; secure a workload/design partner; validate
fixed-precision hardware semantics on FPGA; fund ASIC development after
quality, economics and demand justify it. A clocked FPGA can validate event
semantics; it cannot demonstrate the energy of a fabricated clockless ASIC.

## A basis for today's valuation

The current opening proposal is **€3M at €50M priced pre-money**; **€100M**
is a stretch scenario. This prices the potential of a model/learning/runtime
and computing-substrate platform, supported by research execution and a staged
program. It is a negotiating position, not an independent appraisal.

**Public de-risking since the proposal was set.** The confirmed EasyTPP wins (Taobao, Taxi, StackOverflow) and the P19 clinical win are the first instances of
the evidence the rationale names as most valuable: a repeated useful-quality and full-cost advantage, on a public
leaderboard with published baselines.
- They make the €50M opening price materially more defensible.
- The €100M case needs the advantage beyond one benchmark family. Its three conditions hold as of 7 October
  (VALUATION_RATIONALE.md, condition status):
  - further datasets at matched compute: StackOverflow at 1.015× S2P2's per-event compute;
  - a second domain: P19 sepsis, AUROC 0.916 vs 0.903 published;
  - an independent reproduction: the Taxi win rerun from scratch on separate hardware, 0.5252 ± 0.0007 vs 0.5250 ±
    0.0010 (AWS) and S2P2's 0.522 (same team; a third-party rerun is next).

The rationale now treats asynchronous ingestion/output, learned memory policies
and embodied/cognitive integration as concrete expansion options. The common
structure could permit end-to-end learning across their information and credit
paths. Each has its prior art and first discriminating test.

Execution supports taking the ambition seriously: implemented integrated
models, completed mechanism comparisons, larger-data learning, retained
failures and protocol corrections, and reviewable source/resource contracts.
Tero's relevant public engineering/research/invention history adds an execution
case, subject to references and commitment. Scaling a company, building a
team and establishing usable rights are separate obligations. Research
co-authorship remains credited; sole-founder status does not settle ownership.

At €50M pre-money, €3M produces €53M post-money and 5.6604% initial investor
ownership; the €100M case produces €103M and 2.9126%, before other dilution.
The planned €3M/18-month program adds learning/runtime/evaluation capacity and
stages compute and hardware feasibility against explicit gates. Software and
one useful workload can create value before the full platform is complete.

The [valuation rationale](VALUATION_RATIONALE.md) documents positive assets,
execution evidence, financing precedents, equity arithmetic and remaining
proofs. Comparable seed funding announcements establish category investability;
unreported pre-money prices cannot be inferred from round amounts. Several
applications share risks and must not be summed into a TAM or independent
success probabilities. The vision supports exceptional upside; investor demand,
rights, execution and reproducible benefits determine attainable terms.

## What the next investment should buy

1. **A reproducible model advantage at scale.** Race attention against
   competent Transformers at equal complete compute on modern language data,
   across a scaling curve on GPU, with replications.
2. **Extend the home-field wins on event streams.** Public EasyTPP wins are
   achieved (Taxi, Taobao, StackOverflow). Next:
   - the remaining datasets at matched compute;
   - a second public domain (irregular clinical and sensor series: P12, P19, PAM);
   - the sealed FAS v2 confirmation on anonymous interleaved logs, against small LSTM/Transformer references;
   - identity-stripped real logs with a design partner.
3. **A deployable sparse backend.** Admit actual-trained winner/state/cache/RNG
   parity and heldout rescore, then measure setup, residency, latency,
   throughput and total cost. The prepared packed-weight worker is an initial
   runtime implementation; its lifecycle checks pass, native admission is open.
4. **One customer workload and one adoption route.** Define the buyer, useful
   quality level, deployment constraints and acceptable integration cost.
   Proposed first route: a software/model/runtime proof for datacenter AI,
   while validating event-stream strengths for later edge/robotics products.
5. **A credible adaptation result.** Demonstrate prequential gains under drift,
   retained stationary quality and bounded update resources against competent
   online baselines. Distinguish state updates from learned weight adaptation.
6. **A hardware execution contract.** Export validated traces; establish event
   ordering, precision, timing, local state and backpressure. Measure clock,
   communication, memory and learning overhead before an ASIC commitment.
7. **Company-owned IP protection.** Document contributions and assignments,
   review prior disclosures and prior art, and file selectively for qualifying
   technical inventions. Stage prosecution and international protection by
   commercial relevance; retain confidential implementation know-how.

The main failure conditions are explicit: gains disappear against competent
controls; discovery/learning/traffic consume the savings; depth or adaptation
fails to retain useful information; timing precision/control defeats hardware
economics; or customers cannot capture enough value after integration. These
determine which product path to pursue and when to change course.

The investable thesis is a staged path from an unusual trainable architecture
to a repeatable economic advantage, then to a platform spanning the cloud and
physical world. We can earn substantial value with a successful first product
while retaining the much larger ambition of frontier intelligence and a new
computing substrate.

## Evidence and commercial diligence

[Evidence manifest](evidence_20261003.json) records completed-parent hashes,
precise metrics, arithmetic conventions and primary market sources accessed
on 3 October 2026. [Technical milestones](../experiments/DATACENTER_VALUE_MILESTONES.md),
[hardware thesis](../experiments/HARDWARE_VALUE_PROPOSITION.md) and
[frontier protocol](../experiments/FRONTIER_COMPUTE_PROTOCOL.md) supply the next
validation boundaries. No customer commitments, revenue, patent grants,
fabricated silicon or current frontier-scale capability are represented as
existing assets in this case.
