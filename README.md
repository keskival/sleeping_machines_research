# Sleeping Machines

**Deep learning that computes with time.**

## The ambition

Sleeping Machines aims to make intelligence composable within **one trainable computing substrate**. Language and reasoning, perception and world models, embodied action, typed tables and interleaved process streams become different forms of experience available to the same learner. The ambition extends through continual learning, communication and self-design to the hardware that executes them.

An observation enters through an interface that respects its meaning: a token, a sensory event, a typed comparison or an action. It becomes an addressed message interacting with persistent memory. Learned delays and temporal races decide which evidence meets and which computation happens. Small messages carry new evidence into deeper state; separate keys and values distinguish where evidence goes from what it says. Counterfactual credit teaches the routes and writes that could have happened.

The larger promise is that these are **reusable learned computations**. Comparison, binding, retention, prediction and planning learned from one form of experience can support another through shared representations and end-to-end credit. Perception can inform reasoning; reasoning can guide action; process traces and tables can contribute to the same world model. Joint training must measure that transfer and the retention of existing skills.

This also connects the learner to its execution. Time performs computation, including useful sleeps; local events can advance it without a mandatory global clock. Inference and learning are designed around the same temporal, addressed substrate. Available memory and skill can grow beyond the activity recruited for one observation, with work tuned from small data and constrained devices to large datasets and datacenters. Candidate discovery and alternative learning work count in that budget.

The project is therefore a proposal for **a common foundation for learning, representing and executing intelligence**. The experiments test pieces of that construction. Mixed-type tabular learning is a decisive bridge: meaningful numeric, ordinal, categorical and missingness comparisons should feed the same temporal core used for sequences and asynchronous events. Strong results across small and large data would demonstrate the breadth of the unification. Current CPU implementations and separate-domain fits support specific mechanisms; joint transfer, scalable typed learning and clockless learning hardware have concrete tests ahead. See [the full vision](VISION.md).

## Wins and advances at a glance (9 October 2026)

**Nine public leaderboard wins**, each on the official splits with a sealed test scored once per run, 40 of 41 independent
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
| **TGB tgbn-trade** (temporal graph: node affinity, world trade) | NDCG@10 0.868 ± 0.0005 | 0.863 (NAVIS, ICLR 2026) | 2,107 parameters, 4 CPU-minutes; **a new domain**: interaction graphs, where heuristics beat every temporal GNN |
| **TGB tgbl-wiki** (temporal graph: link prediction, Wikipedia edits) | MRR 0.835 ± 0.0003 | 0.827 (TPNet) | 7,995 parameters; exact per-event state; 4.5 ms per query on one CPU thread |

**Rigor:** pre-registered reporting rules; the benchmark's own scorer agrees; a recording-grid audit every win passes; a
one-command kit for third-party reproduction. **One configuration wins all five:** a single configuration with no per-dataset tuning is ahead of the best
published result on all five EasyTPP datasets under a pre-registered protocol (24 of 25 seeds ahead): Taobao 1.397,
Taxi 0.526, StackOverflow −2.144, Retweet −6.324, Amazon 0.802.

**Advances beyond the leaderboards:**
- **Tokenized language:** 6.009 ± 0.010 nats per token vs a Kneser–Ney trigram's 6.537 on GPT-2-tokenized FineWeb (3 seeds);
  at 4× the data 5.510 vs 6.100, so the lead grows with data.
- **Binding and recall:** associative recall across irregular gaps 97.5% (3 seeds) vs a 14% baseline, 91.6% at twice the
  training length (seed 0; three-seed mean 90.2 ± 1.2%); **learning through the race alone** (no backpropagation into the network) reaches 77%.
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

**New event domains (9 October):**
- **Temporal Graph Benchmark, node affinity: won** (table). Pre-registered sealed test, 3 seeds: 0.8683 / 0.8674 /
  0.8683, every seed above NAVIS's 0.863; persistent forecast 0.855. Validation 0.874 vs 0.860 had predicted it.
- **Temporal Graph Benchmark, link prediction: won** (table). Exact per-event state lifted our model from 0.776 to
  0.852 validation MRR (theory note 159 predicted at least +0.03); three pre-registered sealed seeds score 0.8353 /
  0.8350 / 0.8356 on test, every one above TPNet's 0.827, with 7,995 parameters. On tgbl-review a training-free decayed
  event state already reaches 0.344 validation MRR, above TGN, TGAT and TNCN (0.313–0.325); GraphMixer leads at 0.428.
- **Second neural-TPP benchmark** (Bosser & Ben Taieb; 7 datasets × 5 fixed splits; total NLL per sequence, lower is
  better): the frozen one-configuration EasyTPP model, with no tuning, has six completed verdicts, all
  losses against the composite bar (best published time + best published mark NLL). **Wikipedia lost** under the
  pre-registered repair rule: split 0's original run ended with NaN weights, and its guarded rerun scores +58,218 on
  TEST because two of 118 test sequences receive astronomically confident hazards (millions of nats; median sequence
  3.9), so the five-split mean is +11,421.66 vs −122.62. The other four splits average −277.54, and the **marks are the
  strongest result in this benchmark**: 27.65 vs 144.79 best published, because our addressed mark memory copies pages
  never seen in training (10–27% of test edits), which per-mark embeddings cannot do. Bounding the hazards is the
  development fix. **MOOC lost:** −226.85 ±
  2.83 SE vs −239.7; timing carries the gap (recording-grid analysis in the B4 loss analysis). **Stack Overflow lost:** 12.714 ± 0.874 SE vs 11.9; its time NLL beats the
  published time component (−91.598 vs −91.1), while mark NLL trails (104.312 vs 103.0). **MIMIC2 lost** (7.01 vs
  2.42; its mark path overfits ~100-sequence training sets). **Github lost by 2.3 nats:** −270.62 ± 48.69 SE vs the bar
  −272.90, yet below every single published model's total (best LNM++ −269.7) and ahead on time (−390.93 vs −382.4);
  marks carry the gap (120.30 vs 109.5). Splits 1–4 are guarded reruns after a numerical failure (one long training
  sequence gives a non-finite gradient and is skipped). **Retweets lost:** −515.16 ± 3.24 SE vs −538.70; marks are level with the best published (83.65 vs 82.63), and timing carries the gap (−598.81 vs −621.33) on a one-second grid with 4% zero gaps (recording-grid analysis in the B4 loss analysis). LastFM is pending. These are
  frozen-configuration transfer results; mark-path development continues outside the sealed protocol.

**Where we are behind** (stated with the numbers in the report): large-scale character language (90M), tuned LSTMs at 10M
characters, P12 mortality AUROC, real-world tables, FAS v2 (sealed-test tie: native 0.70366 vs time-encoded Transformer 0.70432 AUROC, three seeds; mean AUROC difference −0.00066; native ahead of the best classical detector's 0.68223).

## Headline evidence for the universal substrate

**Token sequences, anonymous interleaved process logs and mixed-type comparisons now learn within the temporal/sparse family.** They exercise different information structures through addressed messages, persistent memory, temporal computation and credit to unrealized alternatives. This is evidence for a common computing construction spanning small-data tables, asynchronous analytics and tokenized language.

| Evidence front | Completed indication | What it establishes |
| --- | --- | --- |
| **Public benchmark: temporal point processes (EasyTPP)** | **Taobao 1.399 ± 0.003 vs best published 1.318** nats/event (+0.081); **StackOverflow −2.144 ± 0.004 vs −2.163** (S2P2, NeurIPS 2025), and −2.153 ± 0.005 at matched size and compute; **Taxi 0.525 ± 0.001 vs 0.522** at **1/12 of S2P2's parameters and per-event compute**, reproduced from scratch on separate hardware (0.5252 ± 0.0007); **Retweet −6.326 ± 0.001 vs best published −6.348** (NHP; S2P2 −6.365) at **1/15 of S2P2's parameters and per-event compute**; a 5-model mixture reaches 0.536 at 0.41× S2P2's compute. Five seeds, official splits and protocol. StackOverflow: mixture ahead (−2.154 vs −2.163) at 4.6× compute, an accuracy-only result. **Amazon 0.803 ± 0.001 vs 0.781** (S2P2) at 0.29× its per-event compute, all five seeds ahead: **all five EasyTPP datasets won, and a single configuration with no per-dataset tuning also wins all five** (24 of 25 seeds ahead; 0.07–1.03× S2P2's per-event compute). | Confirmed wins on a public leaderboard of the family's home field: a race of delayed clocks over persistent temporal memory, with exact likelihood. [Report §4.0](report/I_SCIENCE.md) |
| **Public benchmark: ICU sepsis prediction (P19, irregular clinical time series)** | **AUROC 0.916 ± 0.022, AUPRC 0.639 ± 0.039** vs best published 0.903 / 0.583 (MTM, 2025), five official splits; without its temporal memory 0.900 / 0.572 (every split), trees on the same statistics 0.914 / 0.618; reproduced on separate hardware. | Event-native clinical analytics: addressed channel memories with sufficient statistics, typed comparisons and informative silence. [Report §4.0b](report/I_SCIENCE.md) |
| **Public benchmark: wearable activity recognition (PAM)** | **Accuracy 0.978 ± 0.007, F1 0.980 ± 0.008** vs best published 0.975 / 0.976 (MTM, 2025), five official splits, 46,316 parameters; four of five splits ahead. | A third data type won with the same event-native construction: dense multichannel sensor streams. |
| **One family across data types** | The same temporal memory layer (same code, same size) underlies the EasyTPP wins and the P19 clinical win; further evidence on process logs, language and reasoning. Losses stay visible: large-scale language and real tables. | A shared construction wins public benchmarks of different kinds. Transfer tests next: [generality plan](experiments/GENERALITY_PLAN.md) |
| **Anonymous interleaved processes** | FAS mean AUROC **0.5924 vs 0.5587** at 256 events; **0.7370 vs 0.7272** at 512 events. All three native seeds beat the best of six saved generic controls at both points. | Replicated early-detection quality wins on FAS v1. Privileged identity-assisted diagnostics are excluded; sealed v2 and stronger neural references are the next benchmark. |
| **Mixed-type tables, small data** | **100% accuracy on 256 synthetic DEV rows from 64 FIT rows**,6,370 learned parameters. Numeric, categorical, Boolean and missingness comparisons enter before neural message processing. | The integrated core learns the mixed-type interaction; relabeling, column-order, missing-value semantics and actual alternative-write credit checks pass. Fixed predicates, one seed; learned discovery and real tables versus trees are the next tests. |
| **Properly tokenized language (FineWeb, GPT-2 BPE)** | With the keyed predecessor read, the temporal-memory token model scores **6.009 ± 0.010 nats/token** (3 seeds) on the 65,528-target slice at 1M TRAIN tokens, against Kneser–Ney trigram 6.537 at the same data and 6.100 at 4×; the read contributes 0.175. At 4M tokens it scores **5.510** against the trigram's 6.100: the margin grows with data (0.528 → 0.590, 2 seeds at 4M). The earlier native P24 model sits at bigram level (7.25 on the 2,040-target slice). | First language evidence that persistent memory binds context; the recall gate behind it is met over 3 seeds. [Report §4.5](report/I_SCIENCE.md) |

**The unifying advantage is the reusable construction:** type-respecting comparisons and tokens become events; events recruit meaningful state and teach hard routes through their consequences. Available capacity, selected activity, temporal learning and execution are separate design axes. The larger ambition connects language and reasoning, multimodal world models, embodiment, continual learning, communication, self-design and clockless hardware through this substrate. Joint training with shared parameters must next measure skill transfer and donor-skill retention. These completed instances were trained separately.


Today's deep learning computes in lockstep: every layer of a dense model runs at every step, whether or not anything
changed, and its memory is a buffer it must rescan. Brains and physical systems compute differently, through timing,
sparse events, persistent local state and competition. Sleeping Machines turns that into a trainable engineering
discipline. Every message carries **content, an arrival time and an address**. Receivers hold persistent memories that
decay and rotate with the real time elapsed between events. Candidate routes **race through learned delays**; the
winner updates state and sends the next message, and the routes that did not win still receive **counterfactual
credit**. Arrival order decides which memories meet and which computation happens, so delays do computation rather than
merely label it. Dense synchronous layers are included as a special case: one model can compute densely where a task
needs it and selectively everywhere else.

### One substrate, many fronts

| Front | What the substrate brings | First anchors → next proof |
|---|---|---|
| **Frontier language and reasoning** | A Transformer-capable function class: delay-coded aggregation reproduces softmax attention exactly; winner-only races skip value aggregation; persistent memory beyond a context window; learning at test time | At 10M characters the native model beats Transformers at 0.40× their training and 0.18× their inference compute → properly tokenized CPU fitting against pinned public Transformer records, with complete work accounting and a reserved larger point |
| **Multimodal world models** | Language, vision, audio, touch, sensor and event streams update one persistent world state at their own cadences; dense and event-driven regions coexist | Common event interface and primitives implemented across language, speech, vision events, markets and synthetic tasks → joint training with measured cross-domain transfer |
| **Embodied intelligence and robotics** | Timed actions, irregular sensing, instruction-conditioned control; motor experience and reasoning shaping shared representations | [Robotics protocol](experiments/AWS_EARLY_INDICATION_MATRIX.md) on recorded force/pose streams → joint manipulation and reasoning with transfer in both directions |
| **Event-native analytics** | Interleaved, timestamped streams from many concurrent processes: industrial and IoT maintenance, IT operations, security, transactions and order flow, clinical data, neural recordings for brain–computer interfaces, event cameras, speech | Early fault detection ahead of every generic control on FAS; 79.7% held-out spiking-speech accuracy → [pre-registered FAS v2](experiments/FAS_V2_CONFIRMATORY_PROTOCOL.md), real logs, a design partner |
| **Continual and on-device learning** | Delays, routes and content adapt at the point of use; slow learned rules govern fast memory writes, retention and scheduling | Online adaptation improves 3.191 → 3.096 bpc on a new stream → native test-time training and drift retention |
| **Communication** | Learned messages become variable-rate asynchronous codecs; distributed agents send only the evidence a remote predictor needs | [Codec and task-communication designs](report/model_family_opportunities.md) → bits-versus-quality curves on real streams |
| **Datacenter and edge serving** | Capacity beyond activity: more stored skill and context per unit of selected work, memory traffic and energy | Modelled 5.8× fewer bytes per character than Transformer-256×4 at better quality → trained sparse runtime with measured traffic and energy |
| **Computing substrates** | Globally clockless, memory-local event hardware; idle capacity at leakage power; learning on the chip itself | [Hardware thesis](investment/HARDWARE_THESIS.md) and cost model → FPGA and neuromorphic-silicon calibration, then an ASIC |
| **Self-designing models** | Data chooses operators, reception, depth and allocated structure; graphs grow and prune with their state | Learned event routes already adapt execution → structural growth with state migration |

The fronts share one core. Every improvement to temporal computation, route credit or persistent memory advances all of
them, and the platform's value spans models, learning rules, runtime software, communication and hardware IP. Each
front is also a market and a research program in its own right.

[Model family at a glance](report/model_family_overview.md) ·
[Interactive visual atlas](report/architecture_atlas.html) ·
[Definition and design rationale](report/model_family_design.md) ·
[Formal core](report/model_family_specification.md) ·
[How capabilities compose](report/model_family_composition.md) ·
[Claims and evidence](report/architecture_evidence.md) ·
[Application opportunities](report/model_family_opportunities.md)

## Typed learning: preserve meaning, then compose evidence

A category label, a temperature, a duration and a missing observation support different operations. Our typed interface applies meaningful comparisons before their outcomes enter addressed messages and persistent neural state. Computational races select evidence; counterfactual consequences teach the routes. Neural sums remain useful once representations share a declared latent space.

The proposed advantage combines tree-compatible conditional decisions with deep temporal memory and a common interface across modalities. Valid changes of units or category labels should preserve the evidence and downstream computation; training requires compatible parameter and optimizer coordinates too. The new theory derives these conditions and retains credit through computational time. **39 mathematical/interface contracts pass.** The integrated synthetic mixed-type fit already reaches 100% on 256 DEV rows from 64 FIT rows with fixed predicates. Learned predicate discovery, real tables versus trees and shared-core skill transfer are the next proofs.

Modern tabular neural models already use typed feature maps; comparison layers and neural/tree hybrids have precedents. Our contribution is the complete temporal/sparse composition and its tested consequences. See [typed semantics and race composition](experiments/theory/TYPED_SEMANTICS_AND_RACE_COMPOSITION_20261006.md).

## The core mechanisms

| Bet | Why it matters | Where it stands | Decisive next test |
|---|---|---|---|
| **Time performs computation** | Learned delays, races and phase rotation select and combine information. An exponential race picks route *i* with probability softmax(score)<sub>*i*</sub> exactly, and delay-coded aggregation reproduces softmax attention exactly over the delivered keys ([note 08](experiments/theory/08_vector_memory_and_deep_stacks.md), [note 151](experiments/theory/151_normalized_clock_noise_and_precision_credit.md)). Winner-only races deliver the sampled winner's value instead of the weighted average, skipping the value-aggregation half of attention arithmetic; the saving grows with context length. | Proven identities; integrated temporal models learn order, timing and rules (results below). | Integrated tokenized temporal/sparse models, on CPU, against public Transformers with equal complete fitting FLOPs and a reserved larger point ([crossover plan](experiments/LANGUAGE_CROSSOVER_PLAN.md)). |
| **Hard routes learn through counterfactual credit** | Discrete sparse routing is a long-standing training problem. Crediting unrealized alternatives trains it directly and deepens useful computation without dense inference. | Value-informed route credit: **2.507 → 2.371 bpc for 0.3% extra training work** (text8 10M). | Credit at larger pools and depth; sampled credit with flat cost in pool size. |
| **Capacity beyond activity** | Stored skills and context can grow while the work per input stays bounded. | Pool 2 → 4: **2.371 → 2.345 bpc at the same eight writes per character.** | Large-pool capacity curve with sampled credit ([capacity program](experiments/AWS_CAPACITY_PROGRAM.md)). |
| **One substrate for synchronous and asynchronous data** | Language, sampled signals and timestamped events can share one persistent world state, with transfer between the domains. | Shared interface and primitives implemented; each domain trained so far as a separate model. | Joint training on event logs plus text or sensor data, measuring transfer in both directions against single-domain models. |
| **Event-native inductive biases** | Interleaved, timestamped streams from many concurrent processes (industrial and IoT logs, IT operations, security, transactions) match the architecture: elapsed time changes memory, the next event is a first arrival, concurrent processes keep separate addressed state. | FAS v1 replicated win: native seeds6/7/8 score **0.600/0.591/0.586 vs 0.559** for the best of six generic identity-free controls at256 events. An oracle-assisted FIFO diagnostic reaches 0.755 using hidden training item identities; it is excluded from fair-reference verdicts. | Pre-registered FAS v2 with ambiguous identity ([protocol](experiments/FAS_V2_CONFIRMATORY_PROTOCOL.md)), then a real-data track. |
| **Clockless hardware and on-substrate learning** | Local memory access, no clock tree and idle capacity at leakage power; data movement dominates AI-chip energy. Delays, routes and local maps can learn in place. | Hardware cost model: the p96 native model moves **5.8× fewer bytes per character** than Transformer-256×4 at better quality (modelled). Online learning: **3.191 → 3.096 bpc** on a new stream. | Calibrate the cost model on FPGA or existing neuromorphic silicon ([hardware thesis](investment/HARDWARE_THESIS.md)). |

## Strongest results

Single training seed unless noted. Native compute is traced; reference compute is shape-estimated. Hardware figures are
modelled, not measured on silicon.

**Language (text8, characters; lower bits per character is better).**
- **Beats Transformer-256×4 (4 passes, 1.908 bpc) with 1.888 bpc at 0.40× its training compute and 0.18× its
  inference compute per character** (352 vs 889 TFLOPs; 1.32 vs 7.41 MFLOPs per character).
- **Beats the validation-selected tuned Transformers at both 10M budgets:** 1.888 vs 1.996 and 1.955 vs 2.215, at
  101–104% of their training compute. Strict at-or-under-budget runs are queued.
- **Scales with data:** the first 90M-character fit reaches **1.800 bpc** with 422K parameters at 0.97 PFLOPs.
- **90M near-budget loss (single seed):** the completed tuned Transformer-192×4 reaches **1.780 bpc at 946 TFLOPs**, versus the four-pass native **1.800 at 965 TFLOPs**. Same T256 test protocol; native final weights versus development-selected Transformer, with traced/extrapolated native work and shape-estimated Transformer work.
- Tuned LSTMs lead at these small budgets (1.826 vs 1.888 at 10M). LSTMs are the small-data diagnostic; the
  strategic contest is against Transformers at scale.

**Mechanisms in integrated models.**
- **Temporal order learning:** 99.73–99.93% across five runs after one pass over 2,000 examples; Transformer controls
  reach 33.25–40.80% after repeated fitting.
- **Timing as information:** 95.3% on paired timing against an exact 50% ceiling for any order-only model.
- **Shared rules, private memories:** 75.4% vs 44.1% order accuracy at 16 occupied sources, with **11.1× fewer
  parameters** and 6.6% less fitting work.
- **Joint addressed outcome races:** 0.102 bits / 96.1% on a distant relation vs 0.750 bits / 78.1% for the matched
  local model, for 2.56% extra fitting work ([note 68](experiments/theory/68_joint_addressed_outcome_races.md)).
- **Rule generalization:** 100% on all 3,440 unseen mod-17 triples with 69 learned phase scalars.
- **Length generalization:** 100% retrieval at four times the training context.
- **Speech (SHD):** 79.7% on 512 held-out development utterances from reserved speakers; official test untouched.

The report comes in three parts: [Part I — The science](report/I_SCIENCE.md) ([PDF](report/I_SCIENCE.pdf)): ambition,
model family, theory, evidence per front and the next decisive tests; [Part II — Methods and machinery](report/II_METHODS.md)
([PDF](report/II_METHODS.pdf)): protocols, accounting, contracts and drivers; [Part III — Experiment record](report/III_RECORD.md)
([PDF](report/III_RECORD.pdf)): every experiment entry, wins and losses alike. The complete combined record remains
[REPORT.md](REPORT.md) and [its PDF](report/sleeping_machines_status.pdf).

## Architecture and learning

The current integrated native core ([AddressedEventHeads](sleeping_machines/addressed_event_heads.py)) stacks event blocks
(depth 4–8 in current runs) of independent parallel race heads. At each depth a learned query races state-dependent keys of a
receiver pool selected by the observed stream address; the winning receiver mixes the incoming message with its
persistent rotating memory, updates it, and sends a gated content-bearing message with an arrival time. Unselected
receivers keep their state with no empty-tick evaluation. Content, keys, clocks, gates and retention all learn;
losing alternatives are evaluated in training to give the route its counterfactual teacher, and that work is charged.

The [shared model](experiments/SHARED_MODEL.md) provides the configurable mechanisms:

- **Sparse event carriers:** vector messages pass through depth without a hidden grid of empty time steps.
- **Local temporal memory:** causal state that retains and combines observations with linear scan work.
- **Hard races and learned delays:** the winning route emits its own payload and time; losers remain training
  alternatives.
- **Separate keys and values:** a key stream computes routing and clocks while value learning keeps that schedule.
- **Structured memories:** conditional evidence, relative pointers and periodic transformations where useful.
- **Silence-aware supervision:** next-event type and waiting-time likelihoods, including the information in silence.

Arrival times are elapsed intervals and local precedence, not a global tick: shifting the time origin leaves
predictions unchanged. The CPU reference serializes events on a common axis; a clockless ASIC executes the same
semantics natively. All candidate discovery, teaching and optimizer work is counted in the resource accounting.

## Read the theory and evidence

| Document | Purpose |
| --- | --- |
| [Report Part I — The science](report/I_SCIENCE.md) | Ambition, model family, theory, evidence per front, next decisive tests |
| [Part II — Methods](report/II_METHODS.md) · [Part III — Record](report/III_RECORD.md) · [Complete record](REPORT.md) | Protocols, accounting and implementation checks; every experiment entry; the combined record |
| [Research value plan](experiments/RESEARCH_VALUE_PLAN.md) | Prioritized uncertainty-reducing experiments, protocols and scaling gates |
| [Integrated language](experiments/INTEGRATED_LANGUAGE.md) | Native receiver, episodic KV architecture, evidence and queues |
| [Shared model](experiments/SHARED_MODEL.md) | Earlier cross-task implementation, adapters, contracts and retained evidence |
| [Theory index](experiments/THEORY.md) | Formal derivations organized by theme, with assumptions and proof scope |
| [Mathematical program](experiments/MATHEMATICAL_PROGRAM.md) | Open analytic problems and their decisive measurements |
| [Research roadmap](experiments/ROADMAP.md) | Next experiments and architectural priorities |
| [Language scaling protocol](experiments/LANGUAGE_SCALING_PROTOCOL.md) | Generic prediction, baseline matching and physical work measurements |
| [Parallel training protocol](experiments/PARALLEL_TRAINING_PROTOCOL.md) | Sequence parallelism, persistent inference and proposed architecture controls |
| [Online language protocol](experiments/ONLINE_LANGUAGE_PROTOCOL.md) | Test-time adaptation, memory and frontier comparison experiments |
| [Findings](experiments/FINDINGS.md) | Completed experiment history and detailed observations |
| [Historical manifesto](HISTORICAL_MOTIVATION_MANIFESTO.md) | Original motivation, exploratory ideas and early references |

Source lives in [sleeping_machines/](sleeping_machines/); experiment drivers,
executed queue commands and result records live in [experiments/](experiments/).
The implementation is a research reference with explicit numerical contracts and
source hashes.

## Repository layout

| Path | Contents |
| --- | --- |
| [sleeping_machines/](sleeping_machines/) | Event engine, shared models and work accounting |
| [experiments/](experiments/README.md) | Experiment drivers, research documents, queue commands and results |
| [report/](report/) | Report generators, figures and published PDF |
| [tests/](tests/) | Unit checks for the event engine and foundational experiment claims |
| [scripts/](scripts/) | Host provisioning and development helpers |
| [legacy/](legacy/) | Earlier implementations and historical results |

Local datasets, virtual environments, caches, runner logs and model checkpoints
stay outside version control. Queue commands and result records remain tracked.
The [experiment guide](experiments/README.md) explains how to find the commands
and evidence for a result.

For the foundational unit checks in a Python environment with
[requirements.txt](requirements.txt) installed:

```bash
python -m pytest tests/ -q
```

## Run experiments safely

Read [AGENTS.md](AGENTS.md) before launching work. Every experiment runs through
[run_safe.sh](experiments/queue/run_safe.sh), which holds a host-local lock,
limits threads and monitors memory. Run **one job at a time per host**, preserve
at least **8 GiB of available host memory**, and give changed settings a new
queue name and result tag. Inspect existing jobs, completed results and GPU
occupancy first.

The current runner expects this checkout at `/workspace` and its interpreter at
`/workspace/.venv-docker/bin/python`. The CPU environment uses
[requirements.txt](requirements.txt), PyTorch and `h5py`; task datasets are
supplied separately. The [installation runbook](AWS_EXPERIMENT_RUNBOOK.md) and
[bootstrap script](scripts/bootstrap_aws_experiments.sh) document a provisioned
experiment host. Choose resource limits from the actual host capacity.

For a small shared-model contract check in an already provisioned environment:

```bash
RUN_TAG="shared_contracts_$(date -u +%Y%m%dT%H%M%SZ)"
QUEUE="experiments/queue/${RUN_TAG}.txt"
printf '%s experiments/e120_shared_contracts.py --tag %s\n' "$RUN_TAG" "$RUN_TAG" > "$QUEUE"
MEM_CAP_KB=3600000 MEM_CAP_RSS_KB=2600000 MIN_AVAIL_MB=8192 JOB_TIMEOUT_S=600 \
  bash experiments/queue/run_safe.sh "$QUEUE"
```

These caps suit the bounded CPU check on a host with enough free memory. Training
runs need their own measured memory budget and timeout. Completed experiment
commands are preserved under `experiments/queue/`; use a new tag when reproducing
them. Logs and result records carry the executed settings and measured metrics.

## Citing

Sleeping Machines — Tero Keski-Valkama and Karoliina Salminen.

```bibtex
@article{keskival2021sleeping,
  title={Sleeping Machines},
  author={Keski-Valkama, Tero and Salminen, Karoliina},
  year={2021},
  doi={10.5281/zenodo.13207423}
}
```

[![DOI](https://zenodo.org/badge/342583401.svg)](https://zenodo.org/doi/10.5281/zenodo.13207423)
