# Sleeping Machines: the investment case

Sleeping Machines aims to be a **universal trainable computing substrate**: one composable architecture for language and reasoning, multimodal world models, embodiment, typed tabular data, synchronous samples, asynchronous events, anonymous interleaved process traces, token sequences, continual learning, communication, self-design and hardware.

**Demonstrating compositional scaling.** The completed causal race language result establishes a useful prediction mechanism: **5.4665 versus 5.8414** development NLL on the same 4M-token recipe. The next construction lets higher stages query several persistent lower memories, learn joint refinements, and send delayed task credit back to the actual memory producers. Bounded child-credit replay is implemented; its bounded gradient, prediction-parity, work-cap and acquired-learner recovery gate passes; the full-depth sampled extension awaits acquired-model validation. The investment milestone is better held-out prediction and reusable relations from deeper refinement at disclosed learning work, followed by useful capacity with measured activity, discovery and traffic. These are the next proofs, not completed scaling results. [Two-page demonstration](sleeping_machines_composition_demonstration.pdf); [source-bound evidence](composition_demonstration_evidence.json).

The ambition combines distinct design axes: **computation through sleeps and temporal races; globally clockless event-driven execution; learning in the same temporal, addressed and asynchronous substrate as inference; capacity beyond selected activity; and tunable work for small-data, low-resource and large-scale settings**. Data modality is a separate axis from each of these. Scaling advantages and economical intelligence from sensors to datacenters are targets to measure. See [the full vision](../VISION.md).

**A learning architecture and computing substrate for intelligence everywhere.**

Prepared 3 October; revised 16 October 2026. Investment thesis and proposed commercial strategy;
completed research evidence is identified separately from product ambitions.

Current terms: **€3M raise at €50M priced pre-money**, with €100M as a separate
stretch scenario. Read the [current valuation rationale](VALUATION_RATIONALE.md)
and [application opportunity register](../report/model_family_opportunities.md).
**Progress toward deeper reusable representations.** Keyed language development loss improves **5.9572→5.8987→5.8741 nats/token** at 2/4/8 layers on the 65528 target slice and 1.05M-token corpus, one seed and different total work. These depth results come from the dense keyed hybrid, now a labelled diagnostic control (founder direction 16 October: build our own models); in one pass over 4M fresh tokens it favors eight over two layers (**5.8414 versus 5.8648**), and 16M fresh tokens take it to **5.2183** at eight layers (5.2699 at four). The native causal-race member is measured against these numbers. The earlier native FIT probe gives **1.9258 NLL** after readout calibration on the same 12 FIT prefixes (learnability evidence). Native local grounding improves calibrated root geometry by **34.6%**, while whole-label transfer still selects the acquired actor. The new **8 level/32 stage** construction keeps exact role-bearing child-memory versions accessible as lower memories evolve; its acquired actor outputs, schedules and parameter gradients match bit-for-bit. A native race head pools the two actual root parts with live credit;**16 numerical contracts** pass. These are demonstrated representation/access mechanisms; stable semantic identity and relation transfer are the next learning test. [Bound evidence](bound_hierarchy_evidence.json).

**Composition test: more training did not teach whole-object structure.** On a synthetic task the native hierarchy must tell apart objects built from the same parts in different arrangements. It recognises every part (512 of 512) but still classifies whole objects at chance after 32 times more training (2,048 extra updates on 2,048 new scenes): 50% accuracy and NLL 0.693, the value of a coin flip, on held-out and fresh scenes alike. It does not fit the training arrangements either (53%), so the limit is how the model learns to bind parts into wholes, not data or training length; the next construction targets that binding step. The earlier, shorter-trained model remains slightly better (NLL 0.687). Source: `investment/composition_scale_evidence.json`.

**Upper sensitivity is acquired; the required relation is not reliably aligned.** A frozen audit of eight TRAIN pairs at checkpoints 128/512 finds 51.62× greater root-feature swap sensitivity and 27.73× greater output contrast, with nonzero upper relation credit. Correct pair rankings are 5/8 then 4/8. This rules out complete upper insensitivity and wholly cancelled credit on these samples; it does not identify a unique cause. Formal analysis shows why separate readable child features can remain insufficient for this balanced whole-label task. Source:`investment/composition_credit_evidence.json`.

**Our own race model beats the dense hybrid.** The causal event race model predicts each token through a race among its candidate causes: copying or continuing earlier text, dormant concept units woken by the context, and a background word race (theory note 169). On the same 4M fresh FineWeb tokens, in the same order and with the same schedule, it scores **5.4665** on the flagship development protocol against **5.8414** for the eight-layer dense hybrid, and **5.3094** against 5.6955 on the 65,528-target slice: 0.38–0.39 nats better while using 3.3 times less training compute per token (7.8M versus 25.9M FLOPs, counting the exploration term and the optimizer) and 3.2 times less CPU time. Its race causes deliver 0.45 nats of the prediction beyond its background route. One seed, development slices. This member still builds its state with two temporal-memory layers and has one race level; the compositional hierarchy of races is the next test. Source: `experiments/results/token_language/curie_r1_cerl_4M_s0_20261016_v1.json`.

**Native continuation.** The fully native eight-layer learner completes 65,536 chronological GPT-2/FineWeb targets: selection NLL **7.9307 to 6.9981**, final flagship DEV **7.0473** (8,192 targets, 1,024-token windows; one seed). Native receivers, persistent indexed memory and causal messages learn without recurrent BPTT or an SSM/attention/MLP carrier. The acquired learner now continues toward 4M cumulative targets with version-bound lower-memory queries, joint composition and bounded delayed credit into the actual producer. Learned document relevance is validated with zero-initialized repeat/continuation biases; history stays available. Shared weights, Adam state, numerical memory and RNGs are preserved through explicit migrations. Conditional local derivatives and checkpoint recovery pass; changing-parameter credit remains approximate and the optimizer synchronous. The fully native deep comparison is in development. Source: `investment/local_native_language_evidence.json`.

**Completed native language composition at 16M tokens.** Four-level residual memory scores **5.0433 NLL**, improving the native causal-race member's **5.0634** by **0.0201 nats per token**. The saved sixteen-layer dense hybrid scores **5.2034**. One seed, one pass over 16M FineWeb GPT-2 tokens; the flagship evaluation uses 8,192 development targets in 1,024-token windows. The separate 65,528-target slice improves from 5.0282 to 5.0244. This supports useful composition in the current-slot memory variant; old-slot rewrites are disabled, and it does not establish fully asynchronous deep learning or a public leaderboard win. Complete learning arithmetic is not audited. [Bound evidence](native_composition_16m_evidence.json).

**Old-memory refinement makes depth pay more.** When the four residual-memory levels may also write corrections into *older* memory slots (later context refining earlier memories, which a Transformer's frozen key/value cache cannot do), the same 16M-token run scores **4.9901** on the flagship development protocol, 0.073 nats better than the single-level race model (5.0634) and 0.21 better than the sixteen-layer dense hybrid (5.2034), with 15.5M versus 32.2M training FLOPs per token. Selection (5.1906 versus 5.2826) and held-out training loss (5.0426 versus 5.1204) improve similarly. On the 65,528-target slice, whose lanes are 8,192 tokens long against 256 in training, it is worse (5.0874 versus 5.0282): corrections accumulate over 32 times more history than training ever showed. A learned decay of old corrections, which bounds that sum at any length, is the fix now under test. One seed. Source: `experiments/results/token_language/curie_r1_rmr_resid_K4_16M_s0_20261016_v1.json`.

**Observed ownership repair, 16 October.** A label-free spatial-support audit finds that **25%** of inferred groups never become pure complete memories under the old dyadic supports, on both 128 TRAIN and 64 DEV scenes. The new static observed-ownership candidate composes all pixels of a group in one native part memory before upper composition. Its full 32 event TRAIN gate produces **8 complete parts×4 pixels**, retains 8 levels/32 native stages and 255394 actor parameters, and gives nonzero finite content gradients at every level. Training/evaluation forward equality passes;19 combined contracts pass. No optimizer fit, DEV class improvement or TEST result is claimed. Grouping is an explicit TRAIN-fitted static spatial policy; streaming ownership revision and stable semantic identity remain distinct capabilities. [Theory](../experiments/theory/OBSERVED_PART_OWNERSHIP.md); [evidence](bound_hierarchy_evidence.json).

**Composition engineering milestone.** Persistent memories now integrate local evidence and emit composed views through native races. On one paired mechanism fixture with the same acquired parameters, coalescing reduces delivered messages **349→125** and native candidate scores **11,168→4,000**, while timer scores rise **2→188**. A separate repair keeps pixel memories world-indexed and composes upper state in an observed scene frame: pure translation gives exactly identical upper features, without new parameters or sensor replay. Combined translation/timing root-feature displacement falls **0.4162→0.00316** on the controlled fixture. These are mechanism and resource results; prediction quality differs between constructions and complete arithmetic/energy are unaudited. The first synthetic held-out composition fit scored **50% accuracy** and failed its transfer gate. The repaired 128-update model also scores 50% DEV accuracy after TRAIN-only calibration; the coordinate contract passes, but learned part-combination transfer still requires a representation/credit repair. TEST remains closed. [Evidence](hierarchy_development_evidence.json).


## Online market forecasting: confirmed on a new month

**A preregistered out-of-sample win while every model keeps learning.** On Binance BTCUSDT public trades, our native temporal learner, an online LSTM and two online Transformers each adapt their weights after every 32 revealed trades and are scored before any outcome can change them. All model choices were fixed from August evidence and committed before the 1 October data were downloaded; each learner carried its full August memory and optimizer state across the five-week gap.

| Model (all learn online) | 1 Oct loss, 65,536 trades | 25 Aug loss, 65,536 trades | Online arithmetic |
| --- | --- | --- | --- |
| **Sleeping Machines** | **8.0785** | **5.1759** | **8.5 GFLOPs** |
| LSTM | 8.1027 | 5.1967 | 9.5 GFLOPs |
| Transformer, cached history | 8.1792 | 5.2804 | 124.7 GFLOPs (14.7×) |
| Transformer, re-encoded history | 8.1764 | 5.2797 | 7,963 GFLOPs (936×) |

Lower loss is better (nats per trade: next-trade timing, aggressor side and price direction). Our model leads in **all 16 of 16** consecutive 4,096-trade windows of the new month, with **14.7× less arithmetic** than the cached Transformer and **16× less retained state** (0.146 MB of weights plus prediction memory versus 2.334 MB). This is the capability a deployed forecaster needs: keep learning from a changing stream at low cost. Single seed; the sealed September test stays unopened until seed replication. Next: return-distribution forecasts of the price itself (10 s, 1 min, 5 min), now in development. No trading-profit or measured chip-energy claim. [Source-bound evidence](online_market_evidence.json).

**Trade-event acquisition now reproduces across three seeds.** Native mean development loss is **6.8842**, versus LSTM **6.8996** and Transformer **6.9686**; native leads both references at every paired seed. All nine acquired fits are complete. These are fixed-weight August development results. The October online confirmation above remains single seed. Next: reuse these learners for online August/October replication with unchanged settings and causal updates; October is an already accessed day, so this is replication rather than a new prospective test. [Source-bound three-seed evidence](market_replication_evidence.json).

## Industrial fault detection: sealed win on interleaved factory logs

**Predictive maintenance on anonymous machine logs.** In our FAS benchmark (assembly-line simulator, two interleaved lines of 30 items, no item labels), a failing machine shows only as production steps that slowly take longer. Our model learns each step's duration from normal logs, binds every event to the earlier event it most likely continues through a race of delayed clocks, and keeps a separate running anomaly score per step. The full plan was committed before the test data were generated; each model was scored once.

| Detector (2,000 normal + 2,000 faulty runs) | Fault-detection AUROC |
| --- | --- |
| **Sleeping Machines** (three independent fits) | **0.801** |
| Time-encoded Transformer (three seeds) | 0.704 |
| Best of nine classical detectors | 0.676 |
| LSTM (three seeds) | 0.667 |

**Ahead by 0.097 AUROC (95% interval 0.088 to 0.121), every fit above every reference** (wear faults 0.772 vs 0.650; retry faults 0.829 vs 0.758). No item identities are used and the model's knowledge is fixed during scoring. [Source-bound evidence](fas_evidence.json).

## Language: addressed memory leads

**Addressed memory is our strongest language lever.** On the same 8,192 FineWeb development targets (GPT-2 tokens, 1,024-token history), our keyed temporal-memory model, whose events write keyed messages that later tokens read directly, scores **5.601 and 5.624 nats/token** (two seeds) against **5.978** for the earlier flagship; removing the keyed read costs 0.10 at 1M tokens. A 6,248-parameter addressed episodic memory with read-time credit lifts the flagship itself to **5.634 (+0.345)**, while an identical plain continuation does not improve (native readout alone: 5.973); the learner's use of history older than 64 tokens rises from 0.004–0.008 to **0.227 nats**, and it learns on its own to discount memories from earlier documents. The same principle, explicit addressed access to the right past events, produced the sealed fault-detection win above. [Source-bound evidence](language_memory_evidence.json).

**Continuation uses the memory gains inside the native model.** Finish existing hybrid diagnostic acquisitions and reuse their scores as targets. The native eight-layer construction restores sparse receiver writes, computational clocks, counterfactual route credit and local eligibility. Its next admission requires persistent indexed episodic history and complete work accounting. The previous serial native stack failed to compose depth usefully; the new composition gives each layer its own horizontal memory and direct predictive-cause credit.

**Price forecasting is in development.** Native leads the fixed-window development comparison (29.5089 summed loss versus LSTM 29.6516 and linear Student-t 29.6998). On the corrected full-day online stream it scores 30.6203 versus linear Student-t 30.5936: the classical reference leads overall, while native leads at 10 seconds. Next: causal streaming learning with non-overlapping delayed labels, followed by a preregistered fresh October price test. The confirmed trade-event result above has a different target and scoring protocol.

**Learning more efficiently.** Sparse counterfactual teaching improves development NLL from **6.62066 to 6.52787 nats/token** while cutting complete CPU training time from **9,129 to 7,734 seconds (15.3%)** at 8.39M GPT-2 FineWeb targets, all core, discovery, backward, replay and optimizer time included. [Source-bound evidence](language_learning_evidence.json).

## Forward computation teaches credit — current status and potential

Prospective real-token calibration reduces credit error **98.4% versus static
credit and 97.4% versus reference-only credit**, improving all eight fresh
TRAIN-tail prefixes. Shared forward-derived geometry tracks alternative
parameter and Adam-moment trajectories. A separate matched four-prefix,
three-repeat acquisition pipeline is **2.09x faster** than exhaustive alternative
learning, with **93.5% fewer supported arithmetic operations**, including context,
route discovery and exact factual/reference losses. In a fixed connected
37,377-parameter controller, conditional action-gradient noise is **77.8% lower**
than static credit, with exact mean-gradient parity.

Scope: component evidence on a fixed two-layer pretrained native model,
50,257 GPT-2 token classes, 64-dimensional lexical learners and 65 candidate
routes. Accuracy uses eight new 48-target prefixes; execution and controller
noise use four earlier prefixes. Source pretraining, oracle acquisition and
numerical gates are additional. Supported arithmetic omits unsupported work
included in wall. Early small route-policy fits have no confirmed held-out
prediction gain. [Completed evidence and formulas](forward_credit_evidence_20261011.json).

The opportunity is to make continual adaptation economical: models that learn
as data changes, capable local machines and industrial event systems, and
reusable learned credit across the general-purpose substrate. The sparse
asynchronous hardware path targets memory, communication and energy budgets.
These are conditional platform implications, not measured customer demand or
industry-wide savings. The next gate is **better held-out prediction per total
learning cost**, followed by larger data/depth and online application confirmation.

## Wins and advances at a glance

**13 public benchmark wins** against the best published single-model results on official dataset/protocol entries. Across these winning entries, 53 of 61 seed/split scores are ahead; wins compare dataset means, not every individual run. The four added Bosser–Ben Taieb wins are Wikipedia, Github, Stack Overflow and LastFM. Wikipedia/Github also beat stronger composite targets; Stack Overflow/LastFM beat the best single-model totals. B4 uses recorded-data density scoring; these are quality wins, with complete work unaudited. [Source-bound headline ledger](public_benchmark_headlines.json).


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
| **B4 Wikipedia, developed** | total NLL −258.76 ±37.73 SE | best single −2.67; composite −122.62 | Five official splits; beats composite; full work unaudited |
| **B4 Github, developed** | total NLL −284.47 ±54.29 SE | best single −269.70; composite −272.90 | Five official splits; beats composite; full work unaudited |
| **B4 Stack Overflow, developed** | total NLL 11.9956 ±0.6476 SE | best single 12.10; composite 11.90 | Five official splits; single-model win, composite missed |
| **B4 LastFM, frozen transfer** | total NLL −774.64 ±48.38 SE | best single −697.50; composite −849.65 | Five official splits; single-model win, composite missed |

**Rigor:** official scorers, validation selection and once-only test scoring; the original nine-entry record has recording-grid audits. B4 results use recorded-density scoring, with composite and single-model targets distinguished. A one-command kit supports third-party reproduction. **One configuration wins all five:** a single configuration with no per-dataset tuning is ahead of the best
published result on all five EasyTPP datasets under a pre-registered protocol (24 of 25 seeds ahead): Taobao 1.397,
Taxi 0.526, StackOverflow −2.144, Retweet −6.324, Amazon 0.802.

**Advances beyond the leaderboards:**
- **Tokenized language:** 6.009 ± 0.010 nats per token vs a Kneser–Ney trigram's 6.537 on GPT-2-tokenized FineWeb (3 seeds);
  at 4× the data 5.510 vs 6.100, so the lead grows with data.
- **Binding and recall:** associative recall across irregular gaps 97.5% (3 seeds) vs a 14% baseline, 91.6% at twice the
  training length; **learning through the race alone** (no backpropagation into the network) reaches 77%.
- **Learning without a backward sweep or weight transport:** a race-of-clocks event model learns from each event as it
  arrives, with exact memory credit carried forward (no stored history). On Taxi development data at the same 20 data
  passes it scores 0.4805 nats/event vs 0.4557 for backpropagation through time (3 seeds).
  On Taxi it matches backpropagation run to convergence (0.4774 after 100 passes) with one fifth of
  the data passes. Across three datasets it leads at equal passes on Taxi (+0.025) and StackOverflow (+0.046) and is level
  on Amazon (+0.002), while backpropagation run to 100 passes ends ahead on StackOverflow and Amazon. Separate backward weights
  trained by the forward pass's own updates align to the exact dual (cosine 1.000) and learn as well as shared weights
  (0.4715 vs 0.4736); fixed random feedback fails (0.17). This is the learning half of clockless hardware.
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

**New event domains:**
- **Temporal Graph Benchmark, node affinity: won** (table). Pre-registered sealed test, 3 seeds: 0.8683 / 0.8674 /
  0.8683, every seed above NAVIS's 0.863; persistent forecast 0.855. Validation 0.874 vs 0.860 had predicted it.
- **Temporal Graph Benchmark, link prediction: won** (table). Exact per-event state lifted our model from 0.776 to
  0.852 validation MRR (theory note 159 predicted at least +0.03); three pre-registered sealed seeds score 0.8353 /
  0.8350 / 0.8356 on test, every one above TPNet's 0.827, with 7,995 parameters. On tgbl-review a training-free decayed
  event state already reaches 0.344 validation MRR, above TGN, TGAT and TNCN (0.313–0.325); GraphMixer leads at 0.428.
- **Second neural-TPP benchmark (Bosser & Ben Taieb): four public-reference wins.** Developed Wikipedia total NLL **−258.76 ±37.73 SE** beats composite **−122.62**;developed Github **−284.47 ±54.29 SE** beats composite **−272.90**. Developed Stack Overflow **11.9956 ±0.6476 SE** beats the best single **12.10**, while missing composite 11.90. Frozen-transfer LastFM **−774.64 ±48.38 SE** beats best single **−697.50**, while missing composite−849.65. Five official splits, recorded-density scoring; developed variants selected by aggregate validation before once-only TEST. These add four dataset/protocol wins to the original nine. Frozen-configuration composite failures remain separately labelled in the full record. No B4 trainer remains active.

**Why the developed models improve.** Persistent addressed mark memory is combined with a causal episodic pointer race that can copy a recent event or its observed successor. Numerically stable clock-spread learning and bounded row microbatches preserve training on complete long histories. Earlier completed Github development evidence isolates 13.822 mark nats/sequence of direct memory-read benefit on 28 of 34 DEV sequences, with timing unchanged. Complete recorded-density wins and this mechanism evidence have different scopes. MOOC, MIMIC2 and Retweets remain development targets. [Current B4 evidence](../experiments/tpp/B4_DEVELOPED_CURRENT.md).

**Where we are not yet ahead:** large-scale language (our best token model reaches 5.60 nats per token after 4M training
tokens on single-thread CPUs; depth scaling is under test), P12 mortality (0.871 vs 0.880 AUROC) and real-world tables.

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
| Event-driven, globally clockless, memory-local hardware | Energy from sparse local memory access (data movement dominates AI-chip energy) plus no clock tree and leakage-only idle capacity; average-case latency; thermal headroom for memory-on-logic stacking; a competitive model for the existing neuromorphic chip class ([hardware thesis](HARDWARE_THESIS.md)) | Mechanisms and energy accounting derived; no fabricated Sleeping Machines chip or measured joule advantage; a fully local learning rule (per-event traces + mutually trained feedback) works in software on Taxi development data, on-chip execution open |
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

The program also spans event speech, event vision, tabular tasks, interleaved
industrial event logs and temporal prediction, each at its own maturity; the
report lists every win and loss per domain. Invalid target-dependent mixtures
are quarantined and excluded from this investment case.

Completed parents and exact values are indexed by the accompanying evidence
manifest. [Research status](../report/sleeping_machines_status.pdf)

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
2024; Innatera reported a EUR 15M Series A for neuromorphic edge technology in
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

**Why independent timing matters.** A shared clock makes connected units follow a common timing schedule. Local clocks let fast regions run at a higher frequency than a chip-wide schedule would permit, while slower regions proceed at their own pace. Fully clockless units can advance when their work finishes, without waiting for the next shared tick. This creates three opportunities: less energy distributing a clock across the chip, less waiting for coordination, and faster local progress. Communication, memory and completion-signalling costs determine the resulting system benefit. Our local learning control removes one coordination requirement and slightly improves prediction in the completed software comparison; chip speed and energy gains will be measured on hardware. These are established hardware mechanisms, not proprietary claims about clocking. [ETH Zürich's GALS research](https://iis-people.ee.ethz.ch/~kgf/acacia/acacia.html), [DTU's asynchronous circuit course](https://www.imm.dtu.dk/courses/02204/).

## Evidence and commercial diligence

[Evidence manifest](evidence_20261003.json) records completed-parent hashes,
precise metrics, arithmetic conventions and primary market sources accessed
on 3 October 2026. [Technical milestones](../experiments/DATACENTER_VALUE_MILESTONES.md),
[hardware thesis](../experiments/HARDWARE_VALUE_PROPOSITION.md) and
[frontier protocol](../experiments/FRONTIER_COMPUTE_PROTOCOL.md) supply the next
validation boundaries. No customer commitments, revenue, patent grants,
fabricated silicon or current frontier-scale capability are represented as
existing assets in this case.


Training-locality engineering has a reproducible measurement: fixed nonuniform localcaps give heldoutTRAIN RMSgradient1.116 versusglobal 0.995, with calculatedAdam displacement 0.928 versus 0.916 (+1.28%). Learning units can limit their updates locally without all stopping to coordinate this step, supporting learning without a global clock; completed fitted local/global NLL is 6.22624/6.23119 (gain 0.00495), at near-equal measured single-thread CPU process/scoring time (+0.41%, including calibration); distributed synchronization costs are not measured. One paired seed; actual asynchronous speed/energy is the next measurement. The stronger richer-readout member improves paired 8M DEV 6.160183→6.099337 for 11.94% more training time. Single-seed development and frozen-learner geometry, respectively; [source-bound evidence](locality_scaling_evidence.json).
