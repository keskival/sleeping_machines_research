# Sleeping Machines

**Deep learning that computes with time.**

## The ambition

Sleeping Machines aims to make intelligence composable within **one trainable computing substrate**. Language and reasoning, perception and world models, embodied action, typed tables and interleaved process streams become different forms of experience available to the same learner. The ambition extends through continual learning, communication and self-design to the hardware that executes them.

An observation enters through an interface that respects its meaning: a token, a sensory event, a typed comparison or an action. It becomes an addressed message interacting with persistent memory. Learned delays and temporal races decide which evidence meets and which computation happens. Small messages carry new evidence into deeper state; separate keys and values distinguish where evidence goes from what it says. Counterfactual credit teaches the routes and writes that could have happened.

The larger promise is that these are **reusable learned computations**. Comparison, binding, retention, prediction and planning learned from one form of experience can support another through shared representations and end-to-end credit. Perception can inform reasoning; reasoning can guide action; process traces and tables can contribute to the same world model. Joint training must measure that transfer and the retention of existing skills.

This also connects the learner to its execution. Time performs computation, including useful sleeps; local events can advance it without a mandatory global clock. Inference and learning are designed around the same temporal, addressed substrate. Available memory and skill can grow beyond the activity recruited for one observation, with work tuned from small data and constrained devices to large datasets and datacenters. Candidate discovery and alternative learning work count in that budget.

The project is therefore a proposal for **a common foundation for learning, representing and executing intelligence**. The experiments test pieces of that construction. Mixed-type tabular learning is a decisive bridge: meaningful numeric, ordinal, categorical and missingness comparisons should feed the same temporal core used for sequences and asynchronous events. Strong results across small and large data would demonstrate the breadth of the unification. Current CPU implementations and separate-domain fits support specific mechanisms; joint transfer, scalable typed learning and clockless learning hardware have concrete tests ahead. See [the full vision](VISION.md).

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
| **Frontier language and reasoning** | A Transformer-capable function class: delay-coded aggregation reproduces softmax attention exactly; winner-only races skip value aggregation; persistent memory beyond a context window; learning at test time | At 10M characters the native model beats Transformers at 0.40× their training and 0.18× their inference compute → race attention inside a competent Transformer, scaled on GPU against equal-compute Transformers |
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

## The core mechanisms

| Bet | Why it matters | Where it stands | Decisive next test |
|---|---|---|---|
| **Time performs computation** | Learned delays, races and phase rotation select and combine information. An exponential race picks route *i* with probability softmax(score)<sub>*i*</sub> exactly, and delay-coded aggregation reproduces softmax attention exactly over the delivered keys ([note 08](experiments/theory/08_vector_memory_and_deep_stacks.md), [note 151](experiments/theory/151_normalized_clock_noise_and_precision_credit.md)). Winner-only races deliver the sampled winner's value instead of the weighted average, skipping the value-aggregation half of attention arithmetic; the saving grows with context length. | Proven identities; integrated temporal models learn order, timing and rules (results below). | Race attention inside a competent Transformer, on GPU, at equal FLOPs across a scaling curve ([crossover plan](experiments/LANGUAGE_CROSSOVER_PLAN.md)). |
| **Hard routes learn through counterfactual credit** | Discrete sparse routing is a long-standing training problem. Crediting unrealized alternatives trains it directly and deepens useful computation without dense inference. | Value-informed route credit: **2.507 → 2.371 bpc for 0.3% extra training work** (text8 10M). | Credit at larger pools and depth; sampled credit with flat cost in pool size. |
| **Capacity beyond activity** | Stored skills and context can grow while the work per input stays bounded. | Pool 2 → 4: **2.371 → 2.345 bpc at the same eight writes per character.** | Large-pool capacity curve with sampled credit ([capacity program](experiments/AWS_CAPACITY_PROGRAM.md)). |
| **One substrate for synchronous and asynchronous data** | Language, sampled signals and timestamped events can share one persistent world state, with transfer between the domains. | Shared interface and primitives implemented; each domain trained so far as a separate model. | Joint training on event logs plus text or sensor data, measuring transfer in both directions against single-domain models. |
| **Event-native inductive biases** | Interleaved, timestamped streams from many concurrent processes (industrial and IoT logs, IT operations, security, transactions) match the architecture: elapsed time changes memory, the next event is a first arrival, concurrent processes keep separate addressed state. | FAS v1: native beats all six generic identity-free controls at early detection (AUROC **0.600 vs 0.559** at 256 events, 3.2σ). An oracle-assisted FIFO diagnostic reaches 0.755 using hidden training item identities; it is excluded from fair-reference verdicts. | Pre-registered FAS v2 with ambiguous identity ([protocol](experiments/FAS_V2_CONFIRMATORY_PROTOCOL.md)), then a real-data track. |
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

Full evidence, protocols and every completed comparison, wins and losses alike: [REPORT.md](REPORT.md) and
[the PDF](report/sleeping_machines_status.pdf). Earlier README text: [archive](report/archive/README_20261005T150000Z_previous.md).

<!-- scoreboard:start -->
## Scoreboard (generated from result files; definitions in experiments/WIN_CRITERIA.md)

| Comparison | Reference | Ours | Verdict |
|---|---|---|---|
| 10M vs LSTM-256 at ≤ its training compute (20.3 TF) | 2.171 | 2.326 (p32/d8, skip2 + route credit; 14.2 TF) | Loss |
| 10M vs LSTM-256 at ≤ its inference compute (0.7 MF/pos) | 2.171 | 1.955 (p64/d4 + route credit, 4 passes; 0.6 MF/pos) | WIN |
| 10M vs Transformer-256x2 at ≤ its training compute (111.3 TF) | 2.427 | 1.955 (p64/d4 + route credit, 4 passes; 107.2 TF) | WIN |
| 10M vs Transformer-256x2 at ≤ its inference compute (3.7 MF/pos) | 2.427 | 1.888 (p96/d4 + route credit, 6 passes; 1.3 MF/pos) | WIN |
| 10M vs Transformer-256x4, 4 passes at ≤ its training compute (888.8 TF) | 1.908 | 1.888 (p96/d4 + route credit, 6 passes; 352.1 TF) | WIN |
| 10M vs Transformer-256x4, 4 passes at ≤ its inference compute (7.4 MF/pos) | 1.908 | 1.888 (p96/d4 + route credit, 6 passes; 1.3 MF/pos) | WIN |
| 10M vs LSTM-512, 6 passes at ≤ its training compute (432.6 TF) | 1.799 | 1.888 (p96/d4 + route credit, 6 passes; 352.1 TF) | Loss |
| 10M vs LSTM-512, 6 passes at ≤ its inference compute (2.4 MF/pos) | 1.799 | 1.888 (p96/d4 + route credit, 6 passes; 1.3 MF/pos) | Loss |
| 10M vs tuned dense at ≤ 352 TF (P0-6; 6 arms, validation-selected) | 1.825 (lstm512 4.5p lr0.002) | 1.888 (p96/d4 + route credit, 6 passes; 352 TF) | Loss |
| 10M vs tuned Transformers at ≤ 352 TF (P0-6; 4 arms, validation-selected) | 1.996 (tf128x4 5.4p lr0.003) | 1.888 (p96/d4 + route credit, 6 passes; 352 TF) | Better quality at 102% of its compute (not matched) |
| 10M vs tuned dense at ≤ 107 TF (P0-6; 4 arms, validation-selected) | 1.915 (lstm384 2.5p lr0.003) | 1.955 (p64/d4 + route credit, 4 passes; 107 TF) | Loss |
| 10M vs tuned Transformers at ≤ 107 TF (P0-6; 2 arms, validation-selected) | 2.215 (tf128x4 1.6p lr0.003) | 1.955 (p64/d4 + route credit, 4 passes; 107 TF) | Better quality at 104% of its compute (not matched) |
| 90M vs LSTM-512, 6 passes | 1.661 | 1.800 (p64/d4/pool2 + route credit; 4× less training compute) | Efficiency point; run queued |
| 90M vs Transformer-256x4, 4 passes | 1.604 | 1.800 (p64/d4/pool2 + route credit; 8× less training compute) | Efficiency point; run queued |
| NeuroBench Mackey-Glass (sMAPE; LSTM 13.37, ESN 14.79) | 13.37 | 14.84 (57.6 KB vs 490 KB) | Loss vs LSTM |
| NeuroBench primate reaching (R²; leaderboard 0.71 six-session) | 0.710 | 0.724 (validation-selected arm, development session indy_20170131_02, also one of the six official sessions; tinyRSNN .746 there) | Pending (six-session run; also report the five untouched sessions) |
| SHD (accuracy; best published 96.4%) | 96.4% | development queued | Pending |

Single seeds unless stated; same test interval; native/Transformer reset T256 windows, saved LSTM carries state. Identical-context LSTM rescoring is pending. Native compute traced, references shape-estimated or as published. Full report: REPORT.md; orders: experiments/PRODUCT_ORDERS.md.
<!-- scoreboard:end -->

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
| [Report and applications](REPORT.md) | Accessible overview, strongest results, potential and benchmark appendices |
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
