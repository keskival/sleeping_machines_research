# EIC Accelerator short proposal: draft (founder working copy, 7 October 2026)

Proposed acronym: **EVENTCORE**.
Title: *Event-native AI for operational event streams: early fault and anomaly detection at a fraction of the compute*.
Call: EIC Accelerator Open. Indicative request: grant only, €2.5M (70% of €3.57M TRL 6–8 costs).

Keywords to select at submission: artificial intelligence/machine learning; industrial IoT and predictive maintenance;
digital health / clinical decision support (secondary).

Before submitting:
- replace every bracketed item;
- cut the text to 12 pages in the portal questionnaire;
- check each number against its evidence file.

---

## 1. Excellence

### 1.1 The problem

Most data that runs factories, networks, logistics and hospitals arrives as **events**, not as tidy sequences: machine
signals, log lines, transactions, clinical measurements. These events are:
- **timestamped irregularly** (silence between events carries meaning);
- **interleaved** (many processes write into one log without process identifiers);
- often **partially missing**.

Today's AI is built for synchronous token sequences. A Transformer recomputes attention over the whole history for every
new event. Its cost per event grows with history length, it needs GPUs, and it must *learn* from data that time
matters and that concurrent processes are separate. Industrial anomaly detection therefore falls back on hand-built
rules or generic detectors. These miss early, subtle faults: a conveyor slowing by 20%, or a feeder retrying, hidden
among thousands of interleaved events.

### 1.2 The innovation

[Company] has built a new class of trainable neural model that computes **through time and sparse events**.
- **Memory follows elapsed time.** Persistent memories decay and rotate with the real time between events.
- **Races decide routing.** Each incoming event is routed by a *temporal race*: learned clocks compete, and the earliest
  determines which memory is updated and what is predicted.
- **The race is the prediction.** It *is* the mathematics of event data: the next event of a superposition of
  processes is the earliest among their pending events. The model therefore produces exact event likelihoods,
  including the evidence of *silence* (expected events that have not happened).
- **Sparse addressed state.** Each event updates a few addressed memories. Capacity grows without growing per-event
  computation, and inference cost per event stays constant however long the stream runs.

The construction includes attention and selective state-space models (Mamba) as exact special cases, verified
numerically. It is a superset of the current architecture classes, not a niche alternative.

### 1.3 Evidence (sealed test sets, published state of the art)

| Public benchmark | Our result | Best published | Compute |
|---|---|---|---|
| EasyTPP Taobao (e-commerce events) | 1.399 ± 0.003 nats/event | 1.318 (IFTPP) | 0.92× the leading model's per-event compute |
| EasyTPP Taxi (mobility events) | 0.525 ± 0.001; 5-model ensemble 0.536 | 0.522 (S2P2, NeurIPS 2025) | single model at **1/12** of the leading model's parameters and compute; ensemble at 0.41× |
| EasyTPP StackOverflow (activity events) | −2.1525 ± 0.0045 | −2.163 (S2P2) | **matched compute** (1.015×) |
| EasyTPP Retweet (social cascades) | −6.326 ± 0.001 | −6.348 (NHP); S2P2 −6.365 | **1/15** of the leading model's parameters and compute |
| EasyTPP Amazon (shopping events) | 0.803 ± 0.001 | 0.781 (S2P2) | 0.29× the leading model's per-event compute |
| P19 ICU sepsis prediction (irregular clinical records) | AUPRC 0.639 ± 0.039, AUROC 0.916 | AUPRC 0.583, AUROC 0.903 (MTM) | 62,681 parameters; without its temporal memory the same network falls to AUPRC 0.572 on every split |
| PAM wearable activity recognition | accuracy 0.978 ± 0.007, F1 0.980 | 0.975 / 0.976 (MTM) | 46,316 parameters vs MTM's 873K |
| TGB tgbn-trade (trade graphs, node affinity) | NDCG@10 0.868 ± 0.0005 | 0.863 (NAVIS, ICLR 2026) | 2,107 parameters, 4 CPU-minutes |
| TGB tgbl-wiki (link prediction) | MRR 0.835 ± 0.0003 | 0.827 (TPNet) | 7,995 parameters, 4.5 ms per query |

All five EasyTPP datasets, P19, PAM and two Temporal Graph Benchmark leaderboards (tgbn-trade, tgbl-wiki) are won. A single configuration with no per-dataset tuning also wins all five EasyTPP datasets under a pre-registered protocol (24 of 25 seeds ahead). All results use five seeds or five official splits, test sets scored once,
and pre-registered protocols. The Taxi result was reproduced from scratch on separate hardware (0.5252 vs 0.5250), and a
one-command kit is available for third-party reruns. The evidence files are in the diligence pack.

In development on new event domains (9 October 2026, development data): on the Temporal Graph Benchmark's node-affinity
task (tgbn-trade), our model reaches validation NDCG@10 0.875 against 0.860 for the current leader NAVIS, ahead in every
validation year with 2,107 parameters; all three pre-registered sealed test seeds are ahead (0.868 ± 0.0005 vs 0.863): won. On TGB link prediction (tgbl-wiki, evaluation reproduced exactly) exact per-event state gives validation MRR 0.852
against 0.842 for the leader TPNet, and all three sealed seeds score 0.835 on test against its 0.827: won. A second neural-TPP benchmark (35 pre-registered runs with
the frozen one-configuration model) is won on Wikipedia (total NLL −240.42 vs the bar −122.62; marks 28.49 vs 144.79: the addressed mark memory copies pages never seen in training) and lost on MOOC (−226.85 vs −239.7), Stack Overflow (12.714 vs 11.9; time NLL −91.598 beats −91.1, marks trail 104.312 vs 103.0), Github (−198.5 vs −272.9) and MIMIC2 (7.01 vs 2.42). Retweets and LastFM are pending.

**Properly tokenized language (first evidence, research track).** On GPT-2-tokenized FineWeb the same family's
temporal-memory token model scores 6.009 ± 0.010 nats per token at 1M training tokens (3 seeds) against a Kneser–Ney
trigram's 6.537, and 5.510 against 6.100 at 4M tokens: the lead grows with data.

**Anonymous interleaved industrial logs.**
- Benchmark: FAS, the founder's assembly-line simulator.
- Result: the model beats every generic detector at early prefixes across three seeds.
- Next: a harder sealed benchmark with ambiguous interleaving and small LSTM/Transformer references is in development.

### 1.4 Novelty and state of the art

- **Transformer and SSM sequence models** (THP, S2P2, MTM and similar) require dense recomputation, or assume a single
  synchronous sequence.
- **Spiking and event-driven networks** have so far reported parity or gaps against conventional models.
- **Classical detectors** cannot separate interleaved processes without identifiers.

Our model is, to our knowledge, the first event-driven architecture to beat the published state of the art on public
event-stream leaderboards. It does so at equal or much lower compute, with the same core across event prediction and
clinical classification.

### 1.5 Technology readiness

- **TRL 5 [to be confirmed by the pilot]:** components validated on real-world event data from the target domains
  (e-commerce, mobility, online activity, ICU records) under sealed protocols, with measured compute.
- **Project goal, TRL 6 → 8:** pilots on design partners' operational logs, a productised CPU runtime, and a
  validated deployment.
- [Insert pilot status: partner, data, metric, first result.]

### 1.6 Intellectual property

- The core constructions are the company's own work. [Priority patent applications filed on (date): (titles).]
- An IP protection plan defines what to patent and what to keep as trade secrets.
- The founder has prior invention experience: named inventor on 35 published US/EP patent documents (former employer;
  not company assets).

## 2. Impact

### 2.1 Market and first customers

**Beachhead:** early fault and anomaly detection on operational event logs.
- **Customers:** manufacturers and industrial-IoT platform providers; IT operations and security teams.
- **Selling points:** detection lead time; false-alarm rate; deployment on existing CPUs (no GPU fleet, on-premises
  for data-sensitive sites).

**Second market:** clinical early-warning on irregular patient records, where the P19 result shows quality and
regulation (AI Act, MDR) shapes the timeline.

[Insert market sizing for the beachhead with the customer denominator (sites × event volume × price), sources.]

### 2.2 Business model

- Model and runtime licences priced per monitored stream or site, plus pilot and integration fees.
- Later: OEM licensing to IoT and observability platforms.
- CPU-efficient inference keeps gross margins high and makes on-premises deployment easy.

### 2.3 Why now, and why Europe

- Event data are growing faster than the GPU compute to process them.
- Europe's industrial base runs on event-generating equipment.
- An architecture that needs no GPU fleet and exposes calibrated, interpretable alarms supports European technological
  sovereignty in industrial AI.

### 2.4 Broader impact

The same substrate extends to language, world models and hardware (clockless, memory-local execution). These are
long-term upside, not part of this project.

## 3. Level of risk, implementation and need for Union support

### 3.1 Team

- **Founder and CEO:** Tero Keski-Valkama.
  - Lead ML and AI engineer at HERE; staff engineer on multimodal foundation models.
  - Software architect.
  - Named inventor on 35 published US/EP patent documents.
  - Built the core architecture, its theory and the benchmark programme.
- **Planned hires (round and grant):** learning research lead, runtime engineer, commercial lead, evaluation engineer.
  [Names or committed candidates and advisers.]
- **Governance and gender balance:** [board or advisory plan; hiring plan with gender-balance targets].

### 3.2 Financing and traction

- Planned private round: €3M [status: LOIs, commitments].
- Grant-only request: the round finances the 30% co-funding and the scaling beyond the grant.
- [National funding: CDTI NEOTEC / ENISA status.]

### 3.3 Risks and mitigation

| Risk | Mitigation |
|---|---|
| Technology transfer from benchmarks to customer data | Pilots with pre-registered metrics; sealed evaluation discipline already in place |
| Market adoption | CPU deployment, measurable detection lead time, design partners from the start |
| Team scale | Funded hiring plan; advisers |
| Competition from large-model vendors | Structural advantage (constant per-event cost, silence-aware likelihood); IP filings |

### 3.4 Why EU support is needed

- Deep-tech, long validation cycles and a new architecture class: private investors under-fund this before market proof.
- EIC support turns a validated technology into deployed European industrial AI, faster than private capital alone.
