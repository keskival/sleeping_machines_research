# Valuation rationale: ambition, platform potential and execution

4 October 2026, revised 6 October · Private founder/investor discussion · Not an independent appraisal.

**Current proposal: raise €3M at €50M priced pre-money.** The €100M case remains
a stretch scenario. Since the proposal was set, the family has its first public-leaderboard wins (EasyTPP Taobao, Taxi
and StackOverflow, 6 October, §3). These make €50M materially more defensible; §5 states what would carry €100M. The breadth of the family is the rationale for seeking a
platform premium. The proposal funds the next decisive evidence,
team and execution capacity rather than requiring that the entire vision be
finished before investment.

## 1. The ambition being financed

Sleeping Machines aims to be the next general-purpose substrate for machine
intelligence: a learning and thinking substrate that connects content,
computational time, persistent evidence, selective work and learning. One
architecture, learning rule and execution model spans frontier language and
reasoning, multimodal world models, embodied intelligence and robotics,
event-native analytics, continual on-device learning, adaptive communication
and self-designing models. The common event/state interface permits
heterogeneous local programs, dense and selective regions, synchronous and
asynchronous schedules, and streamed or batched learning; each member fixes its
operators and contracts.

The largest outcome would be a widely adopted model/learning/runtime layer,
with compatible communication and compute substrates from datacenters to mobile
and robotics. Integrated embodied and cognitive learning expands the core
model's reach, with motor experience and reasoning teaching shared
representations. General intelligence is the overarching aspiration; the
financial model does not price it.

The [opportunity register](../report/model_family_opportunities.md) gives this
breadth concrete mechanisms and proof conditions. Asynchronous codecs,
irregular-stream ingestion, task-oriented communication, learned memory
policies and embodied transfer are further uses of the same core, each with its
evidence stage and first proof. Familiar neural methods address portions of
these problems; the thesis is the integrated construction and its
quality/resource behavior.

## 2. Why the platform could earn a premium

| Potential asset | Route to economic value | What must become true |
| --- | --- | --- |
| Models and learning policies | Better useful quality per total learning/inference resource; persistent adaptation | Useful depth, memory credit, scalable discovery and repeated relevant comparisons |
| Runtime and deployment software | Lower full service cost at required quality, latency and reliability | Trained sparse parity and measured traffic, residency, throughput and energy |
| Adaptive communications/codecs | Fewer transmitted bits and less recomputation for reconstruction or a declared task | A real independently decodable wire format and a superior quality/rate/work/latency curve |
| Embodied generalist learning | Share useful representations across sensing, acting, language and reasoning | Joint learning and separately measured transfer in both directions, including held-out tasks |
| Compute/hardware IP | Co-design local state, scheduling and communication for target platforms | Precision, scheduling, memory/interconnect and physical energy economics demonstrated |
| Event-native vertical models (inductive biases, existing hardware) | Better detection and forecasting on asynchronous, timestamped, interleaved event data than dense models, sold as models/runtime on ordinary CPUs and GPUs | **Begun:** confirmed public EasyTPP wins (Taobao, Taxi) at a fraction of the state of the art's compute. Next: the remaining EasyTPP datasets; irregular clinical/sensor series (P12, P19, PAM); sealed FAS v2 against anonymous-log classical detectors and small LSTM/Transformer references (oracle-assisted trackers reported separately). Then one design-partner dataset |

One successful deployment can provide value before the full platform exists.

**Inductive biases are a value driver without special hardware (user direction, 4 October 2026).** The architecture
builds in assumptions that dense sequence models must learn from data:
- Time is computational: memory decays and rotates with the real time between events.
- The next event is the first arrival among competing processes, which is a race.
- Concurrent processes keep separate addressed state, with no shared dense vector.
- Learning credits the alternatives that were not taken.

Where data have this structure, these biases can turn into accuracy, earlier detection and data efficiency on existing
hardware, before any compute-cost argument or chip. Data of this kind include:
- industrial and IoT event logs (predictive maintenance, fault detection);
- IT-operations and security logs;
- transaction and order-flow streams;
- irregularly sampled clinical data;
- neural recordings for brain-computer interfaces;
- event sensors.

This is the shortest commercial path: a vertical model on a customer's own logs, priced on detection value. It shares
the core with the other rows, so it is correlated with them and not additive. It lowers execution risk by giving an
earlier revenue route that does not depend on scale or silicon.

Evidence stage, 4 October:
- The FAS benchmark (experiments/FAS_BENCHMARK.md) is built. Classical baselines are near chance for early detection
  (AUROC 0.50–0.56 up to 256 process events).
- The first native arm is queued. Dense, SSM and point-process references follow on AWS.
- The value is to be claimed only from completed comparisons.

Evidence update, 6 October:
- **Public leaderboard wins (EasyTPP):**
  - Taobao 1.399 ± 0.003 vs best published 1.318 nats/event at 0.92× the leading model's per-event compute;
  - Taxi: a 5-seed mixture (0.536) beats every published model at 0.41× S2P2's compute; a single model (0.525)
    leads S2P2 (0.522) on the mean at 1/12 of its parameters and compute.

  - StackOverflow: −2.144 ± 0.004 vs −2.163 (+0.019) at 1.26× S2P2's compute (accuracy win).
- **Public clinical win (P19 sepsis, 7 October):** AUPRC 0.639 vs 0.583 and AUROC 0.916 vs 0.903 against the best
  published model on the five official splits, with the same temporal memory core as the event model — evidence that the
  family generalizes across data types, not only across event datasets.

  All are 5 seeds on official splits with a sealed test: three of five datasets won, past the battle's pass criterion.
  Amazon and Retweet are in development.
- **FAS v2:** the sealed setting is selected on validation (2 interleaved lines, 2% dropped events). The identity
  oracle reaches AUROC 0.821 against 0.685 for the best anonymous classical detector, so a binding model has room
  to win. Development is in progress with a race readout derived from the interleaving theory.

Evidence update, 5 October: native wins against all six tested generic classical controls early (single-seed AUROC **0.600 vs 0.559 at 256 events**). The FIFO method reaches 0.755 using a route recovered from hidden training item identities. It is an **oracle-assisted diagnostic**, not a fair reference for discovering structure from anonymous traces. The v2 timing-aware probe also uses identity-derived training timing statistics. These scores do not establish a classical win under the native information protocol. FAS v2 is pre-registered; its primary reference comparisons must exclude privileged methods and retain them separately as oracle diagnostics ([protocol](../experiments/FAS_V2_CONFIRMATORY_PROTOCOL.md)).
Reusable mechanisms and tooling could then lower the cost of entering adjacent
workloads. Value capture could include model/runtime licensing, deployment
software, communication SDKs, accelerator IP and later hardware. These are
commercial hypotheses; neither vendor strategic fit nor repository privacy
establishes customer interest, legal exclusivity or adoption.

These opportunities are correlated through shared learning and execution
dependencies. Do not add their entire markets, count them as independent
success probabilities or infer a present price from hypothetical endpoint TAM.
The upside supports a premium only alongside credible execution and a path to
capturing part of the value created.

## 3. Execution evidence makes the ambition more than an idea

The project has progressed from a manifesto to implemented temporal/state
mechanisms, integrated language models, controlled mechanism comparisons,
source-bound results and a formal family with explicit composition/learning
contracts. This is evidence of research execution.

The [current evidence map](../report/architecture_evidence.md) supplies anchors:

- Matched value-informed route credit improves 2.506386→2.371491 BPC for about
  0.30% extra estimated fitting work. This supports a core learning mechanism.
- Doubling available receiver capacity improves 2.371491→2.345157 BPC with
  eight selected writes retained; fitting work rises about 1.650×. Useful
  capacity beyond activity is supported within that scope, with costs charged.
- At 10M characters the native model beats the 4-pass Transformer-256x4
  (1.888 vs 1.908 BPC) at 0.40× its training and 0.18× its inference compute,
  and beats the validation-selected tuned Transformers at both budgets. Tuned
  small LSTMs lead at these budgets by 0.04–0.06 BPC.
- At 90M characters, at equal training compute:
  - ≈ 2.1 PF: native 1.783 BPC beats a tuned Transformer-256×4 (1.811) at 1.06× compute, but a Transformer-192×4
    (1.704) leads;
  - ≈ 0.96 PF: native 1.800 loses to a tuned LSTM-512 (1.729) and a Transformer-192×4 (1.780).
- **Public benchmark:** confirmed EasyTPP wins on Taobao and Taxi (5 seeds, sealed test, a fraction of the leading
  model's compute) and StackOverflow (accuracy win at 1.26× compute).
- A causal online pilot improves 3.190859→3.095738 BPC through adaptation;
  economical native adaptation is the next step.

The language anchors are single-seed; the EasyTPP wins are 5-seed. The retained negative findings, causal-protocol
corrections, numerical contracts and full-resource definitions make the program
reviewable. The future premium depends on turning this
execution discipline into replicated, deployable benefits.

Tero Keski-Valkama is the sole founder. His public
[professional history](https://www.xing.com/profile/Tero_KeskiValkama) and
[project portfolio](https://keskival.github.io/) support relevant applied-ML and
software-architecture experience, subject to reference/availability diligence.
The published patent record ([Google Patents, all publications](https://patents.google.com/?inventor=Keski-Valkama&dups=language))
names him as inventor or co-inventor on 35 published US and EP patent documents (priority 2019 onward), all assigned
to HERE Global B.V.; it is evidence of sustained invention experience. The founder confirmed on 5 October 2026 that his earlier
patents belong to his former employer; they are excluded from this venture's
assets and valuation. Sole-founder status
does not imply sole research authorship: preserve Karoliina Salminen's credit
and resolve contributions, employment assignments and licensing.

The proposed €3M buys an 18-month increase in research and engineering capacity:
€1.35M team, €0.90M compute/replication, €0.25M hardware feasibility/measurement,
€0.20M legal/IP/operations and €0.30M reserve. The six-average-FTE staffing and
costs are planning assumptions. Hire complementary learning, runtime and
evaluation skills; stage spending by proof gates. Aggressive scope requires
concentrated tests and a team, rather than launching every application at once.

**IP protection is an explicit use of capital.** Within the existing €0.20M
legal/IP/operations envelope, fund an ownership/disclosure inventory, counsel-led
prior-art and patentability review, selective priority filings, and staged
prosecution alongside trade-secret protection. Obtain quotes before allocating
the shared envelope; no filing count or granted portfolio is assumed. The
[IP protection plan](IP_PROTECTION_PLAN.md) defines deliverables and disclosure
constraints. Company-owned rights over commercially relevant technical inventions
can strengthen defensibility and value capture; spending or filing alone does
not justify a mechanical increase in valuation.

## 4. What financing precedents establish

[Sakana AI's January 2024 announcement](https://sakana.ai/seed-round/) reports
a $30M seed raise and an experienced founding/research team, including a
Transformer co-creator. [Fractile's transaction adviser](https://www.goodwinlaw.com/en/news-and-events/news/2024/07/announcements-technology-fractile-raises-15-million-in-seed-funding)
reports its $15M seed financing in July 2024 for new AI chips and systems.
These primary announcements show that investors fund ambitious architecture
and computing-substrate programs at an early stage. Neither cited announcement
discloses a priced pre-money valuation; funding amounts are not valuation
comparables, and team/maturity differences prevent direct price transfer.

The implication is that the category is investable. It does not establish that
our €50M or €100M proposal is accepted or typical. A suitable deep-tech investor
must independently assess founder/team execution, reproducibility, rights,
capital needs, competitive alternatives and the eventual value-capture route.

## 5. Interpret the two prices explicitly

| Proposed priced pre-money | €3M post-money | Initial new-investor stake | Interpretation |
| --- | --- | --- | --- |
| €50M | €53M | 5.6604% | Ambitious current opening price for the model/learning/runtime/substrate opportunity |
| €100M | €103M | 2.9126% | Stretch case requiring stronger conviction, financing competition or additional de-risking |

Ownership excludes option-pool changes, preferences, fees and later financing.
At tenfold gross proceeds, a €3M investor needs €30M: absent later dilution,
that corresponds to €530M or €1.03B exit equity value under the respective
rows. Further dilution raises the required outcome. This is conditional equity
arithmetic, not an exit forecast or a discount-rate-adjusted return estimate.

**Assessment (revised 6 October):** €50M remains aligned with the ambition and is now better supported. The family has
confirmed public-leaderboard wins at a fraction of the state of the art's compute, the "useful-quality/full-cost
advantage" this rationale names as the highest-value evidence. That moves €50M from aggressive toward
defensible.

€100M becomes arguable when two or more of these hold:
- matched-compute wins on further EasyTPP datasets (StackOverflow, won at 1.26× compute, is partway there);
- a second domain (irregular clinical/sensor series, or the sealed FAS v2 confirmation);
- an independent reproduction of a leaderboard result.

Investor competition, not the evidence alone, sets the attainable price.

**Condition status, 7 October 2026.**
- *Second domain:* **met** — P19 sepsis prediction, five official splits, AUROC 0.916 ± 0.022 and AUPRC 0.639 ± 0.039 vs
  the best published 0.903 / 0.583 (MTM), with the same temporal memory core as the event model.
- *Matched-compute wins on further EasyTPP datasets:* **met** — StackOverflow at matched size (0.968× S2P2's parameters,
  1.015× its per-event compute): −2.1525 ± 0.0045 vs −2.163, all five seeds above S2P2. Also: the Amazon protocol (0.29× S2P2's per-event
  compute) finished at 0.784 ± 0.027 vs 0.781 (mean ahead, not confirmed: one low-basin seed); a matched-size StackOverflow protocol (≈1.0× S2P2)
  is queued.
- *Independent reproduction:* **met on independent hardware** — the Taxi win was rerun from scratch on the curie host
  (Intel i5-4690 desktop CPU, separately downloaded HuggingFace `easytpp/taxi` release, frozen driver sha256 73d2f95e…,
  five seeds): TEST 0.5252 ± 0.0007 nats/event vs the AWS run's 0.5250 ± 0.0010 and S2P2's 0.522 ± 0.004. Seeds 0, 1 and 3
  reproduce AWS to ≤ 3·10⁻⁹; seeds 2 and 4 differ by +0.0001 and +0.0010 (different BLAS/CPU arithmetic changes the
  early-stopping path). All five curie seeds are above S2P2's mean. Same codebase and team; a third-party rerun of the
  released code is the stronger form and the next step. The estimator-parity check (all wins hold under EasyTPP's own
  Monte Carlo estimator) is separate supporting evidence.
Supporting, not on the list: exact containment of Mamba (selective SSM) and attention in the family; two private paper
drafts. **All three conditions now hold (7 October), which makes €100M arguable on this rationale's own criteria;** the reproduction is on independent hardware by the same team, and a third-party rerun is the next step. The asking price is the founder's decision.

Earlier assessment (5 October): €50M remains aligned with the ambition as an aggressive,
evidence-informed negotiating thesis. The new definition/opportunities strengthen
the explanation of potential value, while their untested status leaves much
of the financing risk unchanged. €100M is a stretch scenario. We should not
present either as already established fair value. Research progress can justify
a premium before revenue; investor demand and terms determine attainability.

The highest-value next evidence is a repeated useful-quality/full-cost advantage,
trained sparse deployment, effective delayed memory learning and a capable team
with usable rights. Hardware is a real option on top of the software base case: sparse, memory-local,
clockless execution attacks the data-movement and power-density limits of current AI
chips, and supplies the competitive model the neuromorphic chip class lacks
([HARDWARE_THESIS.md](HARDWARE_THESIS.md)). Its value rises with tuned-baseline
quality and a hardware cost model of a trained network. Embodied transfer and
asynchronous codecs add potentially large expansion options, each earned through its own measured first proof.
Keep the bold platform vision and explicit execution milestones together.
