# Sleeping Machines — full pitch deck and diligence notes


Proposed raise: €3M. Bullish negotiating case: €50M priced pre-money; €100M stretch scenario.


The current valuation rationale is VALUATION_RATIONALE.md; the former $10M memo is preserved in the archive. The €50M proposal prices the platform ambition and execution case, not benchmark scores or a sum of application markets.


All financial outcomes, budgets and milestone timelines are assumptions. No customer interest has been reported. Repository is private by founder instruction on 3 October 2026. This deck is a private review artifact; distribution and any future publication require a considered disclosure decision.


Numerical benchmark/financial ledger: frozen 3 October 2026. The protocol notes separately scope the report's completed 90M result. Opportunity and valuation rationale updated 4 October; no pending training scores enter the deck.


Reading guide: slides 1–20 form the investor pitch; the remaining slides are optional technical and financial diligence.


## 1. Sleeping Machines

AI that learns what to compute, when to compute it, and what to remember.

Private review deck. The founder proposes a €3M raise. €50M priced pre-money is the central bullish negotiating thesis developed here; €100M is a stretch scenario, neither an independent fair-value appraisal nor an investor offer. The research evidence is exploratory and single-seed. No customer interest has been reported. No investor, vendor or customer has been contacted in preparing this deck.


## 2. Beats the saved Transformer baselines at a fraction of the compute.

Wikipedia text (text8); same data and test set for all. Error: bits/char (bpc) or sMAPE %; lower is better.

Matched-compute rows follow experiments/WIN_CRITERIA.md: our run uses no more compute than the reference on the stated axis and scores better. Evaluation window T=256 characters for every model. "Passes" = epochs over the 10M training characters. sMAPE = symmetric mean absolute percentage error (NeuroBench Mackey-Glass, tau 17). Training compute: whole-fit estimate; inference: per-position (native exact winner-only trace). The 4-pass Transformer comparison uses a 6-pass native run with 1.5x the optimizer updates; an update-matched rerun is queued. Second seeds are queued. The 90M row is an efficiency point, not yet a matched-compute result. NeuroBench Mackey-Glass is partial (20 of 30 official repeats). Update 5 Oct: the first completed tuned dense control (LSTM-384, lr .003, warmup, 6 passes, 254 TF) scores 1.840 versus our 1.888 at 352 TF: a loss at matched training compute, stated plainly. Five more tuned arms are running; the LSTM's stateful scoring context is being aligned; native runs were not learning-rate tuned either, and a matching native lr check is queued.

- [R1: Private completed language evidence ledger (3 October 2026)](pitch_deck_benchmarks.csv) — Saved native fits and E64 controls. Original JSON records and SHA256 hashes are supplied in the private diligence pack.


## 3. Model capacity and the cost of using it must scale differently.

The opportunity: preserve useful intelligence while reducing the work paid for each prediction.

This is the problem definition and investment hypothesis. It is not a claim that all existing architectures perform dense work or lack memory. MoE already separates available parameters from selected computation; recurrent and selective-state models already retain state. Our proposed contribution is a temporal race/state/alternative-credit construction and its tested consequences. Software and hardware benefits must be evaluated with tuned incumbents under the same workload boundaries.

- [R3: Architecture, numerical contracts and report](../report/sleeping_machines_status.pdf) — Private technical report: whole-family definition, landscape, composition and scoped evidence. Modern matched controls, trained sparse parity and measured hardware energy remain open.

- [S6: Google — Eighth-generation TPU architecture announcement](https://blog.google/innovation-and-ai/infrastructure-and-cloud/google-cloud/eighth-generation-tpu-agentic-era/) — TPU 8t/8i, model–hardware co-design and sparse MoE infrastructure. Existing internal capability creates both potential fit and competition.


## 4. One model family across today's architectural landscape.

Our approach combines strengths of today's main model types.

This is the whole-family position, not a claim that every implemented member contains every incumbent. The envelope intersects recurrent/SSM, attention, sparse expert, statistical/retrieval, adaptive-memory and temporal systems. Supported dense reference blocks and synchronous barriers are permitted endpoints. The proposed differentiation is a common stateful temporal/selection/credit construction and its eventual quality/resource behavior. Known primitives have prior art; wrapping a reference predictor establishes no advantage. Computational universality is not a strictly larger class than all finite computable predictors. The technical report states operator, state, precision and schedule conditions. Current language fits illustrate one branch; joint TTT, multimodal integration, general morphing and clockless hardware economics are separate milestones.

- [R3: Architecture, numerical contracts and report](../report/sleeping_machines_status.pdf) — Private technical report: whole-family definition, landscape, composition and scoped evidence. Modern matched controls, trained sparse parity and measured hardware energy remain open.


## 5. Choose the right computation in each region.

One family can mix selective temporal programs with rich interaction and causal memory.

The local choices form a generative design space, not a catalogue of unrelated benchmark models. Selective and full-support regions can coexist; synchronous barriers are optional local schedules inside event semantics. The same model structure supports causal parameter updates and batching. Current state/statistical adaptation and an online pilot are scoped evidence; they do not establish fully asynchronous on-chip learning or a general TTT gain. Learned event routes already adapt execution inside a fixed graph; operator selection, reception/depth choice, allocation and growth/pruning require future utility, state/version migration and resource accounting. The formal core defines twelve fields and compatibility rules, and illustrative complete members show contrasting constructions without adding benchmark evidence. Preserve computation, representation and trainability through interfaces; forward equality, learning preservation and lower cost are distinct contracts. Learned memory/scheduling can be viewed as fast-state adaptation governed by slow trained rules, drawing on memory-augmented meta-learning and TTT. The opportunity register defines native irregular ingestion, asynchronous codecs and joint embodied/cognitive paths with explicit first proofs. The prior-art boundary and full work/bit accounting are stated in the report.

- [R3: Architecture, numerical contracts and report](../report/sleeping_machines_status.pdf) — Private technical report: whole-family definition, landscape, composition and scoped evidence. Modern matched controls, trained sparse parity and measured hardware energy remain open.


## 6. A substrate for intelligence that learns across mind and body.

The ambition: connect sensing, action, language and reasoning in one trainable model, across their natural timescales.

AGI here means the long-term ambition of broadly transferable learning and competence across domains, not a claim that a shared graph or computational universality guarantees intelligence. The visual is a proposed family composition, not an implemented robotics system. Common content/time/address/state interfaces can connect asynchronous sensory and motor events with language and deliberation; local dense regions, joins and batching remain available. No global periodic tick is required by the family, but causal ordering, timers and task/actuator deadlines remain. Fast persistent state and slower learned rules offer a path to online adaptation and memory-policy learning; fully native sparse asynchronous learning is still a research target. The opportunity is to test whether embodied experience teaches transferable causal, geometric and planning abstractions and whether cognitive learning improves action, using shared representations and credit. Both directions require separate held-out, matched-data/work comparisons with joint dense/VLA, isolated-module and stopped-credit controls, including retention, replay and system resources. Existing PaLM-E and RT-2 demonstrate related joint-learning or web-to-control precedents; we do not claim that conventional models cannot integrate these domains. Neither precedent establishes our proposed motor-to-cognitive transfer or our cost advantage. Shared parameters do not establish AGI, and no project embodied-transfer/robot-control experiment has completed. Platform upside is conditional and correlated with the same unresolved learning and execution risks; it is not an extra independent valuation or an assigned probability. See INVESTOR_PROOF_PLAN.md for staged work and the smallest simulated transfer protocol; no queue or new compute is reserved by this slide.

- [R3: Architecture, numerical contracts and report](../report/sleeping_machines_status.pdf) — Private technical report: whole-family definition, landscape, composition and scoped evidence. Modern matched controls, trained sparse parity and measured hardware energy remain open.

- [T4: Driess et al. — PaLM-E: An Embodied Multimodal Language Model](https://arxiv.org/abs/2303.03378) — Joint embodied, vision and language training with positive transfer; a precedent, not evidence for this substrate or motor-to-general-cognition transfer.

- [T5: Brohan et al. — RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control](https://arxiv.org/abs/2307.15818) — Joint vision/language/action learning transfers web knowledge to robotic control. Neither an AGI demonstration nor evidence for our resource advantage.


## 7. Messages race. Selected memories update. Alternatives learn.

A simple event loop connects representation, timing and sparse computation.

This construction combines familiar vector operations with temporal computation and sparse state. Separate keys determine addressing; values determine delivered content. Counterfactual credit gives unrealized alternatives a learning signal without softening the hard forward route. The present implementation is not a physical asynchronous machine: autograd, shared training windows, clipping and Adam remain. Winner-only inference still scores all candidate keys.

- [R3: Architecture, numerical contracts and report](../report/sleeping_machines_status.pdf) — Private technical report: whole-family definition, landscape, composition and scoped evidence. Modern matched controls, trained sparse parity and measured hardware energy remain open.


## 8. The language experiments establish three useful facts.

Wikipedia text (text8), 10M characters of training. Prediction error in bits per character: lower is better.

T256 test values come from completed held-out evaluations; training scores are excluded. This frozen numerical comparison is the 10M experiment; the completed larger-data result is separately scoped in the protocol appendix and current report. One nominal pass uses about 9.994M training positions. A single seed and differing native/control training order limit generalization. The p96 improvement over LSTM is 0.0081 bpc, with 2.89 times fitting work; this one-pass row is a quality win at higher training compute; the matched-compute wins (slide 2) use the multi-pass and wider rows.

- [R1: Private completed language evidence ledger (3 October 2026)](pitch_deck_benchmarks.csv) — Saved native fits and E64 controls. Original JSON records and SHA256 hashes are supplied in the private diligence pack.


## 9. Single pass: error versus training compute.

Each model sees the 10M training characters once. Lower and further left is better. Slide 2 adds multi-pass runs.

p32 / D4 credited: 2.371491 bpc, 7.243 estimated whole-fit TFLOPs, 108,875 parameters. Saved Transformer-256×2: 2.426909, 111.262 TFLOPs, 1,658,907 parameters. This gives 15.36× lower fitting arithmetic and 15.24× fewer parameters for this saved control, with 0.0554 lower bpc. The strong one-pass LSTM is 2.170597 at 20.306 TFLOPs. Best native p96 is 2.162463 at 58.648 TFLOPs. Tuned dense references at our exact budgets (validation-selected) and the 90M matched-compute runs are running.

- [R1: Private completed language evidence ledger (3 October 2026)](pitch_deck_benchmarks.csv) — Saved native fits and E64 controls. Original JSON records and SHA256 hashes are supplied in the private diligence pack.


## 10. Start with low-batch language inference on existing hardware.

Product hypothesis for local and on-premise text applications; no customer interest reported yet.

This is a product hypothesis chosen to make the research investable and testable, not a confirmed customer need. The present character-language fits do not establish a production-quality token model, application task performance or persistent cross-request serving. Low-batch inference is a proposed test venue for stateful sparse execution, not a measured region of advantage. Tune recurrent, modern Transformer/MoE and selective-state controls; include setup, memory, warmup, traffic, latency and energy. The ≥20% threshold is a proposed economic screen, not a forecast. No customer interest has been reported; demand interviews and evaluation partners remain future work.

- [R3: Architecture, numerical contracts and report](../report/sleeping_machines_status.pdf) — Private technical report: whole-family definition, landscape, composition and scoped evidence. Modern matched controls, trained sparse parity and measured hardware energy remain open.


## 11. Software proves the advantage. Hardware multiplies it.

One research program, with staged routes to deployment.

Hardware is a multiplier on the software result, applied to different parts of the energy budget. Largest lever: memory locality and sparse activity (DRAM read ~640 pJ vs ~3.7 pJ float multiply at 45 nm, Horowitz 2014); dense low-batch inference streams all weights per token, our model reads a few local state slots per event. Clock removal adds: no clock-tree power, average-case latency, near-threshold voltage tolerance, leakage-only idle capacity, and thermal headroom for memory-on-logic stacking and modular chiplets. Training locally is plausible (route credit and delay/gate updates are local) but the current trainer uses non-local backprop and clipping; a local learning rule is a milestone. Event-driven, memory-local chips exist (Loihi 2 and others); the field lacks a competitive trainable model, which is our licensing opening. No Sleeping Machines chip or joule measurement exists; baselines for hardware claims are competent clock-gated, SRAM-heavy synchronous designs as well as GPUs. Details: HARDWARE_THESIS.md.

- [R3: Architecture, numerical contracts and report](../report/sleeping_machines_status.pdf) — Private technical report: whole-family definition, landscape, composition and scoped evidence. Modern matched controls, trained sparse parity and measured hardware energy remain open.

- [H1: M. Horowitz — Computing's energy problem (and what we can do about it), ISSCC 2014](https://doi.org/10.1109/ISSCC.2014.6757323) — 45 nm per-operation energies: 32-bit DRAM read ~640 pJ, small SRAM read ~5 pJ, 32-bit float multiply ~3.7 pJ. Context for data-movement cost, not a Sleeping Machines measurement.

- [H3: Private hardware thesis (4 October 2026)](HARDWARE_THESIS.md) — Mechanism analysis of clockless, event-driven, memory-local execution; no chip or joule measurement.


## 12. A useful alternative could strengthen a hardware ecosystem.

Potential evaluation and licensing routes; no partner interest or relationship is claimed.

AMD is the strongest identified public fit precedent, not a presumed buyer. Its Ventures page lists Series A–C focus, so a €3M research-stage round may need angels/specialist deep-tech leads before strategic institutional investment. Intel’s system is a research prototype; Google has substantial internal alternatives and could build rather than license. Possible routes are benchmark collaboration, co-development, licensing, investment or eventual acquisition; none is a current pipeline asset. No company logos or implied endorsements are used.

- [S1: AMD — Silo AI acquisition completion, 12 August 2024](https://www.amd.com/en/newsroom/press-releases/2024-8-12-amd-completes-acquisition-of-silo-ai-to-accelerate.html) — Announced approximately $665M all-cash acquisition; Poro and Viking on AMD. Mature-team strategic precedent, not a pre-seed valuation comparable.

- [S5: Intel — Hala Point research system, 17 April 2024](https://www.intel.com/content/www/us/en/newsroom/news/intel-builds-worlds-largest-neuromorphic-system.html) — Loihi 2 neuromorphic research prototype, event computing and prospective continuous learning. No Intel interest in this project claimed.

- [S6: Google — Eighth-generation TPU architecture announcement](https://blog.google/innovation-and-ai/infrastructure-and-cloud/google-cloud/eighth-generation-tpu-agentic-era/) — TPU 8t/8i, model–hardware co-design and sparse MoE infrastructure. Existing internal capability creates both potential fit and competition.


## 13. Price the runtime against verified customer value.

Illustrative annual license economics; assumptions to test with an evaluation partner.

This example brings the commercial argument to an auditable customer denominator rather than treating global electricity, capex or subscriber counts as company TAM. All amounts are annual EUR. €1M eligible workload spend × 20% reduction = €200k gross saving; a 20% capture yields a €40k annual license and €160k buyer benefit before migration and other costs. Twenty-five equivalent licenses produce €1M annual revenue. None of these assumptions is established by current FLOP estimates or customers. Customer-count, scope, quality, reliability and adoption must be validated. This replaces the main-deck €1B spend/100M devices illustration; platform-scale scenarios remain explicitly hypothetical in the appendix.


## 14. AI research and software architecture in one founder.

Tero Keski-Valkama · Sole founder · Public career and invention record.

The XING timeline lists Cybercom December 2012–June 2018, HERE July 2018–April 2022, Alloy May 2022–August 2023, and Kaiko September 2023–December 2025. The founder’s portfolio lists Sleeping Machines and FAS Simulator. The primary EPO application EP4148389A2, published 15 March 2023, names Tero Juhani Keski-Valkama as inventor and HERE Global B.V. as applicant. It supports relevant invention experience, not a patent grant or an asset owned by this venture. Public personal biography and university links have availability/staleness issues; no degree or current-employment claim is inferred from them. Founder availability and references remain diligence items.

- [F1: Tero Keski-Valkama — public professional timeline](https://www.xing.com/profile/Tero_KeskiValkama) — Self-reported historical roles: Cybercom, HERE, Alloy.ai, Kaiko.ai. Dates and current commitment require founder confirmation.

- [F2: European Patent Office — published application EP4148389A2](https://patentimages.storage.googleapis.com/25/dd/41/af29b8e1391162/EP4148389A2.pdf) — Primary published document names Tero Juhani Keski-Valkama as inventor and HERE Global B.V. as applicant. Experience evidence, not Sleeping Machines-owned IP or proof of a grant.

- [F3: Tero Keski-Valkama — public project portfolio](https://keskival.github.io/) — Self-published portfolio lists Sleeping Machines and FAS Simulator; project access may have changed. Sole-founder status supplied by founder.


## 15. Build defensibility across the model and its execution.

The strongest potential asset is an integrated stack that is difficult to reproduce and deploy.

Tero instructed on 3 October 2026 that the GitHub repository was made private. This deck is a local private artifact and no public publication is authorized. Earlier public disclosure cannot be undone by repository privacy. README credits Tero Keski-Valkama and Karoliina Salminen as research authors; sole-founder status does not establish sole authorship, inventorship or IP ownership. Contributor agreements, intended licenses and employment-related rights must be resolved before presenting an exclusive technology asset. No granted patent portfolio, customer integration or defensible legal monopoly is claimed.

- [R3: Architecture, numerical contracts and report](../report/sleeping_machines_status.pdf) — Private technical report: whole-family definition, landscape, composition and scoped evidence. Modern matched controls, trained sparse parity and measured hardware energy remain open.

- [F2: European Patent Office — published application EP4148389A2](https://patentimages.storage.googleapis.com/25/dd/41/af29b8e1391162/EP4148389A2.pdf) — Primary published document names Tero Juhani Keski-Valkama as inventor and HERE Global B.V. as applicant. Experience evidence, not Sleeping Machines-owned IP or proof of a grant.


## 16. Three gates determine whether this becomes a business.

Each gate has a falsifiable result and a consequence for further spending.

The principal investor risks are scalable prediction quality, economical execution and commercial adoption. Technical gaps include the trained float32 sparse-backend contract, longer memory/credit horizons, stable adaptation and hardware timing/precision. Corporate rights, team hiring and founder availability remain diligence items. The program preserves its architectural thesis while allowing a narrower viable product. No probability of success is assigned to these gates, and the €50M proposal is not inferred from benchmark scores. Execution priorities and owner admission paths are in INVESTOR_PROOF_PLAN.md. Its first priorities are replicated integrated quality, trained sparse execution and delayed memory-credit fidelity. The proposed ≥20% complete serving-cost improvement is a future commercial gate to lock before measurement, not an observed benefit. The AGI expansion proof is downstream of these technical gates.

- [R1: Private completed language evidence ledger (3 October 2026)](pitch_deck_benchmarks.csv) — Saved native fits and E64 controls. Original JSON records and SHA256 hashes are supplied in the private diligence pack.

- [R3: Architecture, numerical contracts and report](../report/sleeping_machines_status.pdf) — Private technical report: whole-family definition, landscape, composition and scoped evidence. Modern matched controls, trained sparse parity and measured hardware energy remain open.


## 17. The next 18 months produce evidence investors can underwrite.

Proposed milestones measured from funding; spending advances with completed gates.

The timing is a planning assumption measured from funding. Three seeds and tuned modern controls are validation targets, not present assets. The ≥20% complete-cost gate is a proposed commercial screen and should be adapted to actual service requirements before experiments. Model quality and resource accounting must stay jointly comparable. Current owner queues retain priority; no new training or unguarded compute is launched to prepare this deck. Continued funding depends on completed results, measurable economics and a credible next workload rather than predictions entered as benchmark evidence. The immediate autonomous work sequence and current evidence gaps are documented in INVESTOR_PROOF_PLAN.md; timeline estimates do not substitute for gates or physical-host admission.

- [R3: Architecture, numerical contracts and report](../report/sleeping_machines_status.pdf) — Private technical report: whole-family definition, landscape, composition and scoped evidence. Modern matched controls, trained sparse parity and measured hardware energy remain open.


## 18. €3M funds an 18-month research and engineering program.

A proposed six-person average team, staged compute and hardware feasibility.

The proposed allocation totals exactly €3M: team €1.35M, compute/replication €0.90M, hardware feasibility/measurement €0.25M, legal/IP/operations €0.20M, reserve €0.30M. Average six FTE × €150k/year fully loaded × 1.5 years equals €1.35M; this is a staffing assumption, not an actual hiring plan or salary survey. Operating burn excludes reserve; total budget consumption would average about €166.7k/month over 18 months if reserve is spent. Stage compute purchases by quality/system gates. Allocate founder plus representation/credit, systems/compiler, evaluation and hardware-measurement skills; exact staffing mix and ramp remain open.


## 19. Raise €3M to establish a commercially valuable platform.

Proposed priced round; terms remain subject to company formation and investor diligence.

€3M at €50M priced pre-money yields €53M post-money and 5.6604% initial ownership before option-pool changes, fees, preferences and subsequent financing. This is a proposed negotiating price, not an independent appraisal, market-clearing estimate or investor offer. The main pitch explains the use of capital and proof points; it does not use arbitrary success odds to establish price. The appendix preserves transparent sensitivity arithmetic for investor discussion. A small ownership stake at a premium research-stage price may narrow the compatible investor pool; the round structure and lead investor fit are open commercial questions. The current VALUATION_RATIONALE.md explains ambition, potential assets and execution separately: coherent family design; value-credit and capacity mechanism evidence; completed restricted 90M learning; source-bound contracts and relevant founder experience. Further applications remain correlated hypotheses. A premium can fund transformative research before revenue, but breadth does not establish a market-clearing valuation. €100M remains a stretch scenario. Retain research co-author credit and resolve ownership; the financing plan adds complementary learning, runtime and evaluation capacity.

- [R1: Private completed language evidence ledger (3 October 2026)](pitch_deck_benchmarks.csv) — Saved native fits and E64 controls. Original JSON records and SHA256 hashes are supplied in the private diligence pack.

- [R3: Architecture, numerical contracts and report](../report/sleeping_machines_status.pdf) — Private technical report: whole-family definition, landscape, composition and scoped evidence. Modern matched controls, trained sparse parity and measured hardware energy remain open.

- [F1: Tero Keski-Valkama — public professional timeline](https://www.xing.com/profile/Tero_KeskiValkama) — Self-reported historical roles: Cybercom, HERE, Alloy.ai, Kaiko.ai. Dates and current commitment require founder confirmation.

- [F2: European Patent Office — published application EP4148389A2](https://patentimages.storage.googleapis.com/25/dd/41/af29b8e1391162/EP4148389A2.pdf) — Primary published document names Tero Juhani Keski-Valkama as inventor and HERE Global B.V. as applicant. Experience evidence, not Sleeping Machines-owned IP or proof of a grant.


## 20. Build the model. Prove the economics. Open the hardware path.

€3M proposed raise · €50M proposed pre-money · Private investor discussion.

Requested financing is €3M. The financing purpose is an aggressive but gated increase in experiment throughput and engineering capacity. The key next investor asset is replicated quality plus a measured service-level advantage; a clockless-chip thesis adds upside after mapping is credible. This private deck includes an appendix for diligence rather than presenting pending work as completed. Contact details, company identity, jurisdiction, cap table and legal terms should be supplied by the founder before distribution. No outbound messages or public release were made.


## 21. The same model improves when unchosen paths receive credit.

Controlled small-model comparison; held-out error and estimated complete training work.

The local linearized credit estimator adds a derivative pathway for alternative delivered values while preserving the forward winner. It is not the exact full counterfactual trajectory gradient. The paired p32 / D4 T256 gain is 0.134895 bpc, with 0.3034% extra whole-fit estimated operations. The corresponding p32 / D8 gain is 0.130367. Attribution is stronger than a comparison between different widths, but a single seed still leaves variance and hyperparameter sensitivity unresolved.

- [R1: Private completed language evidence ledger (3 October 2026)](pitch_deck_benchmarks.csv) — Saved native fits and E64 controls. Original JSON records and SHA256 hashes are supplied in the private diligence pack.

- [R3: Architecture, numerical contracts and report](../report/sleeping_machines_status.pdf) — Private technical report: whole-family definition, landscape, composition and scoped evidence. Modern matched controls, trained sparse parity and measured hardware energy remain open.


## 22. More available state. The same eight selected writes.

Small model with alternative credit: available memory doubles; selected updates stay fixed.

Pool 2 → 4 changes memory scalars from 512 to 1,024 and parameters from 108,875 to 177,019. At T256, quality changes from 2.371491 to 2.345157. Selected writes remain eight per input position; candidate keys increase 16 → 32. Winner-only per-position arithmetic increases 0.6576%, but whole-fit estimated work rises 64.9917%. All memory and key scoring still cost resources. A negative uncredited pool expansion was measured at D8, while this positive credited pair is D4; this is not a full matched 2×2 interaction test.

- [R1: Private completed language evidence ledger (3 October 2026)](pitch_deck_benchmarks.csv) — Saved native fits and E64 controls. Original JSON records and SHA256 hashes are supplied in the private diligence pack.


## 23. The sparse path is implemented; system proof is next.

Inference arithmetic per predicted character. Green bars: our implemented sparse runtime, which computes only the selected paths.

Small random float64 fixtures match full-emulator logits within roughly 1e-10. Actual trained float32 winner/state/cache parity and a full held-out rescore are prepared and pending. The prepared CPU worker retains prepacked matrices per fixed model version, but native admission remains unrun. Candidate key scans, copying, residency, setup, gather traffic, cache invalidation and quality must all be included in serving comparisons. The deck deliberately does not attach the trained quality to an unverified production backend or convert FLOPs to joules.

- [R1: Private completed language evidence ledger (3 October 2026)](pitch_deck_benchmarks.csv) — Saved native fits and E64 controls. Original JSON records and SHA256 hashes are supplied in the private diligence pack.

- [R3: Architecture, numerical contracts and report](../report/sleeping_machines_status.pdf) — Private technical report: whole-family definition, landscape, composition and scoped evidence. Modern matched controls, trained sparse parity and measured hardware energy remain open.


## 24. An adaptation signal exists. Its work is accounted for.

Predict-before-update experiment on a new text stream; separate from the latest model family.

Frozen versus online evaluation scores 8,191 targets with predict-before-update, 16-character delayed-feedback blocks, lr 1e-4, fresh Adam, clipping and full-backbone gradients. Both arms retain persistent local state. BPC improves by 0.095120; total processing-operation estimates increase from 210,902,697 to 2,264,163,359, a 10.7356× ratio. This supports trainability under adaptation, not inexpensive lifelong learning. Retention, drift, long-horizon memory and local asynchronous learning need new evidence. Loihi 2 already provides an on-chip learning precedent.

- [R2: Private online-learning experiment, 8,191 targets](pitch_deck_evidence_20261003.json) — Predict-before-update pilot: quality and complete processing-operation estimates; one checkpoint, stream and learning rate.

- [R3: Architecture, numerical contracts and report](../report/sleeping_machines_status.pdf) — Private technical report: whole-family definition, landscape, composition and scoped evidence. Modern matched controls, trained sparse parity and measured hardware energy remain open.


## 25. One substrate can support new communication and learning patterns.

Application hypotheses with first proofs; no project codec, embodied-transfer or AGI result.

The register in report/model_family_opportunities.md records six opportunities, mechanisms, evidence status, primary precedents and proof conditions. Dense arithmetic does not inherently require regularized input; variable-rate codecs and joint VLA models already exist. Our opportunity is coherent temporal/selective integration. Quantized message streams need a real wire/decoder contract; metadata/timing/reset and all endpoint work count. Joint motor/cognitive training needs shared information/credit and measured transfer; it does not prove AGI. Proposed application breadth supports platform optionality, not summed TAM or independent success chances. Existing integrated research queues retain priority.

- [R3: Architecture, numerical contracts and report](../report/sleeping_machines_status.pdf) — Private technical report: whole-family definition, landscape, composition and scoped evidence. Modern matched controls, trained sparse parity and measured hardware energy remain open.


## 26. The contribution is the complete construction.

Established alternatives already address parts of this problem; they remain essential controls.

The pitch does not assert sole invention of sparse activation, gating, normalization, key/value separation or hardware learning. It proposes an integrated trainable temporal substrate and evaluates consequences. The latest native language model has no dense carrier or n-gram count component, but remains a restricted software realization. A weak result in one restricted variant should be diagnosed; genuine superiority requires matched strong controls. Intel Loihi 2 is cited on the adaptation and strategic slides.

- [T1: Vaswani et al. — Attention Is All You Need](https://arxiv.org/abs/1706.03762) — Primary architecture reference. The deck’s small saved control is not a frontier-performance claim.

- [T2: Fedus et al. — Switch Transformers](https://arxiv.org/abs/2101.03961) — Capacity beyond selected computation has established MoE precedent; sparsity alone is not a novelty claim.

- [T3: Gu and Dao — Mamba](https://arxiv.org/abs/2312.00752) — Selective recurrent state-space sequence models are relevant competitive controls. No matched modern SSM result is claimed.


## 27. What clockless, memory-local silicon can buy.

Physical mechanisms, their basis and what must be measured. No chip exists yet.

Mechanisms with physical bases, not measurements. Dense models can also use SRAM-heavy or near-memory chips, and MoE partially decouples capacity from activity; our edge is the model's sparse, local access pattern, which must hold at competitive quality (P0-6 tuned baselines). Race logic (ISCA 2014) is prior art for first-arrival computation. Full analysis: HARDWARE_THESIS.md.

- [H1: M. Horowitz — Computing's energy problem (and what we can do about it), ISSCC 2014](https://doi.org/10.1109/ISSCC.2014.6757323) — 45 nm per-operation energies: 32-bit DRAM read ~640 pJ, small SRAM read ~5 pJ, 32-bit float multiply ~3.7 pJ. Context for data-movement cost, not a Sleeping Machines measurement.

- [H2: A. Madhavan, T. Sherwood, D. Strukov — Race logic, ISCA 2014](https://doi.org/10.1109/ISCA.2014.6853226) — Prior art for delay-coded, first-arrival computation in hardware. We build on this primitive; it is not our invention.

- [H3: Private hardware thesis (4 October 2026)](HARDWARE_THESIS.md) — Mechanism analysis of clockless, event-driven, memory-local execution; no chip or joule measurement.


## 28. AMD’s Silo AI deal supports the strategic logic.

Announced approximately $665M all-cash acquisition in 2024; a mature-company precedent.

AMD’s announcement connects the acquisition to end-to-end AI solutions, engineering expertise, enterprise customers and Poro/Viking model work using AMD platforms, including LUMI. It does not establish that chip usage alone caused the acquisition. Silo had a large team and existing customers; neither applies here. The approximately $665M announced transaction value and Liquid AI’s $250M Series A amount are different quantities and are not converted into our pre-money valuation. They demonstrate strategic precedent, not an offer, floor or near-term acquisition expectation.

- [S1: AMD — Silo AI acquisition completion, 12 August 2024](https://www.amd.com/en/newsroom/press-releases/2024-8-12-amd-completes-acquisition-of-silo-ai-to-accelerate.html) — Announced approximately $665M all-cash acquisition; Poro and Viking on AMD. Mature-team strategic precedent, not a pre-seed valuation comparable.

- [S2: AMD — Acquisition announcement and LUMI collaboration, 10 July 2024](https://www.amd.com/en/newsroom/press-releases/2024-7-10-amd-to-acquire-silo-ai-to-expand-enterprise-ai-sol.html) — AMD highlighted enterprise solutions, expertise, software and Poro/Viking training on LUMI using AMD hardware; no single exclusive acquisition cause established.

- [S3: Liquid AI — $250M Series A announcement, 13 December 2024](https://www.liquid.ai/blog/we-raised-250m-to-scale-capable-and-efficient-general-purpose-ai) — Financing announcement and AMD collaboration on efficient models; funding amount is not a disclosed company valuation.


## 29. The constraints reach from power grids to robot batteries.

Economic context for efficient intelligence; these figures are not our revenue market.

IEA projects all datacenter electricity, not only AI demand; GSMA subscribers are not annual device shipments; IFR reports industrial-robot installed stock, not all robotics. These denominators cannot be summed into a TAM. The thesis is an inference about potential demand for verified efficient computation. Incumbents are also improving rapidly, which raises the threshold for a commercially useful alternative.

- [M1: IEA — Key Questions on Energy and AI, executive summary](https://www.iea.org/reports/key-questions-on-energy-and-ai/executive-summary) — 485 TWh datacenter electricity in 2025; central projection 950 TWh in 2030. Sector context, not addressable revenue.

- [M2: GSMA — The Mobile Economy 2026](https://www.gsma.com/solutions-and-impact/connectivity-for-good/mobile-economy/wp-content/uploads/2026/02/The-Mobile-Economy-2026.pdf) — 5.8 billion unique mobile subscribers and 8.8 billion connections. Subscribers are not annual device sales or customers.

- [M3: IFR — World Robotics 2026 release, 24 September 2026](https://ifr.org/ifr-press-releases/news/five-million-robots-now-operate-in-factories-globally) — 5 million operating industrial robots in 2025 and more than 600,000 annual installations; no adoption by this project.


## 30. Successful platform adoption can support very large outcomes.

Conditional commercial arithmetic illustrates the scale required; no forecast or market multiple claim.

The €10B/€50B equity scenarios use deliberately stated revenue multiples rather than observed comparables. They require economics, durable margins and value capture that are wholly unproven. Multiplying revenue by a selected multiple is not an enterprise/equity reconciliation: the examples assume negligible net debt at exit. The scenarios overlap and are alternatives, not additive. Infrastructure relevance can justify funding risky research, but cannot itself prove a current valuation. Smaller niche success, delayed commercialization, licensing-only outcomes, further dilution and total failure remain possible.


## 31. The completed comparisons, using consistent resource units.

Frozen 10M-character, one-pass evidence. Model definitions are on the next slide.

All fitting columns include estimated complete step arithmetic rather than forward-only model FLOPs. Native representative operator traces include backward, alternative credit, clipping and Adam; dense controls use shape formulas with backward approximately twice forward. Inference includes traced unit and special-function arithmetic, but not DRAM traffic, copying, allocation, RNG, evaluation overhead or physical energy. Formula conventions differ and are stated. Per-position columns use one shared denominator type per column. Winner-only arithmetic is explicitly separated from emulator quality. Full CSV contains all 13 native fits and both controls.

- [R1: Private completed language evidence ledger (3 October 2026)](pitch_deck_benchmarks.csv) — Saved native fits and E64 controls. Original JSON records and SHA256 hashes are supplied in the private diligence pack.


## 32. Plain-language labels map to reproducible model records.

Message width counts scalars; available memory counts local state slots.

These names are editorial labels for the frozen completed configurations, not new models, products or a production-version series. Each native configuration has two heads per layer. Selected writes per input position are eight at four layers and sixteen at eight layers. Available slots equal layers × heads × pool size. Credit means the local linearized alternative-value estimator. The small timing-only row and alternative-credit row share forward parameters and architecture. Full filenames, source/parent hashes, exact operation counts and seed/protocol fields are retained in the private evidence ledger.

- [R1: Private completed language evidence ledger (3 October 2026)](pitch_deck_benchmarks.csv) — Saved native fits and E64 controls. Original JSON records and SHA256 hashes are supplied in the private diligence pack.


## 33. The comparison boundaries are part of the evidence.

Maintain the original records and disclose what a result actually measures.

The native score pays overlap-window warmup, evaluating roughly twice as many input positions as scored targets at T256. LSTM recurrent evaluation is a different protocol. Native T256 coverage is 999,936 targets, while original saved controls have a slightly different tail. The raw difference is small but must be disclosed. The frozen numerical ledger above covers 10M training. Separately, the current source-bound report records a completed 90M D4/P64/U2 value-credit evaluation: 1.857306 BPC on test[95M:96M], T256, 999,936 targets, one seed. Estimated whole fitting work is 241.219 TFLOPs, or 2.680 MFLOPs per fitting target presentation over 89,997,312 presentations. It changes data and width and is not the full official 5M test or a matched resource win. Checkpoints alone are not held-out results. Adaptive clock/stopping comparisons must preserve the declared query denominator and fallback, and charge attempted work; modeled time is not measured latency. No pending cells are filled with expectations. No additional Transformer or LSTM training is launched here; the user reserved it for AWS.

- [R1: Private completed language evidence ledger (3 October 2026)](pitch_deck_benchmarks.csv) — Saved native fits and E64 controls. Original JSON records and SHA256 hashes are supplied in the private diligence pack.

- [R3: Architecture, numerical contracts and report](../report/sleeping_machines_status.pdf) — Private technical report: whole-family definition, landscape, composition and scoped evidence. Modern matched controls, trained sparse parity and measured hardware energy remain open.


## 34. Keep the failures beside the positive mechanism results.

A disciplined research program should revise interpretations when evidence changes.

D8 uncredited pool 2 scores 2.4565 and pool 4 2.4981 at T256. The D4 credited pool pair is distinct and should not be sold as a fully matched interaction. Normalized read/write credit gives 2.3840 versus read-credit 2.3715; other write-credit variants diverge and are retained historically. Best p96 quality gains over LSTM with more work. DVS small screens suggested lower coarsening work, but the full all-seed quality gate failed and the strong RBF control remains better. Target-dependent E63/E79 mixtures are quarantined for leakage and never used in this deck.

- [R1: Private completed language evidence ledger (3 October 2026)](pitch_deck_benchmarks.csv) — Saved native fits and E64 controls. Original JSON records and SHA256 hashes are supplied in the private diligence pack.

- [R3: Architecture, numerical contracts and report](../report/sleeping_machines_status.pdf) — Private technical report: whole-family definition, landscape, composition and scoped evidence. Modern matched controls, trained sparse parity and measured hardware energy remain open.


## 35. Reverse-underwriting shows the assumptions behind a price.

This arithmetic is a sensitivity tool; the research does not estimate platform-success probabilities.

The illustrative model discounts the retained fraction of a successful future equity outcome to today and assigns zero failure value. It is not a complete corporate DCF, does not model all cash flows, and assumes the retained cohort benefits from an exit equity value after future financing. €10B exit × 30% retention / 1.15^10 is the conditional present value; €50M requires 6.7426% and €100M requires 13.4852%. These are neither forecasts nor inferred from small benchmarks. A €3M priced round at €50M pre-money gives 5.6604% initial investor ownership before option-pool changes, fees, preferences or future rounds; at €100M it gives 2.9126%. An investor must independently accept technology scalability, commercial value capture, rights and execution to support a premium price.


## 36. The price depends on dilution, exit scale and time.

Required platform-success odds at 10 years and 15% discount; zero failure value.

This table is generated from the frozen financial model, not entered by hand. At a €3B outcome and 10% retained equity, the required probabilities are much higher than at a €30B exit with 50% retention. A lower discount rate increases the present value, while delays and dilution reduce it. In the €3M/€50M round, an investor retains 1.6981% at exit if 30% of its initial stake survives, giving €169.81M or 56.60× gross MOIC in the assumed €10B success outcome. That conditional upside is not expected return; failure is zero and cash-flow timing/preferences are omitted.


## 37. A private evidence pack and primary external context.

Source labels in slide footers are clickable; companion files must travel with the PDF.

Raw research evidence is private, per founder instruction. Original completed result SHA256 hashes are in pitch_deck_evidence_20261003.json. The status PDF contains detailed theory and historical evidence; do not treat invalid-protocol archives or old rendered report scores as active claims. Market sources were accessed on 3 October 2026. These external sector denominators support relevance, not customer intent or a venture revenue forecast.


## 38. Public precedents establish relevance, not endorsement.

Historical roles are self-reported; inventorship is checked against the primary published document.

The founder profile and portfolio are linked from the founder slide (F1, F3). Transformer and Mamba references are linked from the competition slide. The private diligence checklist remains open: corporate entity, cap table, founder commitment, contribution chain including Karoliina Salminen, employer invention assignments, licenses, compute/hiring quotes, trained backend admission, replication and measured service advantage. A private repository does not rescind earlier disclosures or itself establish patentability. No permission to distribute or contact potential partners is inferred from preparing this deck.


## Reproduce and inspect

The editable slide narrative is PITCH_DECK.json. The frozen parent hashes, exact derived metrics and assumptions are in pitch_deck_evidence_20261003.json. CSV exports use identical column units for every model. Run the preparation only when deliberately updating the evidence cut-off; use a fresh publication tag to render. This build imports no numerical model runtime.


Open diligence: company/jurisdiction, cap table and option pool; founder availability; contribution and IP chain of title (including credited co-author Karoliina Salminen); employer invention assignments; intended software/model licenses; budget quotes; three-seed modern controls; trained sparse-backend parity; measured system energy/traffic; design-partner willingness to pay. No contacts have been made by this work.
