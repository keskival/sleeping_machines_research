# Victory-first mandate (user direction, 14 October 2026)

Our purpose is to win: build the strongest model family and achieve material, reproducible advantage at scale. Act with ambition, initiative and determination. Treat success as the engineering objective that organizes the work, not as an incidental outcome of experiments.

- Choose the path with the strongest theoretical and empirical case for victory. Advance formal theory, intuition and implementation together; use each to sharpen the next decisive experiment.
- Exploit and utilise our unfair architectural advantages to the absolute maximum. Make temporal computation, persistent memory, sparse activity, asynchronous execution, local learning, counterfactual credit, online adaptation and capacity beyond activity sources of decisive competitive advantage. Design workloads, implementations and experiments that let these strengths pay off; never suppress them merely to resemble a competitor. Build the capabilities needed to exploit them. Remove self-imposed obstacles and repair weaknesses at their causes. Do not accept an existing driver, familiar architecture or convenient implementation as a ceiling on the model family.
- Spend compute and time to change a competitive decision, improve capabilities or remove a measured bottleneck. Reject busywork, aimless parameter sweeps and repeated experiments that do not advance the winning path. Extract actionable learning from every expensive run.
- Take ownership through completion: implement, validate, run, diagnose, integrate and publish the supported evidence. Coordinate useful advances from other agents and make our strongest available construction concrete.
- Communicate directly. Lead with whether we are ahead, by how much, at what cost, why it matters and what action follows. State supported wins as wins. State failures plainly and repair or replace the failed approach. Do not bury positive evidence in reflexive hedging or recite numbers without their implications.
- Build investor confidence through demonstrated progress: stronger prediction, more useful capacity, better scaling, more economical learning and execution, and credible routes to deployment. Make every claim traceable to the evidence; distinguish measured results from the next objective so the case for victory is trustworthy.

# Capability-first experiments (user direction, 14 October 2026)

- For **every benchmark**, field the strongest valid task-suitable native model and learning pipeline available, integrating suitable demonstrated advances from this and other agents. Consult the current evidence before choosing the fighter. Do not default to the easiest driver, an older weaker model or a deliberately restricted variant. If the strongest construction needs implementation work, do that work; a convenient proxy cannot silently become its replacement or its family verdict.
- Play architectural advantages deliberately: temporal computation and races, useful persistent memory, sparse addressed state updates, separate keys/values, counterfactual credit, silence-aware learning, optional/dormant capacity, asynchronous/local learning, and online adaptation where applicable. Declare actual mechanism coverage and missing integrations before admission. Do not hide supported positive findings or erase these mechanisms to resemble a conventional baseline.
- Do not match our model down to a reference's context, state lifetime, depth, activity, clock precision, synchronization or learning schedule. Use architecture-appropriate strengths on both sides. Pursue best achievable prediction as well as resource-specific wins; extra paid work is permissible when authorized and useful. Measure and disclose the resource/information differences instead of suppressing an advantage for superficial symmetry.
- Our objective is to win by using the model family's advantages. Design for the actual task, not the easiest existing driver. For market forecasting, language and every non-stationary stream, chronological online learning and useful persistent state are primary capabilities. Do not silently disable either and present the restricted variant as the model's competitive verdict.
- Enable causal weight adaptation for every eligible comparator in an online comparison. Score each forecast before its outcome can change weights or state used to issue that forecast. Adaptation uses only already revealed observations. Preserve a separate untouched test when the official protocol requires one; never introduce target leakage to pursue a win.
- Preserve our native temporal memory, independent event timing, sparse addressed computation, counterfactual learning and optional capacity wherever the integrated implementation supports them. State explicitly which mechanisms a candidate includes. A diagnostic missing these is a diagnostic, not the flagship model.
- Separate persistent numerical state from gradient truncation. A bounded credit window is not authorization to reset memory. Do not impose a Transformer's context, synchronization, activity or update schedule on the native model merely to make drivers look alike. Likewise allow references their useful online learning and architecture-appropriate execution. Report differing information, resource and service boundaries explicitly.
- Every new competitive queue admission must declare the task's stationarity/online requirement, the strongest-fighter selection rationale, each arm's online adaptation and state/history policy, retained mechanisms, all imposed restrictions with their concrete justification, resource accounting and the decision the result changes. Validate this declaration before training; unsupported or silently disabled strengths fail admission. Existing source-bound jobs finish unchanged unless explicitly superseded.
- Official fixed-weight benchmark protocols remain useful as separately labelled comparisons; fixed-weight or reset-state diagnostics cannot substitute for the primary non-stationary/online experiment. Report measured advantages plainly with their actual scope. Do not infer a family loss from a restricted implementation, or an online win from a static score.
- New experiments should answer a competitive decision or remove a measured implementation bottleneck. Reuse acquired models, data and completed evidence; avoid re-fitting an unchanged model merely because adding the missing online or persistent-state evaluation needs engineering.

# Experiment host rules

- Read `experiments/HANDOFF.md` for the current report, publishing and running
  benchmark state before continuing work from a new session.
- Before launching training, inspect the existing queue, result files, report,
  host memory, running jobs, and GPU occupancy. Reuse an existing benchmark
  configuration when possible; prioritize the deferred large Transformer
  comparisons on a provisioned AWS host.
- AWS host-specific exception, expanded by the user on14October2026: on
  `ip-172-31-47-132` (16vCPUs/8reportedphysicalcores/~31GiBRAM), the bounded
  gym may run up to six one-thread CPU jobs. Pin slots to six distinct physical
  cores, leave two physical cores and their SMT siblings for responsiveness,
  retain per-slot locks/RSS watchdogs and at least8GiB MemAvailable. The six-slot
  envelope caps each job at3,000,000KiBRSS; larger jobs require a deliberately
  reduced concurrency reservation. Reserve the ordinary host lock and serialize
  publication. Use run_safe_aws6.sh only through the six-slot coordinator.
  Existing trainers finish untouched before scheduler handoff.
- Curie is a smaller shared host: retain its stricter one-training-job limit,
  host reserve and cgroup headroom from its active pipeline. Do not copy the
  larger AWS concurrency policy to Curie, or infer newworker sizing from Curie.
  Other hosts and GPU jobs keep one-training-job admission unless separately
  authorized. Disposable-worker memory reserves use intended-workload
  measurements as clarified below.
- Run exactly one training job at a time on each host. Put every run in a
  uniquely named one-job queue and invoke `experiments/queue/run_safe.sh`;
  never launch a benchmark directly with Python or bypass its lock. If the
  lock is held, wait or report it. The lock is host-local and does not
  coordinate separate machines.
- Choose memory caps and timeout from the actual host capacity and measured
  workload. Keep the RSS watchdog enabled and preserve at least 8 GiB of
  `MemAvailable`. For CUDA, check and monitor `nvidia-smi`, set the PyTorch
  memory fraction, and never run concurrent GPU jobs.
- Use unique AWS-specific run tags and output names. Never overwrite a prior
  result or reuse a successful queue job name for changed settings. Keep long
  runs in `tmux` and preserve their command, logs, hardware, wall time, and
  metrics.
- Work and commit on `main`, as requested by the user; do not create research
  branches. When another host is changing the repository, preserve its changes
  and coordinate overlapping files. Update findings and the report from completed result
  files, and distinguish exploratory single-seed evidence from benchmark
  claims.
- Preserve completed result files (they are the evidence). Documents carry the
  current state: replace a leading result in the text only after a completed,
  comparable run improves it, and state a protocol error or revised
  interpretation where the current number appears. Superseded text, archived
  copies and old versions are not carried in documents or folders; git history
  is the record (user direction, 9 October 2026, below). Keep existing dense
  control results as evidence; launch no new Transformer/LSTM training on any
  host (user direction, 6 October 2026, below).
- Preserve the established architectural case when editing the report: time
  performs computation, hard routes learn through counterfactual credit, deep
  persistent event representations, and capacity beyond activity. Retain the
  supporting temporal algebra, key/value separation, silence-aware supervision,
  depth theory and compute-allocation reasoning. Integrate new findings with
  their evidence and scope instead of replacing these principles ad hoc.

# Newly provisioned disposable workers (user clarification,14October2026)

The existing shared-host memory reserves protect those hosts and remain in force. Do not impose the8GiB reserve as a universal minimum on new disposable experiment workers. Size their reserve, RSS and virtual-memory caps from the intended workload's measured peak, OS/runtime needs and uncertainty margin; retain watchdogs and exact resumable checkpoints. Model/data sizes must scale toward the objective rather than fit an arbitrarily small instance. One training job per new host remains the default; the named AWS gym exception is unchanged. See experiments/runbooks/README.md for scaling procedures. Runbooks do not admit purchases or withdrawn experiments.

# Own-model focus: no new dense controls (user direction, 6 October 2026)

- From 6 October 2026, spend compute on improving our own integrated models. Do not queue or launch new
  Transformer, LSTM or other external-architecture training on any host, including retries, tuning arms or
  "missing" controls listed in older orders. Compare against published benchmark and leaderboard scores under the
  exact matching data, tokenizer and scoring protocol. The 90M D Transformer 256x4 run already in progress on AWS
  completes as admitted; it is the last one. Completed dense-control results stay in the record and tables.
- Exception (user direction, 6 October 2026, 17:46 UTC): small LSTM and time-encoded Transformer references for the
  FAS v2 confirmation (battle B3) may be trained. FAS is our own benchmark and has no published scores, and the
  references strengthen the case. Scope: `experiments/fas/dense.py`, models `lstm` and `transformer`, d ≤ 128, ≤ 2
  layers. Tuning is limited to the protocol's grid on validation only, with three seeds for the selected
  configuration, through run_safe on AWS gym slots. No other external architectures are included.

# Current state only; git is the history (user direction, 9 October 2026)

- Documents (README, report parts, REPORT.md and its PDF, investor and EIC materials, papers) state the current results
  and status. Do not keep superseded sections "as the historical record", previous-version copies, `archive/` folders,
  stacked headline pages or publication receipt files in the repository; earlier versions live in git history.
- `report/make_pdf.py` regenerates REPORT.md and report/sleeping_machines_status.pdf as cover + public benchmark record
  (from the newest `report/public_wins_headline_evidence_*.json` packet) + model-family chapter + experiment appendices.
  Update the packet script and regenerate; do not append pages to the PDF.
- Deck and investment-case builders keep their publication receipts in the local `.git/*-preview/<tag>/` folder only.
- Result files under `experiments/results/` and the invalid-protocol quarantine under `experiments/archive/` are data, not
  document history; they stay.

- Exception (user direction, 9 October 2026): B4 audit of the published log-normal-mixture models. AWS may train the
  reference authors' own LNM models with their MIT-licensed code (github.com/tanguybosser/ntpp-tmlr2023) on MOOC and
  Retweets splits, solely to score them as recorded and dequantized within the recording cell. They are audit
  instruments, not tuned competitors: no tuning beyond the authors' published configuration, results reported only in the
  B4 loss analysis.

# Report structure (user direction, 6 October 2026)

- The report is split into three parts. `report/I_SCIENCE.md` (Part I — The science) is curated by hand and is the
  reader-facing report: ambition, model family, theory, evidence per front and next decisive tests. Update it when a
  completed result changes a headline, a front's verdict or the test priorities; keep it short.
- `REPORT.md` stays the complete combined record that builders and publishers maintain. After changing it, run
  `.venv-docker/bin/python report/split_report.py` to regenerate Part II (`report/II_METHODS.md`: protocols, accounting,
  contracts, drivers) and Part III (`report/III_RECORD.md`: every experiment entry) and all three PDFs. Place new section
  titles via `report/split_manifest.json` (patterns, overrides, reviewed list) when the run summary flags them.

# Benchmark work is development to win, not a verdict on a quick variant (user direction, 6 October 2026)

- Do not assemble a quick family member, run it once on a benchmark and record LOSS. A first-pass variant's result is
  evidence about that variant only; publishing it as a verdict damages the project and teaches nothing.
- Each chosen benchmark is an owned engineering project: study the task and the leading published methods; write the
  design rationale (which of our mechanisms should win and why); build the model for the task within the family;
  iterate on development data with error analysis after every round. The test protocol stays sealed; the model does not
  stay frozen.
- Report development status as "in development: best X vs reference Y (development data)". Declare a win, or a
  *developed-attempt loss* with a written diagnosis of why the family falls short there, only after the owner has worked
  a documented improvement plan. Results of first-pass variants are labelled as such.
- Work only the battles listed in `experiments/PRODUCT_ORDERS.md`. Every queued job names its battle and the decision
  its result changes; jobs without both are not admitted. Contracts and smokes run only inside a battle's pipeline.

# Current product orders (read first)

- `experiments/PRODUCT_ORDERS.md` lists the P0/P1 deliverables, hosts, queues and pass criteria that increase the
  project's value. Work them in priority order before any other experiment; it also says what to stop or defer.

# Defining and reporting wins

- `experiments/WIN_CRITERIA.md` defines matched-compute (training, inference, total), pure-accuracy, leaderboard and
  Pareto wins, efficiency points and evidence levels. Developing agents aim for these explicitly; reporting agents state
  a win plainly as a win with its evidence level and one scope statement, and state losses just as plainly.

# Writing style: no reflexive hedging (user direction, 5 October 2026)

- State results and mechanisms directly, with their numbers. Give scope once per table or section (seeds, data,
  compute convention), not as a caveat attached to every sentence.
- Do not follow a positive result with an unrelated negative or a pointer to limitations that do not bear on that
  claim. Report losses in their own lines, as plainly as wins.
- Keep a qualification only where it changes how a reader interprets the number: measured vs modelled (hardware
  energy), completed vs pending, untouched official test sets, protocol errors.
- Replace "remains open / not yet demonstrated / does not establish" boilerplate with the concrete next test.
- Lead every front-door document (README, report cover, overview, investor materials) with the full ambition: a
  general-purpose substrate spanning language and reasoning, multimodal world models, embodiment, event-native
  analytics, continual learning, communication, self-design and hardware. A single benchmark or domain is evidence
  for one front, never the definition of the project.

# Research direction and architectural continuity

- The objective is a trainable, scalable substrate that computes through time
  and sparse asynchronous events, with useful capacity beyond active work.
  Optimize prediction quality, learning and total resource use together.
  Improving a conventional dense model alone does not establish this thesis.
- Evaluate unusual claims by derivation, implementation and reproducible
  evidence. Familiarity with published architectures is not a truth criterion.
  Preserve and clearly present surprising positive results when supported;
  investigate their limits with equally demanding controls. Neither assume
  supremacy nor impose an arbitrary ceiling on potential improvement.
- Keep the core mechanisms explicit: computational delays and temporal races,
  including race attention; sparse addressed state updates; small messages that
  mix incoming content with persistent memory; separate keys and values; deep
  credit to unrealized alternatives and optional routes; silence-aware
  supervision where applicable. Distinguish available capacity, scored keys,
  selected state updates, value deliveries and counterfactual learning work.
- Prioritize experiments combining the core mechanisms. Use numerical
  contracts and small integrated fits before scaling them. Carrier-only,
  dense, synchronous and single-mechanism models are diagnostic controls;
  keep them labelled and do not silently promote them into the main research
  architecture or let their queue displace the integrated experiments.
- Before a substantive architectural substitution, state the concrete failure
  being addressed, the theoretical reason for the change, which mechanisms it
  retains or removes, and its implications for inference and learning work.
  Record this reasoning and the required comparison in the theory/handoff.
  Demonstrate retained contracts and test the change in an integrated model
  before committing a long run. Routine work within the authorized scope does
  not require a new user-approval step.
- Established embeddings, gates, normalization and local vector operations
  may support the substrate. Do not substitute them for temporal computation,
  sparse selection or counterfactual learning merely because they are standard.
  Attribute known primitives accurately and explain the contribution of the
  complete construction and its tested consequences.
- A weak result from a restricted variant is evidence about that variant.
  Diagnose missing mechanisms, clock precision, information paths, credit and
  protocol before generalizing it to the full architecture. Preserve negative
  findings too; revise or abandon a hypothesis when the relevant evidence
  warrants it, with the reason stated beside the historical claim.
- Keep the report's architectural explanation and strongest valid comparisons
  prominent, quantitative and visual. Explain what each model actually uses.
  Dense-carrier accuracy is not evidence for sparse race attention; dormant
  state is not proof of zero key-scoring or training cost. Surprising results
  need clear scope, not automatic dilution into vague potential.
- Compare consistent resource boundaries and quality/data protocols. Charge
  candidate discovery, losing-value credit and optimizer work; distinguish
  FLOPs, wall time, traffic and measured energy. Report isolated or projected
  advantages as such. Fix accounting errors without treating familiar dense
  computation as the preferred architecture by default.
  Comparison tables must use the same units and denominator for every model
  within a column. Show whole-fit and per-target work for ours and controls
  together; do not juxtapose ours whole-fit GFLOPs with controls per-target
  MFLOPs in separate tables that invite a false comparison.
- After each completed integrated stage, update the report appendix with
  quality, data/passes, capacity/selected activity, full fitting FLOPs,
  per-target fitting work and inference work. Show supported raw work gaps
  against saved references while marking unequal quality/data and estimate
  conventions. Reserve comparable-quality or iso-FLOP supremacy claims for
  completed evidence under the relevant protocol; never fill a pending cell
  with a prediction or an ongoing training score.
- At handoff, name the prioritized integrated model and queue, the remaining
  gaps in mechanism coverage, and any proposed departure from this direction.
  Read `experiments/THEORY.md` and its relevant notes before redesigning a core
  mechanism. `HISTORICAL_MOTIVATION_MANIFESTO.md` records the original motivation;
  it is not a substitute for current proofs or measurements.

# Invalid-protocol quarantine

- Target-dependent E63/E79 mixtures, including word-keyed extensions, are
  quarantined in experiments/archive/invalid_protocol/target_leakage_20261002.
  Never use their scores as research evidence or restore them into active
  result paths. Old queues/drivers are retired. Use causal E173/E174 results;
  a corrected90M mixture comparison is still open. Audit-only archives and
  old rendered reports are not a source of benchmark claims.

# FAS oracle classification (user direction, 5 October 2026)

- FIFO de-interleaving learns its route from hidden TRAIN item identities; the v2 timing-aware probe also learns gap statistics from those identities. Classify both as oracle-assisted diagnostics, never fair anonymous-log references. Test-time anonymity does not repair privileged training information.
- Preserve their measured scores and historical evidence with the protocol correction. Exclude privileged methods from reference selection and native win/loss decisions; retain the native single-seed 0.600 vs 0.559 AUROC win against six generic controls at N=256.
- Admit structure-learning references only when all fitting and selection use the same anonymous inputs as native. Keep oracle diagnostics separate in tables and benchmark decision rules.

# Language evidence: CPU and public-reference reuse (user direction, 5 October 2026)

- Read experiments/OPEN_LANGUAGE_REFERENCE_PLAN.md for the current language path. Use properly tokenized data and CPU-only execution. Reuse published GPT/Transformer runs and checkpoints rather than retraining their replications or launching a new LSTM crossover grid.
- Match the selected reference's exact tokenizer, corpus and scored-target/history protocol. Published quality comparisons need no new baseline fit. Claim matched training compute only with an audited baseline work record; missing work does not block a scoped quality result.
- Spend new fitting resources on the integrated temporal/sparse model, starting with a bounded token-interface fit and measured CPU throughput/RSS. Preserve the core mechanisms and all existing evidence/jobs. GPU benefit is unmeasured, but current work remains CPU-only.

- Do not redirect the main language program to character/byte prediction because it is easier for the current implementation. Proper tokenization is a user requirement. Address CPU costs in the token interface and learning implementation; benchmark convenience does not authorize changing the research objective.

- Before scaling the native language path, read experiments/LANGUAGE_IMPLEMENTATION_AUDIT_20261005.md. Driver/backend restrictions must not be generalized into limits of the model family. Distinguish persistent numerical state from gradient truncation, factual path differentiation from alternative-write credit, and selected inference activity from complete learning work. Repair measured failures while preserving core mechanisms; do not infer intent from code or invent quality gains from static inspection.

# Implementation ownership (user direction, 5 October 2026)

- We make the implementation. Current driver/backend choices are engineering work to change, not restrictions to accept or reasons to narrow the architecture, use handicapped data representations, or avoid useful experiments.
- Implement the capabilities needed for properly tokenized language, useful persistent memory and scalable routing/learning. Use resource measurements to choose economical execution and prioritize work; do not let existing code veto the objective. Numerical contracts and integrated comparisons are how we validate the engineering work.
