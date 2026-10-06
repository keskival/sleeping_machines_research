# Experiment host rules

- Read `experiments/HANDOFF.md` for the current report, publishing and running
  benchmark state before continuing work from a new session.
- Before launching training, inspect the existing queue, result files, report,
  host memory, running jobs, and GPU occupancy. Reuse an existing benchmark
  configuration when possible; prioritize the deferred large Transformer
  comparisons on a provisioned AWS host.
- AWS exception explicitly authorized by the user on 2026-10-01: on
  `ip-172-31-47-132`, the AWS gym may run up to three one-thread CPU jobs
  through the bounded slot scheduler. It reserves the ordinary host lock,
  keeps per-slot locks/RSS watchdogs and at least 8 GiB MemAvailable, and
  serializes result publication. Other hosts and GPU jobs retain the
  one-training-job rule.
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
- Preserve completed result files and older report evidence while new runs are
  pending. Replace a leading valid result only after a completed, comparable
  run improves it, retaining the previous result in the historical record.
  State protocol errors and revised interpretations beside the original
  numbers instead of silently deleting them. Keep existing dense control
  results as historical evidence; launch no new Transformer/LSTM training on
  any host (user direction, 6 October 2026, below).
- Preserve the established architectural case when editing the report: time
  performs computation, hard routes learn through counterfactual credit, deep
  persistent event representations, and capacity beyond activity. Retain the
  supporting temporal algebra, key/value separation, silence-aware supervision,
  depth theory and compute-allocation reasoning. Integrate new findings with
  their evidence and scope instead of replacing these principles ad hoc.

# Own-model focus: no new dense controls (user direction, 6 October 2026)

- From 6 October 2026, spend compute on improving our own integrated models. Do not queue or launch new
  Transformer, LSTM or other external-architecture training on any host, including retries, tuning arms or
  "missing" controls listed in older orders. Compare against published benchmark and leaderboard scores under the
  exact matching data, tokenizer and scoring protocol. The 90M D Transformer 256x4 run already in progress on AWS
  completes as admitted; it is the last one. Completed dense-control results stay in the record and tables.

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
