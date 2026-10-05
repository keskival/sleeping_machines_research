# Benchmark-win execution: owner continuation packet

**Protocol correction — 5 October 2026, user-directed:** FIFO is an **oracle-assisted diagnostic**, not an eligible reference for anonymous-process learning. `deinterleave_baseline.learn_route()` uses hidden TRAIN item identities to recover the route. The v2 timing-aware probe additionally fits transition-gap statistics with those identities. Neither receives test identities for prediction, but both receive privileged training structure unavailable to native and generic controls. Their scores are retained as oracle-assisted diagnostic targets; exclude them from strongest-reference selection and win/loss verdicts. Native's completed single-seed win against the six generic controls stands: **0.600 vs 0.559 AUROC at N=256**. A fair structure-learning reference must fit exclusively on the same anonymous training logs. Historical contrary interpretations below are superseded by this correction; numerical records remain preserved.

5 October 2026, 10:45 UTC · User instruction: make sure the next-step plan is done.

The revised [execution inventory](queue/benchmark_win_execution_20261005T111500Z.json)
retains 40 tasks: 23 active requirements, 16 explicitly deferred rows and one
completed historical R8 row. The original 37-task inventory is preserved.
This is a completion checklist, not a new scheduler or a list of claimed wins. The initial
[snapshot](results/diagnostics/benchmark_win_execution_20261005T104500Z.json)
records two completed primate sessions initially; the next snapshot also admits completed R8. Some unfinished tasks
may already be running; this Docker session cannot inspect physical-host locks
or occupancy and has no SSH executable. No host admission was made.

Refresh from newly published results with standard library only:

```bash
python3 experiments/benchmark_execution_status.py --manifest experiments/queue/benchmark_win_execution_20261005T111500Z.json
```

Use a fresh `--output` filename to preserve a dated snapshot. This checker
validates queue hashes, frozen-stage sources and predecessors, and rejects
ineligible strict-budget results. It does not establish numerical parity,
hardware savings or a benchmark victory by itself.

The [initial language audit](results/diagnostics/language_crossover_initial_audit_20261005T111500Z.json)
reuses existing source-bound validation admission: both 10M groups favor LSTMs;
the 90M groups are pending. Seven standard-library contract groups pass,
including explicit deferral and active dependencies. These are structural
checks; the k-write numerical contracts remain unrun.

## Main research objective: Transformer advantage at larger scale

The latest user direction is in [LANGUAGE_CROSSOVER_PLAN.md](LANGUAGE_CROSSOVER_PLAN.md).
Prioritize the existing 90M C control pair and D Transformer arms alongside the
ongoing native p96 comparison, then a bounded modern-language scaling study on
a provisioned GPU host. A D-optimized recurrent control is missing. Use LSTMs
to map the competitive region; a small text8 LSTM win is not a prerequisite.
Prepare the modern corpus/tokenizer, causal/context contracts and complete
accounting while these controls finish. No modern run is admitted here.

The FAS FIFO oracle-assisted diagnostic reaches .755 at N256 and .913 at N512 using hidden TRAIN identities. Native .600/.742 is compared separately against information-matched references. Further FAS-v1 fits, smokes, replication and binding
work are deferred behind language scaling unless already admitted. Preserve
their queues and completed evidence. The instructions below remain the
admission procedure if an owner later reactivates a deferred diagnostic; they
do not order the whole FAS sweep now. Full k-write fits are also deferred until
contracts/smokes justify one bounded scalable comparison.

## AWS owner

Continue the fixed primate six-session run, both 90M budget-C controls, the
existing D Transformer arms and the already running native p96 fit.
If a deferred FAS reference fit is reactivated, first run its new full-shape two-window smoke:
`queue/aws_fas_reference_smoke_{lstm,transformer,lru,s5t,mamba}_20261005T104500Z.txt`.
The existing data job is a prerequisite. The smokes use the actual full event
length and lanes, 32 fitting runs and 8 evaluation runs. Measure peak RSS and
window time; choose fit caps/timeouts from those measurements and actual host
capacity, retaining 8GiB available memory. Preserve original fit queues and
publish smoke results/provenance. See AWS_FAS_REFERENCES.md for provisional
per-model caps; those estimates do not replace the smoke.

Keep bounded recipe tests subordinate to the scaling objective. Before starting k-write fits,
check that none of the original k-write queues is active. If one is active,
preserve it as historical diagnostic evidence and coordinate its completion;
do not terminate it from this checklist. For future fits, use:

`queue/aws_kwrite_protocol_v2_20261005T104500Z/manifest.json`

Prepared stages: contracts → each model's smoke → its conditional fit, for pool 2/k2, pool 4/k1 and
pool 4/k2. Each stage has its own one-job queue. Contract and smoke deadlines
are 1800s; fits 18000s. RSS ceiling 3,500,000KiB, AS 8,000,000KiB, memory floor 8192MiB.
The source-frozen preflight requires the physical AWS hostname and inherited
run_safe lock; it rejects this container before importing Torch. Passed smoke
RSS must fit the next-stage reservation with a 1.5× margin. Owner inspection
of actual live capacity remains mandatory. Revised envelopes need new tags.

**Protocol repair:** the original wrapper changes the compiled layer only.
The base driver's first traced windows execute optimizer updates with ordinary
`batched_logits`, so those windows use k1 and their work excludes k2. V2 routes
both those eager traced updates and compiled training/evaluation through the
same k-write layer. It includes the executed arithmetic/backward/optimizer
trace and a separately labelled top-k comparison proxy. Indexing, communication,
physical traffic and runtime are not thereby measured. Compiled numerical
contracts and actual-shape smokes are prepared, not passed. Pool2/k2 is a
dense-write control; pool 4/k2 is the sparse-write candidate.

The old wrapper and all old queues remain unchanged. Do not admit an old
k-write trace as complete fitting-work evidence; put this correction beside
any completed historical result rather than deleting it.

The owner has independently queued strict retries at both budgets:

- `queue/aws_language_10M_strictB_p64d4_3.8pass_20261005T110000Z.txt`
- `queue/aws_language_10M_strictA_p96d4_5.85pass_20261005T110000Z.txt`

Reuse these existing queues. Our duplicate seed6 draft was withdrawn before
launch. The matching budget-B seed7 confirmation and its smoke are prepared at:

`queue/aws_native_strict_tfB_20261005T104500Z/manifest.json`

The confirmation uses the same p64/d4/pool2/linear-credit, 3.8-pass settings as
the owner's seed6 retry. It requires that original completed result to pass
actual work, finite quality, T256, 999936 targets and numerical-source checks
before importing Torch. Report both outcomes; this tests the tuned Transformer
sub-comparison and does not establish superiority over tuned LSTMs. The
projection and frozen reference budget are recorded in the manifest; only actual
traced work decides eligibility. No predicted quality is supplied.

The fitted resource ceilings are RSS 2,500,000KiB and AS 8,000,000KiB, timeout 18000s,
memory floor 8192MiB. They derive from the saved p64 fit's 1,452,032KiB peak and
8,186s duration; inspect actual AWS capacity and the prerequisite smoke before
admission. Fit still refuses if the smoke lacks 1.5× RSS margin.

## Curie owner

R8 has now completed: base half-life median about 24s and p90 about 350s,
yet AUROC at N256 is .573 versus tied R1 .567 and untied R0 .578. Long
horizons alone do not close the detection gap. Next, measure per-slot write
purity using oracle identities only in the diagnostic, then inspect original-
write credit across truncated segments. Neither diagnosis is yet proved by R8.
Original pool 2 FAS seed 7 and the
[binding diagnostic specification](FAS_BINDING_DIAGNOSTIC.md) remain visible
as deferred inventory rows under the language priority; FIFO is an oracle-assisted diagnostic. A bounded binding
diagnosis may nominate a transferable memory repair; it cannot establish a
v1 win against that control. Preserve active fits; use any native learning-rate
arms as bounded recipe evidence, then prioritize the larger-data objective.
Numerical running state requires physical owner checks.

## Publication and admission boundary

Use `experiments/queue/run_safe.sh` for each one-job queue, only after physical
inspection/reservation. On AWS, keep the already authorized maximum of three
bounded one-thread slots with inherited host/per-slot locks. Do not start a
second coordinator or a fourth worker. Never use a container-local free lock
as physical authorization. Update the existing owner's plan at a safe boundary
after coordination; this packet does not rewrite an active scheduler manifest.

After results land, record win/loss/efficiency point under WIN_CRITERIA.md,
complete same-unit quality/work tables and queue confirmation where appropriate.
State that FAS is synthetic; primate leaderboard claims require all six sessions;
new model variants need validation selection and fresh held-out confirmation.
The present packet completes preparation, not the physical runs or the research
objective. No success is assumed.
