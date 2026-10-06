# FAS interlaced-event anomaly benchmark (opened 4 October 2026)

**Protocol correction — 5 October 2026, user-directed:** FIFO is an **oracle-assisted diagnostic**, not an eligible reference for anonymous-process learning. `deinterleave_baseline.learn_route()` uses hidden TRAIN item identities to recover the route. The v2 timing-aware probe additionally fits transition-gap statistics with those identities. Neither receives test identities for prediction, but both receive privileged training structure unavailable to native and generic controls. Their scores are retained as oracle-assisted diagnostic targets; exclude them from strongest-reference selection and win/loss verdicts. Native's completed single-seed win against the six generic controls stands: **0.600 vs 0.559 AUROC at N=256**. A fair structure-learning reference must fit exclusively on the same anonymous training logs. Historical contrary interpretations below are superseded by this correction; numerical records remain preserved.

User direction: show the architecture in its home field, asynchronous event representations, with a challenge
in interlaced sequences. Data come from the user's own simulator, FAS-Simulator (Keski-Valkama 2017, *A simulator for
event-oriented data in flexible assembly system fault prediction*, Procedia Computer Science 119:121–130). The simulator is
vendored unmodified at commit 3839b10 (`experiments/vendor/fas_simulator_3839b10`, WTFPL), with SimPy 4.1.2 vendored
at `experiments/vendor/simpy_pkg`.

## Data

The simulator models one production run of a flexible assembly line:
- 30 assembly items pass through a 31-step line of conveyors, cranes, bowl feeders and manual stations, sharing
  resources and queues.
- The items' event streams are interleaved in one log, with no item identity.
- A 10-second clock adds TICK events.
- There are 46 event types.
- A clean run has about 1,364 events, 1,050–1,062 of them process (non-TICK) events, and lasts about 3,030 s.

**Faults** start at time 0 and grow with use:
- *Wear and tear:* a random conveyor's extra delay grows as (e^(t/5) − 1)/30 of its duration, where t is the number of
  uses. That is 0.7% at the first use, 21% at the 10th and 180% at the 20th.
- *Retry delay:* a random bowl feeder adds Poisson((e^(t/5) − 1)/4) × 0.2 of its duration.

Timing noise is 1% of each step's duration. Severity varies widely:
- Wear on conveyor 3, which every item uses twice, stalls the line for about 250 hours.
- Wear on later conveyors adds tens of minutes.
- A retry fault on an early bowl feeder can be almost invisible.

**Generation:** `experiments/fas/generate.py --name <dataset>`. Every run is seeded (disjoint seed ranges per split). Each
run's event ids and millisecond timestamps are kept. The original NumPy export kept only the order; timestamps were dropped
because they were a distractor for LSTMs.

**Splits:** `train_clean`, `val_clean`, `val_faulty`, `test_clean`, `test_faulty`. Fault type is uniform over the two
kinds, and the target module is uniform within its kind.

## Realistic scale tiers (user direction, 4 Oct 22:00 UTC)

A realistic deployment has years of logs from many parallel lines. One 30-item run lasts about 50 minutes, so one line
running continuously produces about 31K runs (about 33M process events) in 3 years.

| Tier | Clean training runs | Process events | Equivalent |
|---|---:|---:|---|
| S (v1, development) | 10K | ~10.6M | about one line for one year |
| M | 100K | ~106M | about 3 lines for 3 years; comparable to our 90M language scale |
| L | 1M | ~1.06B | about 30 lines for 3 years |

Generation costs about 15 ms per clean run on one core, so L is about 4 CPU-hours. M and L need a compact store
(uint8 ids, uint32 millisecond offsets per run) and data streamed from disk during training.

**Plant-stream variant (v2):**
- K lines (for example K = 4) write one plant log, merged by timestamp. Event ids are line-qualified (46 × K) or shared
  with a line tag in the content.
- This is interleaving at two levels, items within a line and lines within the plant, and is closer to a real plant
  historian.
- A fault in one line must be detected from the merged stream.
- Prefixes count process events of the faulty line's plant, so N stays comparable.

## Task and metric (v1)

**One-class protocol** (simulator README): models are trained on clean runs only and never see faulty runs.
- Model selection uses `val_clean` likelihood only; `val_faulty` is used solely for development AUROC reports.
- Each test run receives an anomaly score computed from its prefix up to the N-th process event,
  N ∈ {32, 64, 128, 256, 512, 1024}.
- The metric is AUROC (faulty = positive) on `test_clean` vs `test_faulty`, at each N, overall and per fault type.

**Detection from fault-free training (the core test; user, 4 Oct 22:00 UTC):**
- Every model learns only the normal behaviour of the line.
- The faults (two kinds, any of the 11 conveyors or 4 bowl feeders) are never shown.
- Detection must come from deviation from learned normal operation, as in the simulator paper.

**Fault localization (secondary):**
- For each faulty test run, attribute the anomaly to a module: the module whose events carry the largest mean excess
  NLL over the clean-run average for that event type, within the prefix.
- Score: top-1 and top-3 accuracy against the injected target, at each N. Chance is 1/15.
- Only the anomaly score's per-event decomposition is used, with no fault labels in training or calibration.

**Tracks:**
- **Timestamped** (primary): events with their times.
- **Order-only:** the original format, in which time survives only through TICK counts.

**Why prefixes:**
- A faulty run lasts longer and emits many more TICKs, so full-run length is a trivial detector.
- Small N measures early detection, the useful and hard regime.
- Process-event counts are nearly constant across runs, so N is a fair unit.

**Compute:** report training and inference work per event with the project's conventions (traced for native, shape
estimates for dense references), and inference work up to the decision at N.

## Models

| Group | Models | Where |
|---|---|---|
| Classical | elapsed time at N; TICK count; per-bigram gap z-score; order-only trigram NLL; HMM (the simulator's companion fas_hmm baseline) | curie (`experiments/fas/baselines.py`) |
| Dense neural | LSTM, Transformer next-event models with log-gap regression (NLL score) | AWS (dense training reserved for AWS) |
| Temporal point processes | Neural Hawkes / Transformer Hawkes-style models (current state of the art for timestamped event streams) | AWS |
| Native | integrated core: event content plus native timestamps as computational time, next-event type and timing NLL as score | curie |

**Win definitions** follow WIN_CRITERIA.md:
- A higher AUROC at the same N with no more training and inference compute is a matched-compute win.
- A higher AUROC at smaller N (earlier detection) at matched compute is the headline target.
- Classical baselines must not already saturate the chosen N. If they do, the benchmark moves to a harder setting
  (smaller N, retry-only faults, or milder fault growth) declared before test scoring.

## Status

- 4 Oct 21:50 UTC: simulator vendored, seeded generator written, dataset `fas_v1_20261004` generated
  (10K / 1K+1K / 2K+2K runs, 68 MB; manifest copied to `experiments/results/fas/fas_v1_20261004_manifest.json`).
- 4 Oct 22:15 UTC: classical baselines on test (`experiments/results/fas/fas_v1_classical_test_20261004T221500Z.json`;
  fit on 2,000 clean runs). AUROC, all faults, by prefix N of process events:

| Baseline | N=32 | N=64 | N=128 | N=256 | N=512 | N=1024 |
|---|---:|---:|---:|---:|---:|---:|
| elapsed | 0.504 | 0.532 | 0.566 | 0.542 | 0.680 | 0.822 |
| tick_count | 0.503 | 0.513 | 0.545 | 0.533 | 0.638 | 0.821 |
| gap_z | 0.504 | 0.511 | 0.530 | 0.559 | 0.727 | 0.998 |
| ngram3 | 0.523 | 0.517 | 0.531 | 0.551 | 0.715 | 0.955 |

  Every classical method is near chance up to N = 256. N = 1024 is saturated by the gap z-score, because it nearly
  covers the whole run. The discriminating range is N <= 512, where early detection is the open problem.
- 4 Oct 22:20 UTC: first native arm queued on curie (`curie_fas_native_p32d4_20261004T222000Z`: p32/d4, linear route
  credit, 2 epochs, clean-only training). Dense, SSM and point-process references: next, as AWS jobs.
- 5 Oct 01:00 UTC (Docker review host): **stronger one-sided timing controls do not help** (`experiments/fas/strong_baselines.py`,
  `experiments/results/fas/fas_v1_strong_classical_test_20261005T010000Z.json`). Locally regenerated subset
  `fas_v1_20261004_subset`: test splits are byte-identical to the official manifest (sha256), the fit uses the same first 2,000
  clean runs, and gap_z reproduces exactly. Hypothesis: faults only add delay, so a directional, distribution-free gap score
  should beat squared gap_z. Result: AUROC (all faults).

| Control | N=128 | N=256 | N=512 | N=1024 |
|---|---:|---:|---:|---:|
| gap_z (repro) | 0.530 | 0.559 | 0.727 | 0.998 |
| gap_quantile (signed, per-bigram empirical CDF) | 0.523 | 0.525 | 0.508 | 0.608 |
| gap_cusum (one-sided CUSUM of quantiles) | 0.514 | 0.521 | 0.562 | 0.783 |
| gap_robust_z (signed median/MAD) | 0.540 | 0.549 | 0.557 | 0.748 |

  Falsified: directional gap shift is not the main early signal. gap_z's power at N >= 512 comes from its squared and
  unseen-bigram terms. Faults mainly change which events become adjacent: a delayed step reorders the interleaving of
  concurrent items, a first-arrival (race) effect. This is consistent with the benchmark's purpose and is not evidence for any model.
  Implication for references: an order-and-timing model of adjacency (e.g. timed n-gram likelihood over the interleaving,
  or the point-process references) is the relevant strong control. All controls remain near chance at N <= 256.
- 5 Oct 01:20 UTC: **order-and-timing control** (`experiments/fas/timed_ngram.py`,
  `experiments/results/fas/fas_v1_timed_ngram_test_20261005T011500Z.json`). AUROC (all faults):

| Control | N=128 | N=256 | N=512 | N=1024 |
|---|---:|---:|---:|---:|
| timed_ngram (process-event trigram + per-bigram gap density) | 0.526 | 0.550 | 0.722 | 0.991 |
| order3 (process-event trigram only) | 0.526 | 0.550 | 0.720 | 0.992 |

  Order alone carries the classical signal: order3 ≈ gap_z ≈ timed_ngram. Timing adds nothing measurable on top.
  **Open question before claiming an early win:** whether N <= 256 is information-limited for every model. Five diverse
  controls sit at 0.51–0.55 there, and fault effects grow as e^(t/5): about 10% of one step's duration by the ~7th use.
  **Backlog (cheap, no training):** an identity-aware oracle bound. Regenerate the test seeds with item identity kept from the simulator,
  and score per-item step durations against clean per-step distributions (likelihood ratio). If even the oracle is near
  chance at N <= 256, define the discriminating range as 384–768 and report the native arm there. If the oracle is high,
  the gap between it and identity-free controls is the home-field headroom for race models.
  Feasibility note: log lines carry no item id, and `env.active_process` at log time yields ~51 processes per clean run
  (item sub-processes), not 30 items. The oracle needs item identity propagated through the component calls, in an
  instrumented copy outside the vendored tree. The instrumented logger leaves event ids unchanged (checked on one seed).
- 5 Oct 00:01 UTC: **first native result**
  (`experiments/results/fas/curie_fas_v1_native_p32d4_linear_seg128_t1024_e2_20261004T231500Z.json`).
  - Model: p32/d4 with linear route credit, 111,456 parameters.
  - Training: carried-state segments of 128 events over the first 1,024 events of all 10K clean runs, 2 epochs,
    7,500 events/s.
  - Selection: epoch 2, by validation-clean NLL.

  Test AUROC, all faults, by prefix N:

  | Model | N=32 | N=64 | N=128 | N=256 | N=512 | N=1024 |
  |---|---:|---:|---:|---:|---:|---:|
  | Native p32/d4 | 0.521 | 0.526 | 0.562 | **0.600** | **0.742** | 0.994 |
  | Best classical at that N | 0.523 (ngram3) | 0.532 (elapsed) | 0.566 (elapsed) | 0.559 (gap_z) | 0.727 (gap_z) | 0.998 (gap_z) |

  - Native is ahead at N = 256 (+0.041) and N = 512 (+0.015), and tied with the best classical baseline below that.
  - Single seed; dense, SSM and point-process references are pending on AWS.
  - Next:
    - save the weights;
    - fixed score decompositions (event-type NLL, timing NLL, windowed maximum), declared before test;
    - an estimate of the early-detection ceiling. Early on, wear faults add 0.7% delay against 1% timing noise, so
      small N may carry little signal for any model.
- 5 Oct 00:15 UTC: **scoring rules declared before any further test scoring** (`native.RULES`, shared by
  dense.py):
  - `total`: mean NLL over the prefix; the primary rule.
  - `type`: event-type NLL only.
  - `gap`: timing NLL only.
  - `gap_window32_max`: the maximum over 32-event windows (or the whole prefix if shorter) of the mean timing NLL.

  All four are reported for every model; the primary rule stays `total`. The native driver now saves its selected
  weights. The scoring functions pass numpy checks. A full torch smoke run waits for memory headroom (P0-2 is running
  with about 0.9 GB above the floor).
- 5 Oct 03:40 UTC (Docker review host): **native vs every classical control, including the stronger ones.** At N=256 the native
  0.600 exceeds all six identity-free controls (gap_z 0.559; order3/timed_ngram 0.550; gap_robust_z 0.549; gap_quantile 0.525;
  gap_cusum 0.521). At N=512, native 0.742 vs best 0.727 (gap_z; timed_ngram 0.722). Test-sampling significance
  (Hanley–McNeil SE, 2,000 clean + 2,000 faulty runs, treated as independent, which is conservative for paired scores): N=256
  +0.041 = 3.2 sigma; N=512 +0.015 = 1.4 sigma (not significant). Evidence level: exploratory, single training seed. The
  dense/SSM/point-process references (AWS) and a second native seed decide the claim. Suggested next: seed 7 of the same
  configuration, and paired bootstrap on per-run scores once the native driver saves them.
- 5 Oct 06:30 UTC (Docker review host): **identity-aware oracle bound: the early range is not information-limited**
  (`experiments/fas/oracle_bound.py`, `experiments/results/fas/fas_v1_oracle_bound_test_20261005T060000Z.json`). Item identity is
  recovered from SimPy sub-process creation: all 30 items, 35–36 events each, none unassigned, and every regenerated test run is
  verified byte-identical (ids and times) to the official data. The oracle scores each item's own step-to-step durations against
  clean training durations (robust z; fit on the same 2,000 clean seeds). AUROC, all faults:

| Detector | N=64 | N=128 | N=256 | N=512 | N=1024 |
|---|---:|---:|---:|---:|---:|
| oracle_sq (identity-aware) | 0.611 | 0.684 | 0.755 | 0.885 | 1.000 |
| oracle_max_step (identity-aware) | 0.592 | 0.679 | 0.755 | 0.913 | 1.000 |
| native p32/d4 (identity-free) | 0.526 | 0.562 | 0.600 | 0.742 | 0.994 |
| best classical (identity-free) | 0.532 | 0.566 | 0.559 | 0.727 | 0.998 |

  Signal exists from N≈64. Share of the oracle's above-chance AUROC recovered at N=256: native 39%, best classical 23%; at
  N=512: native 59%. The remaining gap is the cost of interleaving (unknown item identity), i.e. a de-interleaving problem:
  per-process addressed state and first-arrival races are the architecture's stated answer. Retry faults are the most
  detectable for the oracle (N=256: 0.845), wear the least (0.672–0.700). Implication for native design: routing each event
  to a per-item slot (the item boundary is inferable from the 31-step order) is the mechanism to test. Pool size and
  addressing are the levers, not width. The oracle is a bound, not a competitor: it uses identity the benchmark hides.
- 5 Oct 07:37 UTC: **recruitment sweep R0** (pool 8 untied, 1 epoch;
  `curie_fas_v1_recruit_R0_pool8_untied_e1_20261005T033000Z.json`).
  - Test AUROC, primary rule: .541 at N = 128, .578 at 256, .721 at 512. The timing-only and windowed rules are lower.
  - **Occupancy:** almost every slot is written in every run (8/8 in three layers, ~5/8 in one) and no slot is dead.
    *The §419 dormancy picture does not hold for FAS at pool 8*: the slots are used.
  - **Learned memory half-lives** (base rate): median ~7 s in every layer. An item's events are 5–76 s apart and it
    takes ~720 s through the line.
  - Revised interpretation: the bottleneck is memory horizon plus per-item binding, not unused slots. Forgetting
    substitutes for routing (§419); a 7-second memory cannot carry an item across its own steps.
  - Added arm **R8**: tied pool 8 with timescales initialized log-spaced from 1 to 1,000 s (`--tau-max 1000`). It runs
    first after the sweep, and its learned half-lives are reported to test whether long horizons survive training.
- 5 Oct 07:55 UTC (Docker review host): **memory horizon the detectable signal needs** (identity recovery from oracle_bound.py; 150
  clean training seeds). Item-own consecutive event gaps, the quantity the oracle scores: median 29.4 s, p90 46.6 s, max 606 s;
  67% exceed 7 s, 56% exceed 20 s, 6% exceed 60 s. Mean fraction of a write retained after one own-gap: half-life 7 s 0.38,
  30 s 0.65, 100 s 0.85, 300 s 0.94. At the median gap, R0's learned ~7 s half-life retains about 6% (0.5^(29/7)). This
  quantitatively supports curie's R0 diagnosis (horizon plus binding, not dormancy). Target learned half-lives are ≥ 30–100 s;
  R8's 1–1,000 s initialization spans that range. The test is whether training keeps them long. If training shortens them
  again, a horizon-preserving constraint or prior on the slowest timescales is the next arm.
- 5 Oct 09:46 UTC: **R1 (tied) and R3 (tied + free-slot bonus 3)**, test AUROC at N = 256: .567 and .566, against
  R0 (untied) .578. Slots are fully used in all three, and half-lives stay ~8 s.
  - Tying costs a little capacity. The free-slot bonus changes nothing, as expected once slots are already used.
  - The recruitment arms (R4–R7) were moved after the horizon arm R8 and the grokking testbed.
  - R4's first attempt was lost to an operator error during the reorder (orphaned job stopped; no result written). It
    is rerun after R7.
  - R8 (timescales initialized from 1 to 1,000 s) started 09:45.
- 5 Oct 10:45 UTC: **R8** (tied pool 8, timescales initialized from 1 to 1,000 s).
  - **Long horizons survive training:** median learned base half-life ~24 s, p90 ~350 s, maximum ~650 s (R0/R1: ~8 s).
  - Validation-clean NLL .947, better than tied R1's 1.032.
  - **Detection does not improve:** test AUROC .573 at N = 256 and .717 at N = 512 (R0 .578 / .721).
  - Conclusion: horizon is not the bottleneck by itself. A long-lived slot holding a blend of items does not show when
    a *specific* item is late. The remaining gap to the identity-aware oracle (.755 at N = 256) is **per-item binding**.
  - Next diagnostic: per-slot write purity against the oracle's item identities (diagnostic only; identities never enter
    the model). Then original-write credit across truncated segments (BENCHMARK_WIN_NEXT_STEPS.md §2).
- 5 Oct 11:00 UTC (Docker review host): **oracle-assisted FIFO matches the identity-aware oracle (historical fair-reference interpretation superseded)**
  (`experiments/fas/deinterleave_baseline.py`, `experiments/results/fas/fas_v1_deinterleave_test_20261005T110000Z.json`).
  Oracle-assisted: learn the item route using hidden TRAIN item identities from clean runs (35 types; 66% of clean items follow it exactly), then assign each process
  event online to the waiting item whose next expected type matches, FIFO. On 2,000 clean + 2,000 faulty test runs, item-own
  pair accuracy is **1.000** and 99.0–99.4% of events are assigned. Scoring the inferred item-own durations as the oracle does gives
  AUROC identical to the oracle: **0.755 at N=256, 0.913 at N=512** (native pool 2: 0.600 / 0.742; R0/R1/R3/R8: 0.566–0.578).
  Historical consequences below are superseded by the protocol correction at the top: privileged training structure prevents a fair-reference loss verdict. Scores remain valid oracle-assisted diagnostics.
  1. The deck/valuation line "earlier fault detection than every classical method" is withdrawn. The structure-aware
     classical control beats native by .155 at N=256. The earlier comparison (native > six identity-free generic controls)
     stays true as stated, but it is not the strongest classical control.
  2. FAS v1's interleaving is not ambiguous: items never overtake at a station, so FIFO binding is exact. A race-model
     advantage on v1 would require native ≥ 0.755; the native binding gap (R8: horizon fixed, AUROC unchanged) is real.
  3. Design target for the native arms: learn this binding (route each event to the slot of the item whose last event is its
     route predecessor). Diagnostic: train on clean runs and measure whether race winners track true item identity
     (winner-to-item mutual information from oracle_bound.run_with_identity).
  4. Benchmark revision (FAS v2) where FIFO binding breaks: several product variants with different routes, stations that allow
     overtaking (parallel machines), and dropped/duplicated events. Rerun the oracle and this de-interleaver first. The
     home-field claim belongs where structure-aware rules lose information.
- 5 Oct 11:45 UTC (Docker review host): **FAS v2 probe: two merged lines + 2% event dropout breaks structure-aware binding**
  (`experiments/fas/v2_probe.py`, `experiments/results/fas/fas_v2_probe_drop02_20261005T113000Z.json`; 300+300 runs, AUROC SE ≈ .023).
  Construction without simulator changes: runs s and s+5,000,000 of the same split are merged by timestamp (the first faulty
  for positives), and each process event is dropped with p=.02 (seeded). FIFO de-interleaver pair accuracy drops to 0.72
  (clean) / 0.71 (faulty).

| v2 detector | N=128 | N=256 | N=512 | N=1024 |
|---|---:|---:|---:|---:|
| oracle (true line+item identity, max-step) | 0.572 | 0.598 | 0.673 | 0.792 |
| Oracle-assisted FIFO (hidden TRAIN identities) | 0.517 | 0.502 | 0.498 | 0.530 |

  The structure-aware rule collapses to chance while signal remains for a model that binds correctly. This is the property
  v1 lacked. Proposal for v2 before any native run: (1) freeze a seeded v2 generator and dataset (2,000+2,000 test); (2) rerun the
  six generic classical controls and this de-interleaver, plus a timing-aware de-interleaver (assign to the candidate whose
  elapsed time since its last event is most typical). That is the stronger rule a reviewer would try, and it must be run
  before claiming headroom; (3) only then native arms and AWS references. The oracle bounds the discriminating range.
- 5 Oct 12:45 UTC (Docker review host): **v2 probe with the timing-aware de-interleaver**
  (`fas_v2_probe_timed_drop02_20261005T123000Z.json`; same 300+300 runs; the oracle reproduces the first probe exactly).
  The timing-aware rule chooses, among items waiting for the event's type, the one whose elapsed time since its last event
  is most typical. Pair accuracy 0.86 (FIFO 0.72).

| v2 detector | N=128 | N=256 | N=512 | N=1024 |
|---|---:|---:|---:|---:|
| oracle | 0.572 | 0.598 | 0.673 | 0.792 |
| Oracle-assisted timing-aware tracker (hidden TRAIN identities) | 0.512 | 0.587 | 0.568 | 0.679 |
| FIFO de-interleaver | 0.517 | 0.502 | 0.498 | 0.530 |

  The timing rule takes nearly all of the oracle's signal at N=256; ~0.10 of headroom remains at N=512–1024. A v2 claim must beat
  the timing-aware rule, not FIFO. To widen the discriminating range before freezing v2, probe harder settings: 3–4 lines,
  drop 5%, and line-specific speed offsets (which make timing ambiguous). Procedural note: an intermediate run of this probe
  fitted clean statistics from one training run because of an indentation error introduced by an edit. It was
  discarded unpublished, and the fixed version is above.
- 5 Oct 14:30 UTC: **FAS v2 confirmatory protocol pre-registered** ([FAS_V2_CONFIRMATORY_PROTOCOL.md](FAS_V2_CONFIRMATORY_PROTOCOL.md);
  queue manifest `queue/fas_v2_confirmatory_20261005T143000Z/manifest.json`). The v2 setting is selected from the K/drop/speed
  grid by oracle-vs-classical headroom on validation, before any learned model runs. Win rule: native seed-mean AUROC at
  N=512 ≥ strongest baseline + 0.02, paired-bootstrap 95% lower bound > 0, every native seed above the strongest baseline.
- 6 Oct (curie FAS session, B3 Stage 0): **v2 generator frozen in code**. `experiments/fas/generate_v2.py` implements
  the protocol's Stage 0 generator:
  - K lines, line 0 faulty;
  - seeds `seed_base + 10M·split + K·run + line`;
  - dropout from line-seed RNG only;
  - ±δ alternating speed;
  - one 10 s plant clock;
  - seeded random tie-break;
  - no line tag;
  - `identity.npz` sidecar.

  Contracts: `tests/test_fas_generate_v2.py` (5 pass). No v2 data has been generated yet.

  A beam de-interleaver (`beam_deinterleave.py`) was started as the protocol's tracker. It is reclassified as a
  **structure-assisted diagnostic**, because its design used identity-inspected route structure (privileged process
  knowledge; user direction). It is excluded from the Stage 1 gate and from fair-reference selection, and its
  development has stopped.
  - Measured pair accuracy: ~0.49 at K=1 and ~0.26 at K=2 with drop .02.
  - With true-track parameters: 0.96 and 0.62.

  Theory: note 155 §432 (interleaving symmetry; a superposition of timed processes is a race).
- 6 Oct (curie FAS session, B3 Stage 0, continued): **driver contracts**.
  - `native.py` and `dense.py` now write `<tag>_scores.npz` with per-run scores for every rule and prefix. Validation
    scores come from the selected epoch; test scores only when test is scored. These support the protocol's paired
    bootstrap.
  - `dense.py` also saves its selected weights under `checkpoints/`.
  - `experiments/fas/causality_probe.py` is the suffix-mutation probe. Untrained-architecture results
    (`results/diagnostics/causality_untrained_*_20261006T1700Z.json`, 4 runs, cut 200) pass exactly, with zero
    difference for N ≤ cut: LSTM, Transformer, Mamba, S5, native race, carried native, and carried native with
    expected reception.
  - LRU differs by 4.4e-6 for N ≤ cut, against 8.6 for N > cut. This is float32 rounding in its causal FFT convolution,
    not a leak. It passes at tol 1e-5; both records are kept.
  - Trained models are probed from their result JSON with the same script.
- 6 Oct 20:07 UTC (curie FAS session, B3 Stage 1): **amended calibration complete; all 12 settings qualify**
  (oracle .747–.821; gap to the best anonymous classical detector .106–.198; per-line prefixes, line-aware oracle).
  - The original merged-prefix grid qualified none, as predicted.
  - Selected: K = 2, p = .02, δ = 0 (oracle .821 vs order3 .685).
  - Stage 2 data `fas_v2_K2_drop0.02_delta0_20261005` is generating. Development log: `experiments/fas/B3_DEVELOPMENT_LOG.md`.
