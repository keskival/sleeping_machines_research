# FAS interlaced-event anomaly benchmark (opened 4 October 2026)

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
