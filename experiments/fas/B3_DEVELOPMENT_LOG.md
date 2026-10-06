# B3 development log (FAS v2; owner: curie FAS session)

Protocol: `experiments/FAS_V2_CONFIRMATORY_PROTOCOL.md` (with its amendments). The test set is sealed. Development uses
training and validation only. Selection is by validation-clean NLL; validation AUROC is reported separately.

## Setting and references (Stage 1, validation, no learned model)

The amended rule selects **K = 2 lines, event dropout p = .02, no speed offset**. Prefixes are counted per line, and the
merged prefix is N·K.

| detector (v2 validation, 1,000 + 1,000) | N = 512 per line (merged 1,024) |
|---|---|
| identity oracle, line-aware max-step | .821 |
| best information-matched classical (order3) | .685 |
| gap | .136 |

Full grid: `results/fas/fas_v2_calibration_amended_grid_20261006T1550Z.json` (running).

## Native development plan (budget: 8 configurations on v2 validation)

All configurations share these settings:
- deep race network p32, depth 4, heads 2, pool 8, τmax 1000, linear route credit, segment 128;
- lr .003, 1 epoch, 5,000 training samples (~10.5M process events), seed 6;
- `--train-max-events 2048 --max-events 2200`.

The common training cap C is fixed from the C1 smoke.

| id | configuration | theory | question |
|---|---|---|---|
| C1 | race readout (hidden 64) + gated binding memory U_b = 54 (1.5 × anonymous peak concurrency) + step-class mixture M = 3 | §§433–436 | main candidate |
| C2 | standard native head (native.py, pool 8) | — | our previous architecture on v2 |
| C3 | C1 without step classes (single duration law per slot) | §435.2 | does type–duration coupling pay at p = .02? |
| C4 | C1 with U_b = 36 (exactly the anonymous peak, no headroom) | §434.3 | does binding need spare capacity (toy: α .48 vs .90)? |
| C5 | C1 with additive readout context (per-slot laws computable once per write) | §436.3 | does the low-cost readout match C1? (toy: α .922 vs .934) |
| C6–C8 | reserved for iterations after error analysis of C1–C5 | | |

Evaluation-only on the selected C1 checkpoint: particle evaluation, L = 1, 4, 16 (§434.1.2). This does not count
toward the budget, because it adds no fitted configuration.

Target: native validation AUROC at merged N = 1,024 above .685 (order3) by more than the seed spread, approaching the
oracle's .821.

## Log

- 6 Oct 17:05 UTC: plan declared. The v2 data (Stage 2) is queued behind the amended calibration grid. Training
  waits for a curie window or an AWS gym slot.
- 6 Oct 17:09 UTC: error analysis ready (`readout_diagnostics.py`, evaluation only, validation identities used only for
  diagnosis). It reports binding purity α (the α of §434.2), item concentration and the time-rescaling fit (§435.3:
  KS of rescaled intervals, tie fraction), plus AUROC split by binding purity. It is queued right after C1 and its
  particle evaluation.
  - Data note: over a whole run, faulty samples carry ~7× more plant-clock ticks, because the faulty line runs far
    longer and the single clock runs to the last event. Process-event counts are equal. Within a prefix this is
    legitimate elapsed-time evidence, already available to the `elapsed` and `tick_count` references.
- 6 Oct 17:39 UTC: **small integrated fit of the binding mechanism** (`binding_toy.py`). Synthetic data: 3 interleaved
  processes, 8-step routes, log-normal steps with CV .15, no dropout. Model: binding memory 6 slots + race readout,
  2 step classes, trained from scratch, 300 steps of 16 samples.
  - Binding purity α rose from chance .27 to .71 with additive slot writes.
  - With the per-dimension overwrite gate it reached .75 (sampled writes) and .81 (argmax writes); validation NLL fell
    from 4.87 to 1.94.
  - The Bayes-greedy ceiling (true routes and laws, `binding_toy_oracle.py`) is α = .911, so 300 steps close ~75% of
    the gap from chance to that ceiling.
  - Gated writes go into C1-family iterations once compared on v2 (§436.2: a slot must be able to hold its process's
    latest state). A longer fit (1,200 steps) is running.
- 6 Oct 17:46 UTC: binding toy, 1,200 steps (gated): validation NLL 1.97 → 1.77, but α plateaus at .78–.80 (sampled)
  and .80–.84 (argmax), against the .911 ceiling. The remaining binding gap is not a step-budget gap. Testing
  hard-EM (argmax) training writes and a larger readout (hidden 64).
- 6 Oct 17:46 UTC: binding toy arms (600 steps, gated writes):
  - hard-EM (argmax) training: α .82 (no gain);
  - **readout hidden 64: α .934 with argmax writes (.88 sampled)**, above the Bayes-greedy ceiling .911, with
    validation NLL 1.73.

  The binding gap was readout capacity; C1 already uses hidden 64. A no-context ablation (§436.3) is running.
- 6 Oct 17:46 UTC: **small neural references reinstated** (user direction; AGENTS.md exception, protocol amendment).
  LSTM and time-encoded Transformer, d ∈ {64, 128} × lr ∈ {3e-4, 1e-3, 3e-3}, at most 3 passes over the same 5,000
  clean samples, validation only. Two smokes and 12 grid queues are prepared for AWS (`aws_fas_v2_ref_*_20261006T1815Z`).

  Sizes:

  | model | parameters | forward MFLOP/event (T ~ 2,100) |
  |---|---|---|
  | C1 | 338K | not traced |
  | LSTM d64 / d128 | 73K / 277K | 0.14 / 0.54 |
  | Transformer d64 / d128 | 110K / 425K | 1.3 / 3.0 |

  The Transformer's per-event cost grows with log length; ours is constant per event.
- 6 Oct 17:47 UTC: no-context ablation (§436.3), hidden 32: argmax α .908 (.88 sampled), validation NLL 1.83, against
  α .81 / NLL 1.94 with context at the same size. This supports the shortcut hypothesis: merged-stream context lets
  slots predict without binding.

  The toy processes are independent; FAS processes interact through shared stations, so C1 keeps the context at hidden
  64. If C1's diagnostics show low binding purity, a no-context or context-gated readout is the first C5–C8 iteration.
- 6 Oct 17:49 UTC: hidden 64 without context: argmax α .856, NLL 1.89, worse than hidden 64 with context (.934, 1.73).
  C1's choice of context at hidden 64 stands. Toy arms are single-seed; differences of a few points are within noise.
  Timestamp correction: entries from 17:05 to 17:47 were first written with times 5–40 minutes too late. They now
  carry the commit times, and the same correction is applied to the reference-exception notes (AGENTS.md, protocol,
  AWS_FAS_REFERENCES.md, PRODUCT_ORDERS.md). Queue tags keep their original suffixes; they are identifiers, not times.
- 6 Oct 18:23 UTC: **measured work of the C1 configuration** (`work_c1size_20261006T1820Z.json`; operation audit, one
  eager training window of 8 × 128 events, and inference on 4 validation runs).

  | model | fit, MFLOP/event | inference, MFLOP/event | convention |
  |---|---|---|---|
  | C1 | 5.08 (+0.037 M special) | 1.69 (+0.019 M special) | measured: forward + backward + optimizer |
  | LSTM d64 / d128 | — | 0.14 / 0.54 | shape estimate, forward |
  | Transformer d64 / d128 | — | 1.3 / 3.0 | shape estimate, forward, mean over T ≈ 2,100 |

  The references' fit work is about 3 × forward plus the optimizer (dense.py's convention). The Transformer's
  per-event cost grows with log length; C1's is constant. C1's implementation still computes every deep-layer slot
  proposal densely, so these figures are an upper bound on what the architecture needs.
- 6 Oct 18:45 UTC (user question: do models assume the number of parallel processes?). No model assumes a process
  count. The binding memory has a *capacity* U.
  - Unused slots go silent (pending probability → 0), and finished processes free their slot: gated writes let a new
    process take it over.
  - Only more *simultaneously active* processes than U forces slot sharing. That is graceful mis-binding, with the loss
    of §434.2, not a failure.

  **Sizing fairness:** C1's U = 40 was first set from the identity-measured peak concurrency (35), which is privileged
  information. The anonymous estimate (`anonymous_concurrency.py`: starts of the earliest type minus ends of the latest
  universally visited type; training logs only) gives mean 33.5 / max 36 against identity 34.2 / 35. The **declared
  sizing rule is U = anonymous peak concurrency on training logs + 10%**, which gives 40, unchanged.

  **Declared secondary stress test** (descriptive, outside the decision rule): score C1, trained at K = 2, on a K = 3
  validation-format set (concurrency ~54 > U), against references trained at K = 2. This measures degradation beyond
  capacity.
- 6 Oct 18:45 UTC: toy at 5% dropout (§435.2):
  - 3 step classes against one duration law: validation NLL 2.046 vs 2.143 at the end, 2.045 vs 2.222 averaged over
    steps 400–600, a gain of 0.10–0.18 nats/event (predicted .1–.2);
  - argmax α .917 vs .892.

  C1's 3 step classes stand. Single seed.
- 6 Oct 18:48 UTC: **capacity toy** (6 concurrent processes, Bayes-greedy ceiling α .886, 600 steps):

  | slots | argmax α | validation NLL |
  |---|---|---|
  | 3 | .31 | 2.19 |
  | 6 | .48 | 2.11 |
  | 9 | **.90** | **1.82** |

  Binding needs *spare* capacity, not just capacity equal to concurrency. Early posterior mis-writes need free slots
  to escape into; without them, shared slots lock in. This is the optionality argument of §§422–424 measured on
  binding. The earlier remark that binding gets harder to learn at higher concurrency is withdrawn: the K = 6 arm with
  6 slots had zero headroom.

  **Revised sizing rule: U = 1.5 × anonymous peak concurrency** (v2: 1.5 × 36 = 54). C1 and C3 now use U = 54. C4
  becomes the no-headroom ablation (U = 36). None of these queues has started. Single-seed toy evidence.
- 6 Oct 19:06 UTC: **particle evaluation on the toy** (6 processes, 9 slots, trained 600 steps; THEORY §434.1.2). Validation
  NLL by particle count:

  | particles | 1 | 4 | 16 | 64 | argmax path |
  |---|---|---|---|---|---|
  | NLL | 1.885 | 1.834 | 1.815 | 1.810 | 1.821 |

  The bound tightens monotonically with diminishing returns, as FIVO predicts. Particles are an inference-compute dial:
  L particles multiply inference work by L (C1: 1.69 MFLOP/event at L = 1). Any L > 1 used for v2 scoring is declared
  before test scoring, and its work is reported at that L. C1's particle evaluation (L = 1, 4, 16) on v2 validation
  sets it.
- 6 Oct 19:13 UTC: **sparse inference** (winner-only deep layers, exactly cached reads; contracts 16/16) and C1 work
  re-measured at U = 54 (`work_c1size_u54_20261006T1930Z.json`):

  | C1 work (MFLOP/event) | value |
  |---|---|
  | fit (forward + backward + optimizer) | 6.10 |
  | inference, dense | 2.04 |
  | inference, sparse | **1.50** |

  The readout now dominates inference: 54 slots × (P·h + h·(M·V + 3M + 1)) ≈ 1.2 MFLOP/event. Next work optimisation:
  make the per-slot laws change only on writes, with the context entering as a shared additive logit term (O(1) per
  event), so per-event readout work reduces to the U·M duration CDFs. This changes the context interaction; it is
  tested as a development configuration, not swapped in silently.

  References (forward, shape estimates, T ≈ 2,100): Transformer d64 / d128 = 1.3 / 3.0; LSTM d64 / d128 = 0.14 / 0.54.
- 6 Oct 19:16 UTC: toy, additive readout context (hidden 64, 3 processes): argmax α .922, validation NLL 1.75. Full
  context gave α .934 / NLL 1.73; no context α .856 / NLL 1.89. This is within single-seed noise of full context, at a
  per-event readout cost of only the duration CDFs once the per-slot laws are cached per write. Declared as C5 (budget
  slot 5).
