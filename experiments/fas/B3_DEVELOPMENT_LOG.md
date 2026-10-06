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
| C1 | race readout (hidden 64) + gated binding memory U_b = 40 + step-class mixture M = 3 | §§433–436 | main candidate |
| C2 | standard native head (native.py, pool 8) | — | our previous architecture on v2 |
| C3 | C1 without step classes (single duration law per slot) | §435.2 | does type–duration coupling pay at p = .02? |
| C4 | C1 with U_b = 24 (below KλW ≈ 36) | §434.3 | does capacity below the Little's-law bound cost AUROC? |
| C5–C8 | reserved for iterations after error analysis of C1–C4 | | |

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
