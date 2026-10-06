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
| C1 | race readout + binding memory U_b = 40 + step-class mixture M = 3 | §§433–436 | main candidate |
| C2 | standard native head (native.py, pool 8) | — | our previous architecture on v2 |
| C3 | C1 without step classes (single duration law per slot) | §435.2 | does type–duration coupling pay at p = .02? |
| C4 | C1 with U_b = 24 (below KλW ≈ 36) | §434.3 | does capacity below the Little's-law bound cost AUROC? |
| C5–C8 | reserved for iterations after error analysis of C1–C4 | | |

Evaluation-only on the selected C1 checkpoint: particle evaluation, L = 1, 4, 16 (§434.1.2). This does not count
toward the budget, because it adds no fitted configuration.

Target: native validation AUROC at merged N = 1,024 above .685 (order3) by more than the seed spread, approaching the
oracle's .821.

## Log

- 6 Oct 17:10 UTC: plan declared. The v2 data (Stage 2) is queued behind the amended calibration grid. Training
  waits for a curie window or an AWS gym slot.
- 6 Oct 17:20 UTC: error analysis ready (`readout_diagnostics.py`, evaluation only, validation identities used only for
  diagnosis). It reports binding purity α (the α of §434.2), item concentration and the time-rescaling fit (§435.3:
  KS of rescaled intervals, tie fraction), plus AUROC split by binding purity. It is queued right after C1 and its
  particle evaluation.
  - Data note: over a whole run, faulty samples carry ~7× more plant-clock ticks, because the faulty line runs far
    longer and the single clock runs to the last event. Process-event counts are equal. Within a prefix this is
    legitimate elapsed-time evidence, already available to the `elapsed` and `tick_count` references.
