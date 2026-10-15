# FAS v2.1 sealed confirmation: delay-race binding (preregistered 15 October 2026, before the test draw exists)

**Decision.** Does the native delay-race binding member detect FAS v2 faults earlier than every eligible reference, by
the frozen B3 Stage 5 rule, on a fresh sealed test draw?

**Why a new draw.** The v2 TEST was scored once on 10 October (native C10 0.7037 vs Transformer 0.7043, registered tie)
and is spent. Development since then used TRAIN, validation and a separate clean calibration draw only. A fresh draw from
the frozen generator, created after this document is committed, is the only clean test for the new member.

## Data (frozen generator, frozen setting)

- Generator `experiments/fas/generate_v2.py`, setting K = 2 lines, event dropout p = .02, speed offset δ = 0 (the
  amended Stage 1 selection), 30 items per line.
- Fresh sealed test: `--name fas_v2_K2_drop0.02_delta0_test21_620261016 --seed-base 620261016 --train 0 --val 0
  --test 2000` (2,000 clean + 2,000 faulty runs). Seeds are seed_base + 10,000,000·split + K·run + line, so this draw
  is disjoint from every v2 split (all v2 seeds lie below 7.1e7) and from the calibration draw (seed base 520261015).
  `identity.npz` of the new draw is opened only by the reported oracle ceiling.
- Clean calibration draw (already generated, clean runs only used): `fas_v2_K2_drop0.02_delta0_calib_520261015`.

## Native member (frozen before the draw)

`experiments/fas/race_delay_binding.py`, configuration selected on v2 validation (1,000 + 1,000 runs) among a declared
grid (top 2/3 clocks, consumption on/off, one/two-sided evidence, 1,000/5,000 clock-learning runs, κ ∈ {0, .25, .5, 1, 2}):
**top 2 predecessor clocks, window 64, no consumption, one-sided evidence, κ = 0.5, residual clip 8**. Validation AUROC at
the primary prefix: **0.8004** (wear 0.779, retry 0.821). The Transformer reference scores 0.7036 on the same runs.

Mechanism: delay clocks learned from anonymous clean TRAIN logs (per event-type pair log-lag modes); binding by a
temporal race of candidate predecessors; per-step standardized residuals (calibration draw) accumulated in addressed
per-step CUSUM slots; run score = the most anomalous slot. No item identities are used anywhere.

Replicates ("seeds"): three independent clock fits on disjoint TRAIN slices 0:1000, 1000:2000 and 2000:3000; the
calibration and scoring rule are identical.

**Restriction, with reason:** clocks, standardization and weights are fixed during scoring. A fault is a growth of one
step's delay inside a run; adapting to it during the run would learn the fault as normal. Persistent per-run evidence
(the CUSUM slots) is the within-run state.

## References (all scored once on the fresh draw)

- Time-encoded Transformer d128 lr .003, seeds 0–2, and LSTM d128 lr .003, seeds 0–2: the v2 selected checkpoints,
  unchanged. Each family is scored with the v2 primary rule (`total`) and with the addressed per-type statistics of
  `experiments/fas/addressed_evidence.py`; the family's statistic is chosen on v2 validation by AUROC at the primary
  prefix (deterministic, inside the scoring job), so references get the same detector-selection opportunity.
- The nine information-matched classical detectors of `classical_scores_v2.py`, fitted on 2,000 clean TRAIN runs.
- Reported, not eligible: native neural C10 refits (seeds 6–8, if complete), identity-aware oracle (ceiling).

## Endpoint and decision (B3 Stage 5 rule, unchanged)

Primary endpoint: AUROC at merged prefix 1,024 (N* = 512 per line), all faults, native replicate mean.
Strongest baseline: the highest test AUROC among eligible references (family seed mean for learned references,
maximum over their two detector statistics; deliberately favours the baselines).
**Win** if (1) native mean − strongest ≥ 0.02; (2) paired stratified bootstrap (10,000 resamples of clean and faulty
runs, seed-averaged ranks on both sides) 95% lower bound > 0; (3) every native replicate exceeds the strongest
baseline's point estimate. Otherwise tie or loss, reported as plainly.

Secondary: per fault type, prefixes 256 and 512, and work per run. Scoring job:
`experiments/fas/b3_v21_score.py` (committed with this document); machine-readable copy:
`experiments/fas/FAS_V21_CONFIRMATION.json`.
