# FAS v2 confirmatory protocol: interleaved event streams with ambiguous identity

**Protocol correction — 5 October 2026, user-directed:** FIFO is an **oracle-assisted diagnostic**, not an eligible reference for anonymous-process learning. `deinterleave_baseline.learn_route()` uses hidden TRAIN item identities to recover the route. The v2 timing-aware probe additionally fits transition-gap statistics with those identities. Neither receives test identities for prediction, but both receive privileged training structure unavailable to native and generic controls. Their scores are retained as oracle-assisted diagnostic targets; exclude them from strongest-reference selection and win/loss verdicts. Native's completed single-seed win against the six generic controls stands: **0.600 vs 0.559 AUROC at N=256**. A fair structure-learning reference must fit exclusively on the same anonymous training logs. Historical contrary interpretations below are superseded by this correction; numerical records remain preserved.

5 October 2026 · User-directed · Queue: `queue/fas_v2_confirmatory_20261005T143000Z/manifest.json`

## Purpose

Show, in a form that survives investor and reviewer scrutiny, whether the native architecture detects faults earlier
than every strong alternative when several processes write into one log and their identities cannot be recovered by
structure. FAS v1 supplies an oracle-assisted diagnostic: FIFO uses hidden TRAIN item identities to learn its route and reaches the
identity-aware oracle (0.755 at N=256, 0.913 at N=512; native 0.600 / 0.742). The v2 probe showed the property v1
lacked: with two merged lines and 2% dropped events, FIFO binding collapses to chance and a timing-aware de-interleaver
keeps most, but not all, of the oracle's signal (FAS_BENCHMARK.md, 5 Oct 11:45 and 12:45).

The protocol is pre-registered. Everything that can influence the outcome (benchmark setting, primary endpoint,
baselines, tuning budgets, selection rule, decision rule) is fixed below or by a validation-only rule below, before
any learned model sees v2 data.

## What convinces

1. **The benchmark difficulty is set without learned models.** The v2 setting is chosen on validation data from
   the identity-aware oracle, oracle-assisted diagnostics and eligible anonymous-log classical references, by the rule in Stage 1. Native results never influence it.
2. **Fair classical references learn from anonymous logs only.** FIFO, timing-aware and beam trackers using identity-derived routes or timing distributions are oracle-assisted diagnostics. A tracker is eligible as a reference only after its full fitting path passes an anonymous-input audit.
3. **Modern neural references get the same data, inputs, budget and tuning opportunity.** LSTM, Transformer with
   continuous-time encoding, LRU, S5 discretized by real gaps, Mamba-style selective SSM.
4. **One primary endpoint, one decision rule, three seeds, paired uncertainty.** Fixed below.
5. **The test set is sealed.** Every test scoring is logged; each selected configuration is scored once.
6. **An operational metric customers recognise:** detection rate and simulated minutes to alarm at a fixed 1%
   false-alarm rate.
7. **A real-data track** with stripped identities follows the simulator result (Stage 6), so the claim does not
   rest on the founder's simulator alone.

## Stage 0: contracts and freezing (no v2 test access)

- `experiments/fas/generate_v2.py`: frozen v2 generator. Writes the same npz format as `generate.py`
  (ids, times_ms, offsets, fault, target, seed), so `native.py` and `dense.py` consume it unchanged, plus a
  sidecar `identity.npz` (line, item per event) that only oracle and diagnostic code may open.
  - K production runs per sample: the first faulty in `*_faulty` splits, all clean otherwise. Seeds disjoint per
    split and per line (`seed_base + 10_000_000·split + K·run + line`).
  - Merge by timestamp. **Break ties with a seeded random permutation, not stable sort**: stable sort leaks line
    identity whenever timestamps coincide.
  - **One plant clock:** keep a single TICK stream, not K copies.
  - Event dropout: each process event dropped independently with probability p, from an RNG that depends only on
    the run seed, never on fault or identity.
  - Optional per-line speed offset δ: line timestamps scaled by (1+δ) after simulation, applied identically in
    every split.
  - Shared event ids across lines; **no line tag in the primary track.** A line-tagged secondary track may be
    reported separately.
  - Prefix N counts process events of the merged stream. `--max-events` for the learned drivers is set to cover
    N=1024 plus margin (about 1,100·K is safe).
- Driver extensions, contract-tested before use:
  - `native.py` and `dense.py` save **per-run test and validation scores** for every rule and prefix
    (`*_scores.npz`) so paired bootstrap is possible.
  - A suffix-mutation causality probe for every learned model (changing events after position t leaves scores up to
    t unchanged), as in the E171 audit.
- `experiments/fas/beam_deinterleave.py`: proposed classical tracker. It is a fair reference only if route transitions and gap densities are learned exclusively from anonymous logs; otherwise classify it as oracle-assisted. Multiple-hypothesis assignment of each
  process event to a (line, item) track, scoring hypotheses by the clean route-transition model plus per-transition
  gap density; beam width chosen from {16, 64, 256} on validation. Score as the oracle does (item-own durations).
- Record SHA256 of the generator, drivers, scorers and this protocol in the manifest before Stage 2.

## Stage 1: setting calibration (validation only, classical and oracle only)

Generate validation-only calibration sets (1,000 clean + 1,000 faulty samples) over the grid
K ∈ {2, 3, 4}, p ∈ {0.02, 0.05}, δ ∈ {0, ±5% alternating by line}. On each, run: oracle (true identity), oracle-assisted FIFO and timing-aware diagnostics, a beam tracker classified by its fitting-input audit, and the six generic controls (elapsed, tick_count, gap_z, gap_quantile,
gap_cusum, gap_robust_z) and order3/timed_ngram.

**Selection rule.** Primary prefix N* = 512. Choose the mildest setting (smallest K, then smallest p, then δ=0)
that satisfies both:
- oracle AUROC ≥ 0.70 at N*;
- oracle − best information-matched classical reference ≥ 0.08 at N*.

If no setting qualifies, stop: FAS in this form is not a discriminating home-field benchmark, and that is reported
as the result. Publish the full calibration grid either way
(`results/fas/fas_v2_calibration_<tag>.json`).

## Stage 2: dataset

Generate the selected setting as `fas_v2_<setting>_20261005` with 10,000 clean training samples, 1,000 + 1,000
validation, 2,000 + 2,000 test. Publish the manifest and hashes. From here on, the setting is frozen.

## Stage 3: development and selection (training and validation only)

**One-class protocol, as v1:** all learned models train on clean samples only. Primary selection is by
validation-clean NLL. Secondary selection by validation AUROC at N* is reported separately as the
"fault-history-available" variant.

**Equal tuning opportunity:** each learned family gets at most **8 configurations** on v2 validation.
- Native: any of the integrated members (pool, tying, timescale init, addressing, credit, segment length). Prior v1
  development is disclosed, not counted. The binding diagnostic (FAS_BINDING_DIAGNOSTIC.md) may run on training
  data to choose the addressing change; its runs count toward the 8.
- Each neural reference family: width {64, 128} × learning rate {3e-4, 1e-3, 3e-3}, minus one, or another declared
  8-point grid.
- Classical detectors: their few hyperparameters (beam width, n-gram order, window) selected on validation.

**Common training cap:** every selected learned configuration must fit within the same training-work cap C,
fixed in the manifest from smoke measurements before Stage 3 starts. Native work is traced; reference work is shape
estimated; state both conventions in the table. Report actual work for each selected model.

## Stage 4: confirmation (sealed test)

- Train each family's selected configuration with **three seeds** (native 6, 7, 8; references 0, 1, 2).
- Score test once per trained model. Append each scoring to `results/fas/fas_v2_test_ledger.jsonl`
  (tag, config hash, checkpoint hash, UTC time). Any extra test scoring must be declared in the ledger with its
  reason, and appears in the report.
- Classical detectors are deterministic; score each once.

## Stage 5: analysis and decision

**Primary endpoint:** test AUROC at N*=512, all faults, primary scoring rule (`total`), native seed mean.

**Strongest baseline:** exclude every oracle-assisted method (including hidden-identity-trained FIFO and timing-aware probes). Use the highest test AUROC at N* among information-matched classical references and the seed means of all neural
families (taking the maximum on test favours the baselines; this is deliberate).

**Win** if all three hold:
1. native mean − strongest baseline ≥ 0.02 AUROC;
2. paired bootstrap over test samples (10,000 resamples, stratified clean/faulty, seed-averaged scores) gives a 95%
   interval for the difference with lower bound > 0;
3. each native seed individually exceeds the strongest baseline's point estimate.

Otherwise the result is a **tie** (difference within ±0.02 or interval covering 0) or a **loss**, reported in one line
as plainly as a win. A win is a matched-compute win if native training and inference work are no more than the
strongest learned baseline's; otherwise a pure-accuracy win at the stated work ratio.

**Secondary (descriptive, no separate win claim):** AUROC at N ∈ {128, 256, 1024}; per fault type; fault
localization top-1/top-3; the four declared scoring rules; detection rate and median simulated minutes to first alarm,
with the alarm threshold per prefix calibrated on validation-clean so that the any-prefix false-alarm rate is 1%;
share of the oracle's above-chance AUROC recovered.

**Sanity controls, reported beside the result:** timestamps shuffled within each sample (native should degrade if it
uses time), causality probe results, and the line-tagged track if run.

**Report:** one generated table (all methods, AUROC with 95% intervals at each N, training and inference work in the
same units per column, parameters) and one figure (AUROC vs N with the oracle as ceiling). Scoreboard line per
WIN_CRITERIA.md.

## Stage 6: real-data track (after Stage 5, separately frozen)

Identity-stripped real logs, so the claim generalizes beyond the simulator. Candidates:
- **HDFS (Loghub HDFS_v1):** genuinely interleaved system logs with block-level anomaly labels; identity-aware
  per-block models are the oracle.
- **BPI Challenge 2012/2017 event logs:** real process control flow; cases merged by timestamp with case ids stripped;
  anomalies injected with the established process-mining scheme (skip, insert, rework, early, late).

Admit a candidate only if the Stage 1 style gate holds on its validation split (identity-aware oracle minus best
information-matched classical reference ≥ 0.05). Write and freeze a protocol addendum with the same stages before any learned model
runs on it.

## Hosts and order

- Stages 0–2: CPU, numpy/SimPy; curie or the review host. Minutes to a few hours.
- Stage 3–4 native: curie, one job at a time through `run_safe.sh`.
- Stage 3–4 references: AWS slots, after their two-window smokes set RSS caps (≥ 8 GiB MemAvailable kept).
- Admission: first free capacity without preempting running jobs. Every training run gets a uniquely named one-job
  queue created by the physical owner once its stage contracts pass.

## Protocol amendment: oracle-assisted classification, 5 October 2026

This user-directed amendment corrects reference eligibility after auditing hidden TRAIN identity use; it is not the original frozen protocol. Preserve original queue/source pins and historical results. Physical owners must create a new uniquely named, source-bound protocol/manifest before executing an amended stage; existing frozen queues are not silently redefined. Oracle-assisted results remain diagnostic measurements. No training or queue admission was performed by this correction.

## Protocol amendment: no new external-architecture training, 6 October 2026 (user direction)

AGENTS.md ("Own-model focus", 6 October 2026) forbids new Transformer, LSTM or other external-architecture training on
any host, including "missing" controls listed in older orders. For this protocol:
- **Neural reference families are not trained.** The Stage 3–4 cells for LSTM, Transformer, LRU, S5 and Mamba are
  reported as "not run (user direction, 6 October 2026)", never as losses or wins. The AWS FAS reference queues
  (AWS_FAS_REFERENCES.md) are withdrawn. dense.py stays as the published reference implementation for outside
  submissions.
- **Strongest baseline** (Stage 5): the highest test AUROC at N* among the information-matched classical references
  (anonymous-log fitting only). Oracle-assisted and structure-assisted methods remain excluded, as before.
- **Scope of a win:** "native vs information-matched classical detectors". There is no claim against neural
  sequence models until published or submitted results exist under this exact protocol. The public FAS release
  (generator, frozen data hashes, scorer and leaderboard) invites those submissions.
- Everything else is unchanged: setting calibration, native tuning budget, seeds, sealed test ledger and decision
  rule. The native model still develops on validation within its 8-configuration budget.

## Protocol amendment: per-line prefixes and line-aware oracle in Stage 1, 6 October 2026 (curie FAS session)

**Disclosure:** written after the original Stage 1 grid (`fas_v2_calibration_grid_20261006T1745Z`) had reported 3 of
its 12 settings. No learned model has seen v2 data.

The original grid runs to completion unchanged, and its result is reported first. Its first settings at merged
N*=512:

| setting | oracle | best anonymous classical | gap |
|---|---|---|---|
| K=2, p=.02, δ=0 | .683 | .565 | .119 |
| K=2, p=.02, δ=±5% | .554 | .550 | .004 |
| K=2, p=.05, δ=0 | .669 | .567 | .102 |

**Two design errors, each fixed from a principle that holds for every setting:**
1. **Merged prefix.** The fault sits on one line. A merged N* gives every detector N*/K events of that line, so raising
   K also removes data, confounding interleaving difficulty with data volume. **Fix:** N counts process events per
   line: the merged prefix is N·K, with primary N* = 512 per line. Secondary prefixes are 128, 256 and 960 per line;
   960 because each line keeps ≥ ~1,007 events at p=.05.
2. **Pooled oracle statistics.** The identity oracle knows the line but pooled its duration statistics across lines.
   With a ±5% speed offset, that blurs the oracle by as much as the faults themselves (.554). **Fix:** the gate's
   oracle keys statistics by (line, prev type, type) (`oracle_line_max_step`). The pooled oracle stays reported.

Everything else is unchanged: the grid, the thresholds (oracle ≥ .70; oracle − best information-matched classical
≥ .08), the mildest-qualifying rule and every classical reference.
- Implementation: `experiments/fas/calibrate_v2b.py`. Output: `results/fas/fas_v2_calibration_amended_<tag>.json`.
- If the amended grid selects a setting, Stage 2 generates it, and Stages 3–5 measure N in per-line units
  (merged prefix N·K). Native development uses validation only, as before.

## Protocol amendment: small neural references reinstated, 6 October 2026, 17:46 UTC (user direction)

This replaces the "no new external-architecture training" amendment for two families (AGENTS.md exception of the same
time):
- **LSTM** (RMTPP input pattern) and **causal Transformer with continuous-time encoding** (THP input pattern), from
  `experiments/fas/dense.py`, d ∈ {64, 128}, 2 layers. They share the native model's head, loss, clean-only training,
  scoring rules and selection rule (validation-clean NLL).
- **Tuning:** each family runs the 6-point grid d ∈ {64, 128} × lr ∈ {3e-4, 1e-3, 3e-3} on v2 validation (`--no-test`).
  The selected configuration then trains with seeds 0, 1, 2 for Stage 4, and each seed is scored once on the sealed
  test (ledger).
- **Common training cap** (all learned families, native included): at most 3 passes over the same 5,000 clean
  training samples. References may use all 3 passes; native development so far uses 1. This deliberately favours the
  references.
- **Strongest baseline** (Stage 5): the maximum test AUROC at N* over the information-matched classical detectors
  *and* the two neural families' seed means.
- LRU, S5 and Mamba remain out of scope.
- Prefixes stay per line: N* = 512 per line, a merged prefix of 1,024 at K = 2.

## Protocol amendment: native primary scoring rule = per-step slowdown GLR, 7 October 2026 (curie FAS session)

**Disclosure.** Declared before any FAS evaluation of the rule; only toy evaluations exist. FAS v2 validation results
under the old rule (C1 .630, C2 .663 total) are known.

**Rule.** For native models with a race readout, the primary score at prefix N is `glr_max` (THEORY §440.3).
- For each event type e with at least 5 events among events 1..N−1, take
  G_e = max_s Σ_k [log p_s(x_k | past) − log p_0(x_k | past)] / n_e over the declared log-slowdown grid
  s ∈ {0, .05, .1, .2, .3, .5, .8}.
- p_s scales every own duration by e^s. The score is max_e G_e.
- Implementation: `native_race_readout.glr_prefix_scores`, `experiments/fas/glr_eval.py`.

**Why it is declared a priori.**
- *Theory:* §440 shows the per-step score test, and its GLR extension, to be the locally most powerful and the adaptive
  one-class tests against slowdowns. FAS faults are slowdowns (wear and tear, retry delay). The identity oracle's
  statistic is the identity-aware special case.
- *Toy evidence* (binding toy, synthetic step slowdowns): glr_max against mean NLL AUROC

  | slowdown | 3 processes | 6 processes |
  |---|---|---|
  | 1.2× | .731 vs .636 | .723 vs .644 |
  | 1.5× | .911 vs .871 | .832 vs .850 |

**Unchanged.**
- Native configuration selection stays by validation-clean NLL. The rule is fixed, not selected on faulty data.
- `total` stays reported as a secondary native rule, beside the other declared rules.
- Classical references keep the maximum over their nine detectors on test.
- Neural references keep their NLL rules. The GLR needs per-step duration laws that they do not have.
- The decision rule, thresholds and the sealed test.

## Protocol amendment: primary rule reopened after glr_max failed on validation, 7 October 2026, 23:00 UTC (curie owner)

**Disclosure.** Written after the validation results of the 7 October amendment's rule: glr_max at N* scored .578 (C6)
and .466 (C1), below each model's own total rule (.664, .621) and type rule (.682, .675). No test scoring has occurred;
the sealed ledger is untouched. The earlier amendment and its toy evidence stay in the record.

**Diagnosis.** glr_max scores each event type's own-duration slowdown. "Own durations" require attributing each event to
its item; the native models' binding purity is .30 (C1 diagnostics), so the per-type statistics mix items and the
test loses its power. The rule is locally most powerful only under correct attribution (THEORY §440); FAS v2 is the
case where attribution is the hard part.

**Rule from here.** The native primary rule is chosen on validation from the declared rules {total, type, glr_max}, by
AUROC at N* for the selected native configuration. Each (configuration, rule) pair counts against the native tuning
budget, so choosing a rule is charged as selection, not given free. The reference side is unchanged: its strongest
baseline is the maximum test AUROC over the information-matched classical detectors and the two neural families' seed
means.

**Correction to the amendment above, 7 October 2026, 23:20 UTC (curie owner, disclosed).** Charging every
(configuration, rule) pair against the 8-configuration native budget would exhaust it after fewer than three
configurations. Instead, no rule is selected on validation: **the native primary rule reverts to `total`**, the rule
declared on 5 October before any v2 data existed. `type` and `glr_max` remain reported secondary rules. Native
configurations used so far on v2 validation: C1, C2, C6 (previous owner), C7 and C8 (round 3, running); C9 queued =
6 of 8.
