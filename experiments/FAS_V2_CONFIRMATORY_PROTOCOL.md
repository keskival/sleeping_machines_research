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
