# IDF-03 — Event-native classification of irregular multichannel series with statistic-valued addressed channel memories

CONFIDENTIAL · Invention disclosure for counsel · Status: **IMTS embodiment unpublished** (6 Oct 18:54 UTC `efb1c286`;
statistics 7 Oct 00:12 `998d6d7f`; typed semantics 6 Oct 07:23 `20de60cd`). **The generic statistic-valued race memory
is public** (theory note 59, first public 2 Oct 2026) · Jurisdictions: **EP for the specific embodiment; US for both,
the generic form before 2 Oct 2027**

## 1. Technical field and problem

Monitoring devices and analytics that classify records of many channels measured at irregular, channel-specific times
(bedside physiological monitors and laboratory systems, wearable inertial sensors, industrial sensor networks). Leading
methods re-grid the series and apply attention/graph mixing (Raindrop, Warpformer, ViTST, GraFITi, MTM), with models of
0.4M–150M parameters; whether and when a channel is measured is informative but is usually handled by imputation or masks.

## 2. Solution

Each time step with any measurement is an event. For each channel c an **addressed slot** is maintained that is written
only at steps where c is measured, decays with elapsed time between writes, and carries **sufficient statistics** of the
channel's history: count, running mean, minimum, maximum, first value, last value, trend (last − first), time since the
last measurement (staleness, growing through silence), and optionally variance and mean absolute change. Values enter
through **typed comparisons** (per-channel learned soft thresholds σ((x − θ_cj)/s_cj)) and channel-specific projections
before neural mixing. A persistent temporal memory (complex-diagonal, decaying and rotating with elapsed time; the same
layer as IDF-01's event model) integrates the event messages. The readout combines final and mean memory states, static
covariates, slot values, staleness and a projection of the normalized statistics.

## 3. Technical effects (measured)

- P19 sepsis prediction (PhysioNet 2019, five official splits): TEST AUROC 0.916 ± 0.022, AUPRC 0.639 ± 0.039 vs best
  published 0.903 / 0.583 (MTM), with 62,681 parameters.
- Diagnostic: before the statistic-valued slots the temporal model extracted no more than gradient-boosted trees on
  per-channel summaries (0.866 vs 0.867 validation AUROC); with them it surpassed both (0.872).
- PAM wearable activity recognition (development): validation accuracy 0.981, F1 0.983 with 46,316 parameters (MTM TEST
  0.975 / 0.976); five-split protocol running.
- Compute and memory: slot updates only for measured channels (sparse writes); constant-size per-channel state
  independent of record length.
Evidence: `experiments/B2_IRREGULAR_TS.md`, `experiments/results/irts/b2_final_p19_v3_*`.

## 4. Prior art to distinguish

GRU-D (decay toward means with masks and time since last observation), mTAND, Raindrop, Warpformer, ViTST, GraFITi, MTM;
summary-feature pipelines with gradient-boosted trees; our public note 59 (statistic-valued race memory, generic). The
candidate novelty for EP is the specific combination: per-channel addressed slots written only at measurement and
holding the listed running statistics, typed threshold comparisons before mixing, and a shared continuous-time race/
temporal memory core, in a monitoring device or system.

## 5. Draft claim concepts (for counsel)

1. A computer-implemented method of classifying a record from a plurality of sensors or measurement channels sampled at
   irregular, channel-specific times, comprising: processing each time step containing at least one measurement as an
   event; for each measured channel, updating a channel-addressed memory slot only at steps where that channel is
   measured, the slot holding a decaying learned value and running statistics comprising at least a count, a running
   mean, an extremum and a time since the last measurement; updating a continuous-time memory with the event message;
   and producing a classification or alert from the memory and the slots.
2. …wherein each value is first mapped through learned per-channel soft thresholds before projection and summation.
3. …wherein the statistics further comprise first and last values, trend, variance and mean absolute change.
4. …wherein the continuous-time memory is shared with an event-prediction model according to IDF-01.
5. A monitoring device (bedside, wearable, industrial) implementing the method and issuing an alert.
6. US (generic, before 2 Oct 2027): a race memory whose values are sufficient statistics, enabling closed-form
   counterfactual write credit (note 59).
