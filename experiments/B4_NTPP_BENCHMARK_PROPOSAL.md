# B4 (proposed) — the neural TPP benchmark of Bosser & Ben Taieb, with the frozen unified model

**Admitted by the founder on 9 October 2026** (PRODUCT_ORDERS row B4). Pre-registered protocol below, fixed before any
B4 fit.

## Why this battle

B1 is complete: one configuration of the race-of-clocks model wins all five EasyTPP datasets (pre-registered, 24 of 25
seeds ahead). The strongest next claim is **transfer of that frozen configuration to new public datasets it was never
developed on**: no tuning, only the two data rules (window clocks for multi-component gap distributions; the recording
cell from the timestamp resolution). Bosser & Ben Taieb maintain the largest unified neural-TPP benchmark (TMLR 2023,
follow-ups 2023 and 2025), with public preprocessed data, five fixed random splits and published per-dataset tables over
LNM, FNN, RMTPP, THP, SAHP, STHP and their "+"/"++" variants.

## Sources

- T. Bosser, S. Ben Taieb, *On the Predictive Accuracy of Neural Temporal Point Process Models for Continuous-time Event
  Data*, TMLR 2023 ([arXiv 2306.17066](https://arxiv.org/abs/2306.17066)); code and data
  [github.com/tanguybosser/ntpp-tmlr2023](https://github.com/tanguybosser/ntpp-tmlr2023) (MIT); data and splits from the
  Dropbox link in its README.
- *Revisiting the mark conditional independence assumption* (ESANN 2023).
- *Preventing Conflicting Gradients in Neural Marked Temporal Point Processes* ([arXiv 2412.08590](https://arxiv.org/abs/2412.08590), 2025), Table 1.

## Protocol (from the reference code, commit 54c15fd)

- Five fixed random 60/20/20 splits per dataset (`processed/data/<dataset>/split_{0..4}/{train,val,test}.json`; each
  record a list of `{time, labels}`), `window: null` in `args.json`.
- With no window, the observation window is [0, t_last] (`tpp/utils/events.py`: `window_end = times_final`). Hence per
  sequence: **L_T = −Σ_i log f(τ_i)** over all events, the first measured from 0 with an empty history, and no survival
  term after the last event; **L_M = −Σ_i log p(k_i | τ_i, history)**. Reported: mean over test sequences, mean
  (standard error) over the five splits. Lower is better for both.
- Our adapter must therefore score the first event from t = 0 and aggregate per sequence; a numerical contract will
  check the adapter on hand-computed sequences.

## Reference numbers (2025, Table 1; mean over 5 splits; per sequence, lower is better)

| Dataset | Best L_T | Best L_M | Best published total L_T + L_M (same model) |
|---|---|---|---|
| LastFM | −1334.7 (LNM++) | 637.2 (LNM++) | **−697.5** (LNM++) |
| MOOC | −310.6 (LNM+) | 70.9 (THP++) | **−233.8** (LNM++) |
| Github | −381.2 (LNM++) | 109.5 (FNN++) | **−269.7** (LNM++) |
| Stack Overflow | −91.1 (LNM+) | 103.0 (THP++) | **12.1** (LNM+) |

To add before admission: Wikipedia, MIMIC2 and Retweets from the TMLR 2023 Appendix B tables, and the ESANN 2023 numbers;
the 2025 "Reddit" column appears to be a marked Reddit set not contained in the downloaded archive (which holds only the
unmarked `reddit_askscience_comments` and `reddit_politics_submissions`), so it is excluded unless its data is found.

## Data in hand (`data/ntpp_bosser/extracted/processed/data/`, JSON only)

github_filtered, lastfm_filtered, mimic2_filtered, mooc_filtered, retweets_filtered, stack_overflow_filtered,
wikipedia_filtered (marked), plus reddit_askscience_comments, reddit_politics_submissions (unmarked). MOOC split 0:
4,228 / 1,409 / 1,410 sequences, 50 marks.

## Proposed pre-registered rule

The frozen v19 unified configuration and its two data rules, one seed per split (the references report the mean over
the five splits), selection on validation, test once per split. A win requires the mean
total L_T + L_M below the best published total on that dataset; components reported beside it, and per-component wins
stated only where they hold. Compute (parameters, MACs per event) reported.

## Engineering before the first fit

1. JSON loader for the five splits (allow-list parsing, no pickle).
2. Per-sequence scoring with the first event from t = 0 (empty history) and no terminal survival term; contract against
   a hand-computed sequence and against the reference code's formula.
3. Recording-cell detection per dataset (timestamp resolution) and the component rule for windows, applied as in B1.


## Reference bars (fixed 9 Oct 2026, before any fit)

Bar = best published L_T + best published L_M on the dataset, each the minimum over every model in the TMLR 2023
per-dataset tables (Appendix B) and the 2025 Table 1, even when the two come from different models: beating it beats
every published model. Per sequence, mean over the five splits, lower is better.

| Dataset | Best L_T (model, paper) | Best L_M (model, paper) | Bar |
|---|---|---|---|
| LastFM | −1363.78 (GRU-LNM-CONCAT, 2023) | 514.13 (Hawkes, 2023) | **−849.65** |
| MOOC | −310.6 (LNM+, 2025) | 70.9 (THP++, 2025) | **−239.7** |
| Github | −382.4 (GRU-RMTPP-LCONCAT, 2023) | 109.5 (FNN++, 2025) | **−272.9** |
| Stack Overflow | −91.1 (LNM+, 2025) | 103.0 (THP++, 2025) | **11.9** |
| Wikipedia | −267.41 (GRU-RMTPP-LCONCAT, 2023) | 144.79 (Hawkes, 2023) | **−122.62** |
| MIMIC2 | 0.13 (GRU-LNM-TO, 2023) | 2.29 (GRU-RMTPP-LCONCAT, 2023) | **2.42** |
| Retweets | −621.33 (GRU-LNM-CONCAT, 2023) | 82.63 (GRU-RMTPP-LCONCAT, 2023) | **−538.70** |

If a stronger published number on the same data and splits is found later (ESANN 2023, the authors' thesis), the bar
moves to it and the verdict is reported against the stronger number.

## Pre-registered B4 protocol (fixed 9 Oct 2026, before any B4 fit)

- Driver `experiments/tpp/race_tpp_b4.py` with the **frozen unified configuration** of B1 (`race_tpp_v19.py` unchanged:
  2 layers, d 32, 16 memory modes, 2 exponential + 8 delayed clocks, 16 state modes, dropout 0.3, lr 3e−3, batch 32,
  patience 30, seed 0) and its two data rules computed on each split's TRAIN data: windows only for ≥ 2 gap components
  (MOOC, MIMIC2, Retweets); recording cell by the divisor rule (all but Stack Overflow). Adapter contract
  `experiments/tpp/check_b4_adapter.py` PASS.
- Five official splits per dataset, one run per split; checkpoint selected on validation mean per-sequence L_T + L_M;
  TEST scored once per split (`--score-test`).
- Report per dataset: mean (standard error over splits) of L_T, L_M and L_T + L_M, every split's values, parameters and
  per-event compute. **Win on a dataset**: mean L_T + L_M below the bar. Component wins stated only where they hold.
  All seven datasets are reported whatever the outcome. No configuration change between datasets or after any TEST score.

## Verdicts (pre-registered frozen configuration; five official splits, TEST scored once per split)

Per sequence, mean (standard error over splits), lower is better. Bar = best published L_T + best published L_M.

| Dataset | L_T vs best published | L_M vs best published | Total vs bar | Verdict |
|---|---|---|---|---|
| **Wikipedia** | −268.90 (40.82) vs −267.41 | **28.49 (2.82) vs 144.79** | **−240.42 (43.31) vs −122.62** | **WIN**: 4 of 5 splits below the bar (totals −91.92 / −271.68 / −300.05 / −201.15 / −337.29); also below the best single published model (−2.67). Split 0 is a numerical failure (NaN parameters from epoch 6; checkpoint from epoch 5); guarded rerun queued |
| Stack Overflow | **−91.60 (1.56) vs −91.1** | 104.31 (0.72) vs 103.0 | 12.71 (0.87) vs 11.9 | LOSS; time component ahead |
| MOOC | −298.57 (4.08) vs −310.6 | 71.73 (1.27) vs 70.9 | −226.85 (2.83) vs −239.7 | LOSS |
| Github | −357.68 (67.38) vs −382.4 | 159.16 (23.25) vs 109.5 | −198.52 (55.15) vs −272.9 | **Numerical failure, no verdict**: splits 1–4 reached NaN parameters at epochs 3–8, so their checkpoints come from epochs 2–3; guarded reruns queued. Split 0 (trained normally, epoch 36): −351.27 vs −272.9 bar |
| MIMIC2 | 2.99 (0.10) vs 0.13 | 4.02 (0.16) vs 2.29 | 7.01 (0.25) vs 2.42 | LOSS |
| Retweets | 4 of 5 splits | | mean total −516.06 vs −538.70 | pending split 2 |
| LastFM | 2 of 5 valid (0, 2); split 4 original ended NaN | | split totals −720.10 / **−894.02**; split 4 original −875.34 (checkpoint epoch 35, NaN from 39) vs −849.65 | splits 1, 3 (memory retries) and 4 (guarded rerun `b4g_lastfm_s4`, decides the split) pending |

**Why Wikipedia is won, and what it says about the bar.** Wikipedia's marks are pages. In TRAIN, 91–93% of a user's
consecutive edits repeat the previous page. In TEST, 10–27% of events (by split) carry a page never seen in that split's
TRAIN data. A model that learns one embedding per mark cannot copy a page it never trained on. GRU-LNM-CONCAT scores
L_M 259.12 per sequence: log 50 = 3.91 nats per event over ≈ 66 events, i.e. uniform over the 50 marks. Hawkes scores
144.79. Our
addressed mark memory has one slot per mark, written whenever that mark occurs and read by that mark's own score. It
copies any page, seen in training or not: L_M 28.49. A repeat-last-mark rule fitted on TRAIN (repeat probability, otherwise
uniform) scores 31.2–41.1 (mean 37.4) on the same TEST splits. That also lies far below the published L_M, so the
published mark scores miss the dominant repeat structure of this dataset. Our model is 8.9 nats/sequence better than that
rule. The time component is level with the best published value (−268.90 vs −267.41; SE 40.8). The win is a mark-memory
win against the published record. Next test (validation only): L_M split by marks seen and unseen in TRAIN, and against
the repeat rule per split.

**Scoring verification of the Wikipedia win (9 Oct 22:40, AWS; saved checkpoints, VALIDATION only, TEST untouched).** On splits 0 and 4: (1) reloading the selected checkpoint reproduces the recorded validation L_T/L_M to every printed digit (−157.9036/25.7534; −209.5313/21.7601); (2) every event is scored, including the first of each sequence; (3) causality: replacing all later marks and times leaves every earlier event's time and mark log-likelihood unchanged (max difference exactly 0); (4) normalisation: summing the mark probability over all 50 marks at every position of six sequences gives 1 within |log Σ| ≤ 7.2e−4 (≤ 0.04 nats per sequence); (5) units match the published table: a uniform guess over 50 marks on the five TEST splits averages 260.0 nats per sequence against published GRU-LNM-CONCAT 259.12. Our 28.49 is 0.44 nats per event: 90–93% of consecutive Wikipedia events repeat the previous page (0.13–0.17 nats each in our model), and the remainder cost 2.9–4.0 nats, below uniform's 3.91 (log 50). A training-free per-sequence counter reaches 19.5 on split-0 validation, so the low mark loss comes from the data's repeat structure, which the published models fail to exploit.

**Numerical failure (protocol error, found 9 Oct 22:45 UTC).** One batch with a non-finite loss or gradient makes
`clip_grad_norm_` scale every gradient by NaN; AdamW then writes NaN into every parameter and all later epochs are NaN.
Selection keeps the last finite checkpoint. This happened on Github splits 1–4 (epochs 3–8) and Wikipedia split 0
(epoch 6); every other completed split trained without it. The original numbers stay in the results log. The rule for
reruns depends only on training, not on any score: every split whose run ended with non-finite parameters is retrained
with `race_tpp_b4g.py`. That driver skips a non-finite update and logs it, and is otherwise bitwise identical when all
updates are finite (contract job `b4g_contract`). Retweets and LastFM runs still in progress fall under the same rule.

**Error analysis against simple history baselines (validation, fitted on TRAIN).** Unigram, repeat-last-mark and
first-order Markov (bigram) mark models give context for L_M:
- **MOOC and Stack Overflow:** our validation L_M (≈ 71 and ≈ 104) is far below the bigram (≈ 102 and ≈ 124).
- **Github:** split 0 (282 vs 303) is below the bigram. The NaN-failed splits score above it.
- **MIMIC2:** about 3 events per sequence. We sit at the bigram level (3.41–4.47 vs 3.62–4.38) while the best published
  L_M is 2.29. Its splits select checkpoints at epochs 3–11, so a 103–121-sequence training set is overfitting the
  frozen configuration. MIMIC2 is a development target for small-data regularization of the mark path; its time values
  are on a coarse recording cell (0.033), where the gridded-metric analysis below applies to L_T. MIMIC2's time
values are on a coarse recording cell (0.033), where the gridded-metric analysis below applies to L_T.

## Protocol note on zero gaps (9 Oct 2026, recorded while runs train; verdict rule unchanged)

In the reference code an exactly-zero inter-event time is replaced by ε = 1e−20 before the log-normal mixture density is
evaluated (`tpp/models/decoders/log_normal_mixture.py`: `delta_t + (delta_t == 0) * epsilon`, `epsilon = 1e-20`). A mixture
component near log τ ≈ −46 with a small scale then earns tens of nats per zero-gap event: the recording-resolution
pitfall of B1 (a likelihood gain from the timestamp grid, not from the process). Zero gaps occur in **Retweets (4.0% of
gaps, ≈ 4–5 per sequence)** and **Github (1.3%)**; LastFM, MOOC, Stack Overflow, Wikipedia and MIMIC2 have none. Our
model holds every hazard at its one-cell value below the recording cell and cannot place such a spike.

Whether the published models exploit it cannot be verified without their trained checkpoints (we do not train external
architectures). The pre-registered verdicts stand as defined; on Retweets and Github the result is reported with this
note, and with our L_T on the events with positive gaps beside it for interpretation. Early validation fits fit this
pattern: MOOC (no zero gaps) is within 4 nats of the best published L_T, Retweets 35 nats behind.

## Results log

| Date | Run | Validation (L_T / L_M / total) | TEST (L_T / L_M / total) | Notes |
|---|---|---|---|---|
| 9 Oct | MOOC split 0 | −307.25 / 73.37 / −233.88 (best epoch 134 of 165) | **−310.43 / 75.05 / −235.38** | 57,238 parameters; cell 3.89e−6, 4 windows. Below LNM++'s total (−233.8, best single published model, 5-split mean) but above the composite bar (−239.7); mark NLL is the weaker component. Verdict on the five-split mean |
| 9 Oct | MOOC split 1 | −286.95 / 70.30 / −216.65 (best epoch 171; stopped at the 200-epoch cap) | **−290.73 / 68.66 / −222.07** | mark NLL below the best published L_M (70.9); splits differ strongly (published L_T standard error 3.9 over splits), so only the five-split mean decides. The epoch cap is part of the frozen configuration |
| 9 Oct | MOOC split 2 | −300.30 / 72.49 / −227.81 (best epoch 134) | **−298.38 / 71.72 / −226.67** | three splits: mean total ≈ −228.0, behind LNM++ (−233.8) and the composite bar (−239.7) so far |
| 9 Oct | MOOC splits 3, 4 | — | −288.82 / 69.15 / −219.67; −304.51 / 74.06 / −230.45 | |
| 9 Oct | **MOOC VERDICT (pre-registered)** | five splits, TEST | L_T **−298.57 (se 4.08)**, L_M **71.73 (1.27)**, total **−226.85 (2.83)** | **LOSS**: total above the bar −239.7 and above the best single published model (LNM++ −233.8; secondary). L_M is within 0.8 of the best published (70.9) and below LNM++ (73.8); **L_T is the larger gap** (−298.6 vs −310.6, ≈ 0.25 nats/event). **Revised diagnosis:** the split-0 analysis ("time level, marks trail") rested on our best time split; over five splits time is the main gap. MOOC's timestamps lie on a one-second grid (every gap a multiple of the cell), so published L_T may include grid exploitation that our resolution principle forbids; this cannot be verified without the authors' checkpoints and is stated as a caveat, not a correction of the verdict |
| 9 Oct | Retweets splits 0, 3 | — | −596.69 / 83.12 / −513.57; −610.25 / 85.19 / −525.07 | behind the bar (−538.7) and the best single published model (−536.2); the zero-gap note applies (4.0% zero gaps); positive-gap L_T reported at the verdict |
| 9 Oct | LastFM split 2 | best epoch 39 of 70, all epochs finite | **−1554.19 / 660.16 / −894.02** | below the bar −849.65 and the best single published model (−697.5). Time component far ahead of the best published L_T (−1363.78), on a dataset with no gaps of 1–10 recording cells (no lattice scoring); marks behind Hawkes's 514.13. Validation L_M oscillated 788–849 after epoch 39 (the stationary-habit gap; v20 stats arms queued) |
| 10 Oct | LastFM split 4 (original, unguarded) | best epoch 35; NaN parameters from epoch 39; stopped at 66 | −1530.06 / 654.72 / −875.34 | below the bar −849.65 from an epoch-35 checkpoint (before the failure); under the 22:45 rule the guarded rerun `b4g_lastfm_s4` decides split 4 and this run stays as the record |
| 9 Oct | **Stack Overflow VERDICT (pre-registered frozen configuration)** | five fixed splits, TEST; grid-free dataset | L_T **−91.598 (SE 1.563)**, L_M **104.312 (0.719)**, total **12.714 (0.874)**; split totals **11.014 / 15.101 / 14.535 / 11.123 / 11.800** | **LOSS**: total above the bar 11.9 and best single published model 12.1. **Time component win:** −91.598 vs best published −91.1. Mark NLL trails 103.0 by 1.312; the time gain of 0.498 offsets part of it. 29,762 parameters, 22 marks; identical frozen configuration and source hashes across all splits. This diagnoses the frozen transfer result, not the developed mark-path attempt; v20 development stays on validation |

## Secondary comparison (reported beside the pre-registered verdict, never replacing it)

Best total of a single published model (its own L_T + L_M), mean over five splits, lower is better. 2025: Table 1 of
arXiv 2412.08590 (every model reports both). 2023: only the nine models present in both the NLL-T and NLL-M tables of
Appendix B can be paired (GRU-EC-TEMWL, GRU-FNN-LCONCAT, GRU-LNM-CONCAT, GRU-MLP/MC-LCONCAT, GRU-RMTPP-LCONCAT,
GRU-SA/MC-LE, Hawkes, NH, Poisson); those rows are indicative and are checked against the PDF before use.

| Dataset | Best single-model total | Model (paper) |
|---|---|---|
| LastFM | −697.5 | LNM++ (2025) |
| MOOC | −233.8 | LNM++ (2025) |
| Github | −269.7 | LNM++ (2025) |
| Stack Overflow | 12.1 | LNM+ (2025) |
| Wikipedia | −2.67 (verified: L_T −261.79 + L_M 259.12) | GRU-LNM-CONCAT (2023) |
| MIMIC2 | 3.1 | GRU-LNM-CONCAT (2023) |
| Retweets | −536.17 | GRU-LNM-CONCAT (2023) |

## Development track (outside the frozen protocol): MOOC diagnosis and v20

Diagnosis on the split-0 checkpoint (validation, 72,222 events): time is level with the best published model; marks trail.
Mark NLL per event is 1.85 after a session break (previous gap ≥ 1e−2) vs 1.06–1.50 elsewhere, and 1.60 at positions ≥ 100
vs 1.36 before position 20 in the same long sequences. Count models of the previous 1–3 marks score L_M 91–102 (ours 73),
so the model is not missing basic transitions. Interpretation: our memory forgets with elapsed time (the design that wins
EasyTPP) and has nowhere to keep time-invariant user habits; one memory serves both tasks, whose gradient conflict the
reference authors document (separating the paths moved LNM on MOOC from 86.6 to 73.8). **v20**
(`race_tpp_v20.py`, flags off = v19 exactly): `--mark-mem N` adds event-indexed memory read only by the mark
distributions, `--mark-stats` adds per-sequence mark counts and frequencies. Contracts (`check_v20.py`) PASS: flags off
equal v19 on the trained checkpoint; flags on equal v19 at load (zero-initialized term); time gradient into the mark
pathway exactly 0; mark distributions normalize. Development arms on MOOC split 0 (validation only) queued; a configuration
chosen there is pre-registered separately as a developed attempt on all seven datasets, reported beside the frozen result.

**Stationary mark habits (9 Oct 22:20, split-0 VALIDATION, training-free, AWS).** A per-sequence Dirichlet mark counter (running counts of each mark in the sequence, smoothed toward the TRAIN unigram with strength a = 2; no fitting) against the frozen model's selected-epoch L_M per sequence: **LastFM 694.7 vs ours 711.0**, **Wikipedia 19.5 (a = 0.5) vs 25.8**; ours leads on Github (282.3 vs 318.3), MOOC (73.4 vs 200.5) and Stack Overflow (107.5 vs 130.0). On LastFM the frozen model's validation L_M plateaus from epoch 8 and oscillates (713–779), while Hawkes (per-mark self-excitation) holds the best published TEST L_M (514.1 vs ours 567.9 on split 0). Diagnosis: the elapsed-time-decaying memory has no place for stationary per-sequence habits, the same gap as MOOC's post-break marks. v20 `--mark-stats` feeds log1p(count) and frequency into the mark logits and can represent the counter exactly (the normaliser cancels in the softmax). LastFM is grid-free in effect (no gaps of 1–10 recording cells; MOOC 32%, Retweets 29%), so its time gap is modelling error, not lattice scoring. Admitted: `b4dev_lastfm_s0_stats` and `b4dev_lastfm_s0_mem1_stats` (slot 2, validation only); pass = validation L_M below the counter (694.7).

**Github is a numerical-stability failure; MIMIC2 overfits early (9 Oct 22:30, AWS, from the frozen result files).** Github splits 1–4 turned every weight NaN at epochs 3–6 (split 1: finite val total −56.6 at epoch 3, NaN from epoch 4) and kept training on NaN for 26–30 epochs until patience; selection fell back to epochs 2–3. The driver has no non-finite guard, so one non-finite step makes the parameters and AdamW moments NaN permanently. Even these epoch-2–3 models reach test L_M 96.1 (split 2) against the published 109.5. The protocol repair (above) retrains every split that ended non-finite with the guarded `race_tpp_b4g.py` and scores TEST once; the validation-only `race_tpp_b4v22.py` guard arms admitted for this diagnosis were withdrawn unstarted as redundant (22:55 UTC). MIMIC2 (≈3 events per sequence) selects epochs 3–11 of 34–42; a training-free per-sequence counter mixed with the TRAIN bigram scores split-0 validation L_M 3.5 against ours 3.67, so regularisation of the mark path is the MIMIC2 development target. Training-free counter+bigram validation L_M on the other datasets: Github 291.2 (ours 282.3), MOOC 115.9 (73.4), Stack Overflow 123.2 (107.5), LastFM 683.7 (711.0), Wikipedia 19.5 (25.8).

## Understanding the losses: the time metric on gridded data (9 Oct, validation, evaluation only)

A **context-free** log-normal mixture (32 components, fitted to TRAIN log-gaps, no history, no training of any external
architecture: a statistical estimate) was scored with the protocol's metric, as recorded and with every gap dequantized
inside its recording cell (zero gaps scored at the reference code's ε = 1e−20 as recorded). Time NLL per sequence, lower is
better:

| Split-0 validation | Ours raw | Ours dequantized | Mixture raw | Mixture dequantized | Best published L_T (test) |
|---|---|---|---|---|---|
| MOOC (1-s grid, 33% of gaps exactly on 1–10 cells) | −307.25 | −307.11 | **−342.15** | +571.84 | −310.6 |
| Retweets (1-s grid, 3.9% zero gaps; 1,500 sequences) | −584.60 | −584.18 | **−791.55** | +3,096.32 | −621.33 |

By gap size on MOOC, the context-free mixture beats our model only on gaps of 1–10 recording cells (by 49 nats per
sequence: spikes on lattice points) and loses to it on every longer range (by 14 nats per sequence: the history model
works). **Reading:** on gridded datasets the protocol's raw time NLL rewards density spikes on the recording lattice by
hundreds of nats per sequence; a model that ignores history entirely beats every published model on MOOC and Retweets
time NLL, and its advantage collapses by 900–3,900 nats when the rounding is removed. Our model moves by ≤ 0.4. Whether the
published models exploit the lattice cannot be established without their checkpoints; what is established is that the
metric is dominated by it on these datasets. The pre-registered verdicts stand as defined (MOOC: loss); they are reported
with this analysis beside them. **Valid comparisons in B4:** Stack Overflow, the one grid-free dataset (no recording cell
detected), and the mark component L_M on every dataset (unaffected by the time grid).
