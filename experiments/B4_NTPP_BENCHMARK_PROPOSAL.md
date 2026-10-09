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
