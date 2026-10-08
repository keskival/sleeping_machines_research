# B4 (proposed) — the neural TPP benchmark of Bosser & Ben Taieb, with the frozen unified model

8 October 2026 · **Proposal awaiting the founder's admission** (PRODUCT_ORDERS lists B1–B3, G, R1; a new battle needs
the user's go-ahead). No training has run. Data downloaded and studied only.

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
