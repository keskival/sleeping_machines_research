# R1 — Associative recall with irregular gaps (language gate 1)

Owner: curie host (from 7 October 2026). Battle R1 in PRODUCT_ORDERS.md: before any language scaling or claim, the family
must (1) solve associative recall/induction with irregular gaps and (2) beat KN trigram on the 65,528-target DEV slice.
This dossier covers gate 1, tested on the model that wins EasyTPP (B1), because that is the family's strongest trained
model.

## Protocol (frozen 7 Oct 14:20 UTC, v4 task)

- **Stream.** Marks 0–31 are keys, 32–63 values. A sequence shows 8 (key, value) pairs with distinct keys and random
  values, then 8 queries drawn with replacement from the shown keys, each followed by its value.
- **Timing.** A value follows its key after a log-normal gap (median 0.2). Pairs are separated by gaps log-uniform on
  [0.05, 50], three decades of irregular timing.
- **Scoring.** The whole stream is scored as a marked temporal point process (exact time + mark log-likelihood). Recall is
  the argmax over values of the mark distribution at the observed time of each query's answer.
  - *Chance:* 1/32.
  - *Set baseline:* a uniform guess among the values shown in the context, about 14%.
  - *Binding claim:* binding is claimed only for recall above the set baseline.
- **Splits.**
  - Training: 20,000 sequences.
  - DEV and TEST: 1,000 sequences each (8,000 recall events).
  - Length extrapolation: 16 pairs.
  - Selection: by DEV log-likelihood. TEST is scored once.
- **Drivers.** `experiments/tpp/recall_tpp_v4.py` (`recall_tpp_v5.py` adds mixed-length training). Each driver records
  its own and race_tpp_v5's sha256. Diagnosis: `experiments/tpp/recall_diagnose.py`. All runs go through run_safe on
  curie, one job at a time.

## Mechanism under test

The B1 model's addressed mark memory has one slot per mark, read by a fixed vector, so it cannot compare stored content
with the current event. The **keyed mark memory** adds:
- **Separate keys:** each key is written only to the occurring mark's slot. It is built from the event's state and the
  **message of the preceding event** (`--prev-msg`), and decays with elapsed time.
- **A query from each event:** it scores every slot inside each clock's mark race, using normalized keys and queries with
  a learned sharpness (`--qk-norm`).

All B1 mechanisms are kept: the race of delayed clocks, exact survival, temporal memory and addressed value slots.
Inference adds K·dk multiply-adds per event.

**Local race credit** (`--keyed 2`): the state entering the keyed memory is detached, so no gradient from the read
reaches the network. The key and query maps learn only from:
- the race error at the read, which credits every losing value by its probability;
- the elapsed-time-decayed trace stored in each slot.

This is a three-factor rule whose factors are all available at the slot and the read.

## Results (TEST)

| Arm | Seed | Recall, 8 pairs | Recall LL | Total LL | Recall, 16 pairs |
|---|---|---|---|---|---|
| Keyed read, backprop | 0 | **99.6%** | −0.012 | **−2.922** | 51.2% |
| Keyed read, backprop | 1 | **99.4%** | −0.020 | −2.924 | 37.6% |
| Keyed read, backprop | 2 | **99.2%** | −0.027 | −2.913 | 45.1% |
| **Keyed read, backprop, 3 seeds** | mean ± sd | **99.4 ± 0.2%** | | **−2.919** | 44.6 ± 6.8% |
| Ablation: predecessor message only (no normalization), backprop | 0 | 97.3% | −0.164 | −2.993 | **90.1%** |
| Message only, backprop | 1 | 99.3% | | −2.927 | 61.6% |
| Message only, backprop | 2 | 95.8% | | −3.013 | 69.7% |
| **Message only, backprop, 3 seeds (recommended design)** | mean ± sd | **97.5 ± 1.7%** | | −2.978 | **73.8 ± 14.7%** |
| Ablation: normalized match only (no message), backprop | 0 | 21.6% | −2.203 | −3.492 | 12.4% |
| Keyed read, normalized local race credit | 0 | **91.9%** | −0.379 | −3.282 | **79.4%** |
| Same | 1 | 83.8% | -0.578 | -3.182 | 23.1% |
| Same | 2 | 94.2% | -0.212 | -3.067 | 23.3% |
| **Normalized local race credit, 3 seeds** | mean ± sd | **90.0 ± 5.5%** | | **−3.177** | **41.9 ± 32.4%** |
| **Message only, local race credit, 3 seeds** | mean ± sd | **77.4 ± 5.4%** (77.3 / 72.2 / 82.9) | | | 57.4 ± 5.2% |
| Frozen B1 v5 | 0 | 21.6% | −2.231 | −3.512 | 12.3% |
| **Message only, mixed-length training (4–16 pairs), backprop** | 0 | **99.2%** | | −3.124 | 97.5% (in range); **91.6% at 32 pairs** (held out; set baseline 5.0%) |
| **Mixed lengths, backprop, 3 seeds** | mean ± sd | **99.0 ± 0.3%** | | **−3.128** | **96.8 ± 0.7% at 16; 90.2 ± 1.2% at held-out 32** |
| Set baseline | — | 14.1% | | | 7.9% |

**Gate 1 status: met at the training length over three seeds** (99.4 ± 0.2% vs the set baseline 14.1% and frozen v5
21.6%). **Length generalization confirmed with mixed-length training:** three seeds reach 90.2 ± 1.2% at 32 pairs, twice the longest training length (set baseline 4.9%). Parameters: 36,325 for the keyed model, 34,740 for v5. The ablations, mixed-length seeds and normalized local-credit seeds are completed; do not rerun their queues.

**Diagnosis** (backprop, seed 0):
- The keyed match alone ranks the correct value first in 99.76% of queries.
- Recall is flat across pair age (99.2–100%) and elapsed time (0.01 to over 100 units).
- The preceding key is decodable from the state at only 20.7%, so the binding travels through the predecessor message.

## Development log

| Date (UTC) | Round | Change | Result | Decision |
|---|---|---|---|---|
| 7 Oct 13:07 | 1 (v1 task) | Keyed read (state only) vs frozen v5 | 38.9% vs 38.6%; 16 pairs 11.1% vs 17.5% | Keyed read adds nothing |
| 7 Oct 14:16 | Diagnosis | `recall_diagnose.py` | Preceding key not decodable (3.9% vs 3.1%); recall rises with query position to 100% | **Task shortcut:** every key queried once allows elimination; the 38% plateau was that shortcut. Running v3 arms stopped at start |
| 7 Oct 14:18 | 2 (v4 task) | Queries with replacement and set baseline; predecessor message and normalized match | 99.6% vs v5 21.6% (seed 0) | Binding learned |
| 7 Oct 15:40 | 2 | Same, local race credit only | 91.9%; 79.4% at 16 pairs | Learning through the race alone works; it extrapolates better than backprop |
| 7 Oct 16:01 | 2 | Backprop seed 1 | 99.4%; 37.6% at 16 pairs | Reproduced; extrapolation varies by seed |
| 7 Oct 16:22 | 2 | Backprop seed 2 | 99.2%; 45.1% at 16 pairs | Gate 1 met at training length (3 seeds) |
| 7 Oct 16:35 | 2 | Ablation: predecessor message only | 97.3%; **90.1%** at 16 pairs | The message carries the binding; the learned sharpness of the normalized match costs length generalization |
| 7 Oct 17:15 | 2 | Ablation: normalized match only | TEST 21.6%, 12.4% at 16 pairs (identical to v5) | The predecessor message is necessary. Design fixed to **message only**; the not-yet-started queues for message+normalization (`curie_r1_v5len_*_20261007T1520Z`, `curie_r1_v4_pn_keyed2_s{1,2}_20261007T1545Z`) are superseded and were never run; replaced by the message-only chain (backprop seeds 1–2, local race credit seeds 0–2, mixed-length 4–16 pairs) |
| 7 Oct 18:02 | 2 | Message only, backprop seeds 1–2 | 3 seeds: 97.5 ± 1.7%; 16 pairs 73.8 ± 14.7% | Recommended design reproduces; generalizes better than message + normalization (44.6 ± 6.8%) but varies by seed |
| 7 Oct 18:42 | 2 | Message only, local race credit seeds 0–2 | 77.4 ± 5.4%; 16 pairs 57.4 ± 5.2% | **Learning through the race alone binds content reproducibly** (≈5.5× the set baseline). Below backprop (97.5%) and normalized local credit (90.0 ± 5.5%, three completed seeds) |
| 7 Oct 19:14 | 3 | Mixed-length training 4–16 pairs (`recall_tpp_v5.py`), message only | 99.2% at 8; 97.5% at 16; **91.6% at 32 pairs** (2× the longest training length) | Length generalization follows from mixed lengths; all three seeds now completed |

| 9 Oct reconciliation | 3 confirmation | Mixed-length seeds 0–2, identical source/configuration | 99.0 ± 0.3% at 8; 96.8 ± 0.7% at 16; **90.2 ± 1.2% at 32** (91.6 / 89.8 / 89.2) | Length generalization confirmed; reuse completed evidence |
| 9 Oct reconciliation | Local confirmation | Normalized local-credit seeds 0–2 | 90.0 ± 5.5% at 8; **41.9 ± 32.4% at 16** (79.4 / 23.1 / 23.3) | Better training-length mean than message-only local credit; length behavior must be part of the next DEV comparison |

## Gate 2 results (65,528-target FineWeb slice, GPT-2 BPE, nats per token, scored once)

| Model (1M training tokens unless stated) | Seed | Score | Selection slice | Parameters |
|---|---|---|---|---|
| **Keyed temporal-memory token model** (`r1_token_keyed_lm.py`, best epoch 2 of 5) | 0 | **6.019** | 6.310 | 3,333,845 |
| Keyed temporal-memory token model | 1 | **5.9996** | 6.308 | 3,333,845 |
| Keyed temporal-memory token model | 2 | **6.0068** | 6.325 | 3,333,845 |
| **Keyed model, 3 seeds** | mean ± sd | **6.009 ± 0.010** | | |
| Same model **without the keyed read** (ablation, best epoch 2 of 5) | 0 | 6.193 | 6.465 | 3,321,425 |
| KN trigram | — | 6.537 | | |
| KN bigram | — | 6.584 | | |
| KN trigram, 4M tokens | — | 6.100 | | |
| **Keyed model, 4M TRAIN tokens** (best epoch 3; run stopped at its 5.8 h limit in epoch 4) | 0 | **5.521** | 5.788 | 3,333,845 |
| Keyed model, 4M TRAIN tokens (same limit) | 1 | **5.499** | 5.785 | 3,333,845 |
| **Keyed model, 4M, 2 seeds** | mean ± sd | **5.510 ± 0.016** | | |

**Gate 2 met over three seeds (6.009 ± 0.010; seed 0 first):** 0.518 nats per token below KN trigram at the same training tokens (KN standard error on this
slice ≈ 0.017), and below KN trigram trained on 4× the tokens. Ablation: the temporal memory alone scores
6.193 (0.344 below KN trigram); **the keyed predecessor read adds 0.175 nats per token**, which takes the model
past KN trigram at 4× the data. Seeds 1–2 (8 Oct): 5.9996 and 6.0068. **Scaling (8–9 Oct, 2 seeds): at 4M TRAIN tokens 5.510 ± 0.016 (5.521, 5.499) vs KN trigram 6.100; the margin grows from 0.528 (1M) to 0.590 (4M)** — the model gains 0.488 nats per token from 4× data, the trigram 0.437. The model overfits after epoch 2 (train 5.54 → 5.06 while selection rises), so
regularization and the 4M setting are the next levers.

## Next

1. **Sparse credit integration (B1/R1 enabler).** Theory note 162 supplies frozen forward utility predictions plus full-support forced-write corrections for the conditional expected-loss route component. The existing recall model's addressed writes are deterministic: adding an optional destination is an explicit diagnostic extension, not a claim that the current model already has a latent write router. `credit/check_native_write_residual.py` preserves the two-layer temporal core, delayed clocks, exact survival, predecessor message and separate key/value writes. It tests original-write parity, prefix causality, native shared-parameter route-gradient means and the separate continuous-path term. No production driver is changed. Queued non-fitting gate: `queue/enabler/curie_r1_native_write_residual_20261009T2140Z/manifest.json`; result pending behind Curie's host lock.
2. After the native contract, add an integrated smoke before admitting a bounded DEV fit. Compare full credit, message-only local credit, normalized local credit and the corrected forward predictor on the same native model/data/horizon. Measure DEV recall at 8/16/32 pairs, per-parameter credit error/variance, complete replay/discovery/optimizer work, CPU time and storage. Forward-supported route credit alone does not restore the detached continuous producer gradients in local mode; preserve or separately derive their learning rather than attributing the 77.4→97.5 gap to a missing latent route.
3. Tokenized language gates are met. Next quality comparison uses a published small Transformer with matching tokenizer, data and scoring; no new external reference training. Own-model development prioritizes regularization and measured interface cost before scaling.
4. The predecessor-message mechanism is available for B1 datasets with contextual mark recurrence. AWS owns admission and execution there.
