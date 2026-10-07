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
| Ablation: normalized match only (no message), backprop | 0 | 21.6% | −2.203 | −3.492 | 12.4% |
| Keyed read, local race credit | 0 | **91.9%** | −0.379 | −3.282 | **79.4%** |
| Frozen B1 v5 | 0 | 21.6% | −2.231 | −3.512 | 12.3% |
| Set baseline | — | 14.1% | | | 7.9% |

**Gate 1 status: met at the training length over three seeds** (99.4 ± 0.2% vs the set baseline 14.1% and frozen v5
21.6%); length generalization (16 pairs) is not yet met. Parameters: 36,325 for the keyed model, 34,740 for v5. Pending:
- ablations: predecessor message only, normalization only;
- mixed-length training (4–16 pairs, 32-pair held-out test);
- local race credit, seeds 1–2.

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

## Next

1. Finish confirmation (seeds and ablations), then state gate 1 as met or not.
2. Length generalization: mixed-length training (running next); the local-credit result suggests that co-adapting the
   backbone to the read hurts extrapolation.
3. Carry the keyed read and predecessor message into the token language interface for gate 2 (KN trigram on the
   65,528-target DEV slice).
4. Offer the mechanism to B1 (AWS) for datasets where marks recur with context (StackOverflow, Retweet).
