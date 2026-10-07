# Generality track G — one family across data types (opened 7 October 2026, user-directed)

Benchmark wins are a goal in their own right (battles B1–B3, PRODUCT_ORDERS.md). Track G asks a different question: is
this one family, with one core, competitive across structurally different data — and does experience in one form
transfer to another? Each experiment states what it settles. Losses are reported beside wins.

## What is already established (completed evidence)

| Data type | Public benchmark or protocol | Result | Core configuration |
|---|---|---|---|
| Generative event streams (time + type) | EasyTPP, official splits, 5 seeds | Wins on Taxi, Taobao, StackOverflow vs S2P2/IFTPP; Amazon protocol running; Retweet in development | temporal memory d32 × 16 modes × 2 layers (StackOverflow 3), addressed mark memory, race-of-clocks head |
| Irregular clinical records (classification) | P19 sepsis, five official splits | Win: AUROC 0.916 ± 0.022, AUPRC 0.639 ± 0.039 vs MTM 0.903 / 0.583 | **same temporal memory layer code and size** (d32 × 16 × 2), addressed channel memory with sufficient statistics, typed comparisons |
| Anonymous interleaved process logs | FAS v1, 3 seeds | 0.592 vs 0.559 AUROC against six generic controls | temporal/sparse native race model |
| Character language | text8 10M | Beats tuned Transformers at ≤ equal compute; LSTM leads; 90M behind | integrated temporal/sparse core with route credit |
| Temporal reasoning (synthetic) | event-order, timing-only, retrieval | 99.7% vs Transformers 33–41%; timing-only 95.3% vs order-only ceiling 50% | race/delay members |
| Mixed-type tables | synthetic witness; banknote | 100% synthetic; trees lead on banknote | typed comparisons into the temporal core |

Shared code, not only shared ideas: `experiments/irts/race_irts*.py` imports `TemporalMemoryLayer` from
`experiments/tpp/race_tpp_v8.py`. Positioning: S2P2 (the EasyTPP leader) is demonstrated on one task type (event
likelihood). Its parent family, deep state-space models, is general on dense sequences and language; published SSM
results do not, to our knowledge, show one model winning event likelihood and sparse clinical classification together.

## Experiments, in priority order

| Id | Experiment | Design | Pass condition | Settles |
|---|---|---|---|---|
| **G1** | **Self-supervised event pretraining → clinical classification** | Pretrain the B2 encoder on UNLABELED P19/P12 TRAIN records as a marked point process (mark = channel measured, race-of-clocks head predicts the next measurement's time and channel); fine-tune on P12 mortality with 10%, 30%, 100% of labels; compare with training from scratch at equal total compute | Pretrained beats scratch at ≤ 30% labels on validation AUROC, beyond split noise | The same model both generates event streams and classifies from them; generative experience transfers to a discriminative task |
| **G2** | **One model, five event datasets** | Train a single race-of-clocks model on all five EasyTPP datasets at once (shared temporal memory and clock head, dataset-specific mark embeddings); compare per-dataset DEV LL with the per-dataset models at equal total parameters and compute | Joint model within 0.01 nats of per-dataset models on ≥ 3 datasets, or better on any | Shared temporal dynamics across domains (shopping, mobility, Q&A, social) |
| **G3** | **Third public data type** | PAM wearable activity (B2) | Beat MTM 97.5% accuracy on official splits | A dense multichannel sensor domain with the same family |
| **G4** | **Cross-domain transfer, event → event** | Pretrain on Retweet/Taobao, fine-tune on Taxi with 10% of its training sequences vs scratch | Pretrained beats scratch at 10% data | Transferable temporal skill across unrelated event domains |
| **G5** | **Shared core across tasks (stretch)** | One core trained jointly on EasyTPP likelihood and P19 classification (task heads only differ) | Each task within its development margin of the single-task model | One model, two task types, joint training |

Compute convention: equal total fitting compute (pretraining included) for every transfer comparison. Every number comes
from a completed run; pending cells stay pending.

## Order of execution

G3 is already queued (B2 PAM). G1 is next: it reuses the B2 data and the B1 clock head, and it is the strongest single
demonstration of generality an investor can understand (*the same model that predicts the next clinical measurement
learns to predict mortality from fewer labels*). G2 follows when B1's protocols free slots.

## Results log

| Date | Experiment | Setting | Result | Reading |
|---|---|---|---|---|
| 7 Oct | G1 | P12 split 0, 10% labels (959 records), validation | pretrained (10 epochs next-measurement modelling on unlabeled TRAIN) **AUROC 0.838, AUPRC 0.507**; scratch 0.825 / 0.446 | +0.013 AUROC, +0.061 AUPRC from generative event pretraining; one split, one seed — 30%/100% and more seeds/splits next |
