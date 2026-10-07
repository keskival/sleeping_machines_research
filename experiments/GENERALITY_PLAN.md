# Generality track G — one family across data types (opened 7 October 2026, user-directed)

Benchmark wins are a goal in their own right (battles B1–B3, PRODUCT_ORDERS.md). Track G asks a different question: is
this one family, with one core, competitive across structurally different data — and does experience in one form
transfer to another? Each experiment states what it settles. Losses are reported beside wins.

## What is already established (completed evidence)

| Data type | Public benchmark or protocol | Result | Core configuration |
|---|---|---|---|
| Generative event streams (time + type) | EasyTPP, official splits, 5 seeds | Wins on all five (Taxi, Taobao, StackOverflow, Retweet, Amazon) vs S2P2/IFTPP/NHP | temporal memory d32 × 16 modes × 2 layers (StackOverflow 3), addressed mark memory, race-of-clocks head |
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
| 7 Oct | G1 | P12 split 0, 30% labels | pretrained **0.846 / 0.521**; scratch 0.842 / 0.486 (AUROC / AUPRC) | AUPRC gain holds (+0.035), AUROC gain shrinks with more labels; replication on splits 1–2 queued |
| 7 Oct | G1 | P12 split 0, 100% labels | pretrained 0.872 / 0.569; scratch **0.875 / 0.583** | no gain with full labels: pretraining buys label efficiency (10%: AUPRC +0.061; 30%: +0.035), not a higher ceiling. Splits 1–2 replication at 10%/30% queued |
| 7 Oct | G1 | P12 split 1, 10% labels | pretrained 0.844 / **0.540**; scratch **0.853** / 0.524 (AUROC / AUPRC) | AUPRC gain replicates in sign (+0.017, smaller than split 0's +0.061); AUROC −0.008. Pretraining helps the positive-class ranking at low labels, not overall AUROC on this split; split 1 30% and split 2 next |
| 7 Oct | G1 | P12 split 1, 30% labels | pretrained 0.879 / **0.638**; scratch 0.880 / 0.624 | AUPRC +0.014, AUROC level (−0.001). Across the four low-label comparisons so far (splits 0–1, 10%/30%) pretraining raises AUPRC every time (+0.061, +0.035, +0.017, +0.014); AUROC moves −0.008 to +0.013. Split 2 running |
| 7 Oct | G2 | one shared temporal core (d32 × 16 modes × 2 layers, 14,080 parameters) on Taxi, Taobao, StackOverflow, Amazon; per-dataset mark embeddings and clock heads; 68,910 parameters total vs ≈111K for four separate models; seed 0, DEV | Taxi **0.4899** (separate v5 seed 0: 0.4872), Taobao **1.2895** (1.2840), StackOverflow −2.1859 (v8 3-layer −2.177), Amazon 0.6827 at epoch 148, still rising (v8 seeds 0.69–0.77) | **Pass** (better on 2, within 0.01 on a third). Amazon needs more steps than the joint schedule gave it: the others peaked at epochs 26–39. Next: per-dataset step budgets (Amazon oversampled), then TEST |
| 7 Oct | G1 | P12 split 2, 10% labels | pretrained **0.858** / 0.509; scratch 0.852 / **0.515** | AUROC +0.006, AUPRC −0.007: the AUPRC gain does not hold on split 2. Five 10–30% comparisons: AUPRC +0.061, +0.035, +0.017, +0.014, −0.007 (mean +0.024); AUROC mean +0.003. Pretraining at 10 epochs gives a positive but split-dependent label-efficiency gain; a longer pretraining budget (the encoder's next-measurement loss was still falling) is the next lever |
| 7 Oct | G1 | P12 split 2, 30% labels | pretrained 0.890 / 0.607; scratch **0.895 / 0.638** | loss for 10-epoch pretraining (AUPRC −0.031). Six 10–30% comparisons: AUPRC +0.061, +0.035, +0.017, +0.014, −0.007, −0.031 (mean +0.015); AUROC mean +0.002. **10-epoch pretraining does not give a robust transfer gain.** The decision moves to the queued 40-epoch pretraining (splits 0–2, 10% labels); if it is not consistently positive, G1 is reworked (pretraining on P12+P19 unlabeled records, or a lower fine-tuning learning rate for the pretrained encoder) |
| 7 Oct | G1 | 40-epoch pretraining, 10% labels, splits 0/1/2 | AUROC 0.839 / 0.852 / 0.863 vs scratch 0.825 / 0.853 / 0.852; AUPRC 0.477 / 0.496 / 0.524 vs 0.446 / 0.524 / 0.516 | **G1 as designed does not establish transfer**: AUROC mean +0.008, AUPRC mean +0.004, within split spread; a longer pretraining budget did not help (10 epochs: +0.004 / +0.024). Design flaw: pretraining used the same P12 TRAIN records as fine-tuning, so it added an objective but no new data. **Redesign (G1b):** pretrain on P19's 31K unlabeled TRAIN stays (4× P12, different hospitals) with a channel map onto P12's variables, then fine-tune on P12 at 10% labels |
