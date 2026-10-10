# Battle B2 — irregular multivariate time series classification (P12, P19, PAM)

Owner: AWS host session (opened 6 October 2026). Orders: PRODUCT_ORDERS.md. Attempt level: **study** (no fits yet).

## Protocol (Raindrop splits, used by every recent paper)

- Data: Raindrop processed releases on figshare, CC BY 4.0 — P19 (doi 10.6084/m9.figshare.19514338), P12
  (19514341), PAM (19514347). Downloaded to `data/raindrop/` (P12, P19; PAM 664 MB pending).
- Five fixed splits per dataset (`splits/` in each release), 80/10/10 train/validation/test; report mean ± sd over the
  five test splits. Selection on validation only.
- Metrics: P12 in-hospital mortality and P19 sepsis — AUROC and AUPRC (highly imbalanced); PAM eight-way activity —
  accuracy, precision, recall, F1.
- Safety: the processed arrays are pickled NumPy object arrays. Load them only through an allow-list unpickler
  (NumPy and builtin container types), never with `allow_pickle=True` on the raw file.

## Published references (MTM, arXiv 2509.17809, Table 3; mean ± sd over the 5 splits)

| Model | P12 AUROC | P12 AUPRC | P19 AUROC | P19 AUPRC | PAM Acc | PAM F1 |
|---|---:|---:|---:|---:|---:|---:|
| GRU-D | 81.9 ± 2.1 | 46.1 ± 4.7 | 83.9 ± 1.7 | 46.9 ± 2.1 | 83.3 | 84.8 |
| mTAND | 84.2 ± 0.8 | 48.2 ± 3.4 | 84.4 ± 1.3 | 50.6 ± 2.0 | 74.6 | 76.8 |
| Raindrop | 82.8 ± 1.7 | 44.0 ± 3.0 | 87.0 ± 2.3 | 51.8 ± 5.5 | 88.5 | 89.8 |
| ContiFormer | 82.1 ± 2.2 | 44.8 ± 3.5 | 84.4 ± 2.1 | 50.4 ± 4.3 | 85.2 | 86.3 |
| Warpformer | 86.5 ± 1.2 | 54.3 ± 2.7 | 88.7 ± 2.0 | 52.4 ± 4.5 | 94.2 | 94.9 |
| ViTST | 85.1 ± 0.8 | 51.1 ± 4.1 | 89.2 ± 2.0 | 53.1 ± 3.4 | 95.8 | 96.5 |
| GraFITi | 86.6 ± 1.1 | 54.8 ± 3.1 | 89.1 ± 2.6 | 56.6 ± 6.3 | 96.0 | 96.1 |
| **MTM (2025)** | **88.0 ± 1.0** | **58.6 ± 4.1** | **90.3 ± 2.0** | **58.3 ± 5.3** | **97.5 ± 0.2** | **97.6 ± 0.2** |

Split-to-split spread is 1–2 AUROC points; a credible win needs a margin beyond it. Check for newer published results
before freezing the target.

## Why the family should fit, and what to build

Each patient record is many channels sampled at their own irregular times, with informative missingness. In our
substrate every observation is an addressed event (channel, value, time); channel memories decay and rotate with
elapsed time; silence is information (a channel not measured is a clinical decision); typed comparisons (thresholds on
lab values) can enter before neural mixing; the classifier reads persistent state at the end of the record. B1's race
encoder (temporal memory layers + addressed per-mark memory) is the starting point, with an addressed per-channel value
memory and a classification readout. Development follows AGENTS.md: design rationale, iteration with error analysis on
validation, then the five-split protocol.

## Development log

| Date | Iteration | Change | Validation result | Diagnosis / next |
|---|---|---|---|---|
| 6 Oct | build | allow-list loader; cache (P12 11,988 records × ≤214 steps × 36 channels; P19 38,803 × ≤60 × 34; official split sizes 9,590/1,199/1,199 and 31,042/3,880/3,881); event-native classifier `race_irts.py` | — | |
| 6 Oct | smoke | P12 split 0, 2 epochs | val AUROC 0.849, AUPRC 0.516 (69 s/epoch, 2.8 GB) | full development fit queued |
| 6 Oct | r1 | P12 split 0, d32, 2 layers, typed comparisons J=4, dropout 0.2, 41,833 parameters | best val **AUROC 0.866**, AUPRC 0.549 at epoch 5; overfits after | MTM TEST 0.880 ± 1.0; round 2: regularization (dropout 0.4 + decay; lower lr + weight EMA) |
| 7 Oct | r2 | dropout 0.4 + decay 1e-3 | val AUROC 0.859 (best epoch 4) | regularization does not help |
| 7 Oct | diagnosis | gradient-boosted trees on per-channel summaries (count, mean, min, max, first, last, last−first, last time) + statics, split 0 | val **AUROC 0.867, AUPRC 0.581** | our model (0.866 / 0.549) extracts no more than record summaries yet |
| 7 Oct | v3 | addressed channel slots carry sufficient statistics (THEORY note 59): count, running mean, min, max, first, last, trend, staleness; readout mixes them with temporal state | queued | |
| 7 Oct | r2 EMA | lr 7e-4 + weight EMA | val AUROC 0.860 | no gain |
| 7 Oct | v3 | statistic-valued channel slots | val **AUROC 0.872**, AUPRC 0.575 (best epoch 4; 65,001 parameters) | above the summary-tree diagnostic (0.867); MTM TEST 0.880. v4: typed comparisons on the statistics queued |
| 7 Oct | v3 EMA / v4 | weight EMA on v3; v4 soft thresholds on statistics (145K parameters) | 0.871; 0.868 (best epoch 2) | v4 overfits faster; v3 retained. Next: P19 standing, slower lr on P12 |
| 7 Oct | v3 P19 | statistic slots, P19 split 0 | val **AUROC 0.919**, AUPRC 0.626 | MTM TEST 0.903 ± 2.0; P19 five-split protocol next |

**Pre-registered P19 protocol (fixed 7 Oct ~02:20 UTC, before any of its runs).** `race_irts_v3.py` defaults (d32, 2
layers, J=4, dropout 0.2, lr 2e-3, batch 128, patience 8, ≤ 40 epochs), seed 0, the five official splits; checkpoint
selected by validation AUROC; TEST scored once per split. Report mean ± sd of TEST AUROC and AUPRC over the five splits
against MTM 90.3 ± 2.0 / 58.3 ± 5.3.
| 7 Oct | v3 lr 5e-4 | slower learning on P12 | val AUROC 0.870 (best epoch 10) | no gain over lr 2e-3 (0.872) |
| 7 Oct | **final P19 (v3)** | pre-registered: v3 defaults, five official splits, validation-AUROC selection, TEST once per split | per split AUROC 0.942 / 0.923 / 0.883 / 0.926 / 0.907; **TEST AUROC 0.916 ± 0.022, AUPRC 0.639 ± 0.039** (62,681 parameters) | **WIN over the best published model on the official splits** (MTM 0.903 ± 0.020 / 0.583 ± 0.053): AUPRC +0.056 (beyond both split spreads), AUROC +0.013 (within one split sd). Literature check 7 Oct: arXiv 2602.19531 reports ≈0.897 with its own 5-fold CV (different protocol); QuITE (2605.28166) does not report P19 |

**Pre-registered P12 protocol (fixed 7 Oct ~03:50 UTC, before any of its runs).** `race_irts_v3.py` defaults (d32, 2
layers, J=4, dropout 0.2, lr 2e-3, batch 128, patience 12, ≤ 60 epochs), seed 0, five official splits, validation-AUROC
selection, TEST once per split; against MTM 88.0 ± 1.0 / 58.6 ± 4.1.
| 7 Oct | **final P12 (v3)** | pre-registered: five official splits, validation-AUROC selection, TEST once per split | per split AUROC 0.875 / 0.885 / 0.864 / 0.877 / 0.856; **TEST AUROC 0.871 ± 0.011, AUPRC 0.585 ± 0.024** | **Behind MTM on AUROC** (0.880 ± 0.010), **level on AUPRC** (0.586 ± 0.041); ahead of GraFITi 0.866, Warpformer 0.865, ViTST 0.851. Developed attempt on the P19 configuration; P12-specific development (longer records: up to 214 steps, 36 channels) is the next lever |
| 7 Oct | PAM r6 (v5) | statistic slots + multi-class head, split 0 | val acc **0.946**, macro-F1 0.950 | MTM TEST 0.975; behind. v6: variance and mean absolute change per channel; wider memory variant queued |
| 7 Oct | PAM r7 (v6) | + variance and mean absolute change per channel | val acc 0.940 (best epoch 11) | no gain; training loss → 0: memorization on 4,266 records. v7: random crops + amplitude jitter queued |
| 7 Oct | P12 ensemble probe (validation, split 0) | average of 5 existing split-0 models (low diversity: one seed, two identical configs) | 0.875 / 0.582 vs best member 0.875 / 0.583 | ensembling adds ≈ +0.003 here; not the lever for the 0.009 AUROC gap to MTM |
| 7 Oct | model size | MTM reports 873K parameters (on PAM; Raindrop 150M, ViTST 87M, GraFITi 1.1M, Warpformer 378K) | ours 62–65K | MTM's P19 size is not reported; comparing our P19 model with MTM's PAM count is not a like-for-like size claim (corrected 7 Oct 17:10: earlier wording said "≈ 14× fewer parameters than MTM" on P19). MTM FLOPs not reported |
| 7 Oct | P12 seed ensemble (validation, split 0) | protocol configuration, seeds 0–4 (seed 0 = the final split-0 model) | members 0.865–0.873 AUROC (mean 0.870), 0.563–0.591 AUPRC; **ensemble 0.876 / 0.595** | seed diversity adds +0.006 AUROC, +0.017 AUPRC over the member mean. Projected onto TEST (0.871 / 0.585) the ensemble would still trail MTM on AUROC (0.880) while leading on AUPRC (0.586): not a clean win, so no five-split ensemble protocol yet. P12 needs a model-side gain; candidates: v7 crop/jitter augmentation (PAM +0.024 validation accuracy) |
| 7 Oct | PAM r8 (v7) | random 80% crops + 10% amplitude jitter in training, full records at evaluation; 46,316 parameters | **val acc 0.972, F1 0.975** (best epoch 46 of 62; v5 0.946, v6 0.940) | augmentation removes the memorization ceiling: +0.026 accuracy. Within 0.003 of MTM's TEST 0.975 / 0.976 on development data. Epoch-to-epoch swings 0.946–0.972: r10 (EMA 0.999, 120 epochs) running; the PAM five-split protocol is pre-registered on whichever of r8/r10 selects more stably |
| 7 Oct | P12 r11 (v7) | crop 0.8 + jitter 0.1 on split 0 | val 0.867 / 0.560 (best epoch 4 of 25) vs protocol model 0.872 / 0.575 | **no gain**: the epoch-2–5 peak survives augmentation, so P12's limit is not the memorization that augmentation removed on PAM. Crop 0.6 arm running; if it also fails, the next diagnosis is the label signal itself (per-record error analysis of the misranked positives against the statistic slots) |
| 7 Oct | P12 r11 (v7) crop 0.6 | stronger crops | val 0.865 / 0.543 (best epoch 6) | **no gain**; augmentation is closed for P12. Next P12 step: error analysis of misranked positives |
| 7 Oct | PAM r10 (v7 + EMA 0.999) | weight averaging, 120-epoch budget | val acc plateau **0.959** (stable, no swings), below r8's 0.972 peak | r8's peak partly reflects selection over noisy epochs; the stable level is ≈ 0.96 vs MTM 0.975. Round 5: wider memory under augmentation, and stronger augmentation |
| 7 Oct | P12 error analysis (validation, split 0, protocol model) | AUROC by tercile of observations: 0.809 / 0.882 / 0.898; by steps: 0.815 / 0.857 / 0.903. Misranked positives carry 347 observations vs 455 for well-ranked ones | the gap to MTM sits in sparsely measured stays | in sparse records which tests are ordered and when carries the signal; the slots held count and staleness but not ordering latency. **v8**: + per-channel time of first measurement and measurement rate, + record duration and steps (contract PASS on a hand-computed batch). Queued on P12 and P19 split 0 |
| 7 Oct | P12 r13 (v8 ordering latency) | + time of first measurement and measurement rate per channel, record duration | val 0.872 / 0.569 (best epoch 2) vs protocol model 0.872 / 0.575 | **no gain**. Count and staleness already carried the ordering signal the model can use; the epoch-2 peak persists across lr, augmentation, ensembling and inputs. P12 is paused at TEST 0.871 / 0.585 (level with MTM on AUPRC); slots go to PAM. v8 on P19 runs as a regression check |
| 7 Oct | P19 r13 (v8) | regression check of the ordering-latency inputs | val 0.918 / 0.626 vs protocol model 0.919 / 0.626 | no regression, no gain |
| 7 Oct | PAM r10 (v7 + EMA 0.999) | weight averaging, 120-epoch budget | val acc **0.9756**, F1 0.976 at epoch 81, holding 0.974–0.976 over the last five epochs | the averaged model is stable at MTM's level (TEST 0.975 ± 0.002 / F1 0.976); this configuration goes to the protocol |

**Pre-registered PAM protocol (fixed 7 Oct 19:42 UTC, before any of its runs).** `race_irts_v7.py` with random 80% crops,
10% amplitude jitter, EMA 0.999 of the weights (selected and scored), batch 64, ≤ 120 epochs, patience 25, seed 0;
five official Raindrop splits; checkpoint selected on validation accuracy; TEST scored once per split with
`--score-test`. Report: TEST accuracy and macro F1 (also precision and recall) as mean ± sd over the five splits with
every split's value, whatever they are, against MTM 97.5 ± 0.2 accuracy / 97.6 ± 0.2 F1 (higher is better). A win
requires the mean accuracy ahead; ties are stated as ties. Split 0's development run (r10) is not reused. (Audit note: this text was written before split 0 started at 19:42:20 and committed at 19:43, before
its first epoch completed.)
| 7 Oct | PAM r10 final (v7 + EMA 0.999) | 120 epochs | **val acc 0.981, F1 0.983** (best epoch 117 of 120; 46,316 parameters) | development level above MTM's TEST 0.975 / 0.976; still improving at the epoch cap. The pre-registered five-split protocol (same configuration) is running: splits 0 and 2 started |

**Reverse-ablation diagnostic (curie, 7 Oct 22:31 UTC; user request).** Gradient-boosted trees on the per-channel
statistics our slots hold (count, mean, min, max, first, last, last − first, last time) plus statics, five official P19
splits, rounds chosen on the official validation split, TEST once per split (`experiments/irts/summary_tree_diagnostic.py`,
`results/irts/curie_b2_p19_summary_tree_20261007T2320Z.json`): **AUROC 0.914 ± 0.020, AUPRC 0.618 ± 0.041**, against ours
0.916 ± 0.022 / 0.639 ± 0.039 and MTM 0.903 / 0.583. The P19 margin over MTM comes mostly from the statistic-valued
representation; our model is level with the trees on AUROC and +0.021 on AUPRC, within split spread. Next decisive
test for B2: a margin over the statistics reader (temporal interactions the summaries cannot express), measured
paired per split.

**Paired per-split comparison, ours − trees (TEST, from the result files):** ΔAUROC +.010 / +.007 / −.004 / −.006 / +.005
(mean +.002); ΔAUPRC +.037 / +.008 / +.042 / −.000 / +.017 (mean +.021, 4 of 5 splits ahead, paired t ≈ 2.6, p ≈ .06).
**Queued on curie (user direction, 7 Oct 23:40):** `experiments/irts/p19_complementarity.py`, five splits. Per split: our
model rerun from scratch (also an independent-hardware reproduction of the P19 result), a statistics-only ablation of the
same network (temporal head inputs zeroed; zero gradient into the temporal layers, verified), trees on the same statistics,
and a validation-weighted rank blend of ours and trees. Full > statistics-only isolates what the temporal memory adds;
blend > both shows information the summaries lack.
| 7 Oct | P12 G1j (queued) | joint silence-aware supervision: next-measurement log-loss (when, which channels, values) added to the classification loss with weight 0.02 / 0.05 / 0.15; weight 0 is an exact-reproduction control of G1 scratch (0.875 / 0.583) | — | hypothesis: dense per-step supervision keeps the memory learning after the label signal is exhausted (epoch 2–5 peak) |
| 8 Oct | PAM r12 (v7, d48 × 48 modes) | wider memory under crops 0.8 + jitter 0.1, no EMA | val acc 0.968, F1 0.971 (best epoch 34; 98,652 parameters) vs 0.972 at d32 (46,316) | **no gain from capacity** even without the memorization ceiling: the small model is the right size for PAM. The protocol configuration (d32 + EMA) stands |
| 8 Oct | **PAM protocol, split 0** | pre-registered configuration | TEST acc **0.979**, F1 **0.983** (val 0.981, reproducing r10 exactly; best epoch 117) | first of five splits; MTM 0.975 ± 0.002 / 0.976. No verdict until all five splits are scored |
| 8 Oct | PAM r12 (v7, crops 0.6, jitter 0.15) | stronger augmentation, no EMA | val acc 0.972, F1 0.973 (best epoch 70) | same as crops 0.8 / jitter 0.1 without EMA (0.972): augmentation strength is not a lever; EMA is |
| 8 Oct | **PAM protocol, split 2** | pre-registered configuration | TEST acc **0.981**, F1 **0.984** (val 0.987; best epoch 115) | second of five splits; both scored splits ahead of MTM 0.975 / 0.976. Splits 1, 3, 4 running or queued |
| 8 Oct | P12 G1j control (aux 0) | equivalence contract | val 0.874708 / 0.583319, identical to G1 scratch at 100% labels | **PASS**: the joint driver reproduces the baseline exactly; weighted arms running |
| 8 Oct | **PAM protocol, split 1** | pre-registered configuration | TEST acc **0.981**, F1 **0.983** (val 0.978; best epoch 73) | third of five splits; all three scored splits ahead of MTM 0.975 / 0.976 |
| 8 Oct | P12 G1j aux 0.02 / 0.05 | joint next-measurement loss | val 0.871 / 0.578 (best epoch 4); 0.875 / 0.579 (best epoch 9) vs control 0.875 / 0.583 | the dense loss delays the peak (epoch 4 → 9) but does not raise it; aux 0.15 running |
| 8 Oct | P12 G1j aux 0.15 | joint next-measurement loss | val 0.876 / 0.581 (best epoch 14) vs control 0.875 / 0.583 | **closed for P12**: across weights 0.02 / 0.05 / 0.15 the peak moves later (epoch 4 → 9 → 14) and AUROC drifts +0.001, within noise and far from the +0.009 needed. Six well-founded levers (lr, augmentation, ensembling, ordering-latency inputs, same-cohort pretraining, joint dense supervision) leave P12 at TEST 0.871 / 0.585 (level with MTM on AUPRC). Remaining P12 test: G1b cross-cohort pretraining (running) |

**P19 complementarity result (curie, 8 Oct 05:34 UTC; five official splits, TEST once per model; results
`results/irts/curie_b2_p19_compl_split*_20261008T0000Z.json`, predictions `*_preds.npz`):**

| Model | AUROC | AUPRC |
|---|---|---|
| Full model (curie rerun) | **0.916 ± 0.022** | **0.644 ± 0.040** |
| Same network, statistics only | 0.900 ± 0.018 | 0.572 ± 0.037 |
| Trees on the same statistics | 0.914 ± 0.020 | 0.618 ± 0.041 |
| Rank blend, full + trees (weight 0.5 chosen on validation in every split) | 0.925 ± 0.021 | 0.671 ± 0.040 |

Paired per split: full − statistics only +0.017 AUROC (5/5, t 5.3), +0.072 AUPRC (5/5, t 31); full − trees +0.002
AUROC (3/5), +0.026 AUPRC (5/5, t 4.2); blend − better single model +0.007 / +0.027 (5/5 both). Reproduction: the
curie rerun matches the AWS protocol AUROC within 0.0004 on all five splits (AUPRC within 0.0003 except split 3,
+0.025). **Reading:** the temporal memory carries signal the statistics lack; the P19 win is not a statistics artifact.
**In-family improvement indicated:** our neural statistics reader is weaker than trees on the same statistics (0.900 vs
0.914), while the temporal memory adds what trees lack; a stronger statistic reader (soft thresholds / tree-like
splits on the slots, regularized against v4's overfitting) combined with the temporal memory targets the blend's 0.925 /
0.671 inside one model.

| 8 Oct | P12 G1b (cross-cohort pretraining) | see GENERALITY_PLAN | mean +0.004 AUROC at 10% labels, split-dependent | no robust gain; **P12 work paused** at TEST 0.871 / 0.585 after seven levers. Slots go to PAM, B1 unification and the next battles |
| 8 Oct | **PAM protocol, split 3** | pre-registered configuration | TEST acc **0.981**, F1 **0.985** (val 0.985; best epoch 106) | fourth of five splits; all four ahead of MTM 0.975 / 0.976. Split 4 running |
| 8 Oct | **final PAM (pre-registered protocol), 5 splits, TEST** | v7, crops 0.8, jitter 0.1, EMA 0.999, ≤ 120 epochs | accuracy **0.9775 ± 0.0074**, F1 **0.9803 ± 0.0080**, precision 0.9824, recall 0.9788; per split accuracy 0.979 / 0.981 / 0.981 / 0.981 / **0.964**; 46,316 parameters | **WIN** under the pre-registered rule (mean accuracy ahead of MTM 0.975 ± 0.002; F1 ahead of 0.976 ± 0.002). Four of five splits ahead; split 4 behind (best epoch 46, the earliest of the five). Our split spread (0.007) is wider than MTM's (0.002) |

**Development (curie, 8 Oct 08:14 UTC, validation only): quantile-encoded statistics reader** (`p19_quantile_reader.py`;
statistics encoded by the TRAIN empirical CDF, the tree's monotone invariance, entering the head beside the temporal
features; 104K parameters). Validation vs the protocol model: ΔAUROC +.003 / +.003 / −.004, ΔAUPRC +.000 / +.005 / −.009
(splits 0–2); mean +.001 / −.001. **No consistent gain; no TEST run.** The statistic encoding is not what limits the full
model; the temporal memory already supplies what the trees lack (complementarity result above).

**PAM v8 protocol, pre-registered (curie, 9 Oct 09:10 UTC; to run on AWS).** *Diagnosis (validation curves of the final
v7 protocol):* validation accuracy on ~530 records moves in steps of ~0.002 and saturates near ten errors; v7 selected on
accuracy with strict improvement and counted patience from the selected epoch. In four of five splits the selected
accuracy was tied again later (0 / 11 / 3 / 7 / 5 ties after the selected epoch for splits 0–4), so the earliest tied epoch
was kept; split 4 selected epoch 46 and stopped at 72 (test 0.964), split 1 stopped at 99, while splits 0, 2, 3 ran to
120 and peaked at epochs 106–117. *Disclosure:* the motivation includes having seen split 4's low TEST score; the rule
below is generic and applies to every split. *Rule:* `experiments/irts/race_irts_v8.py --select nll` selects on
validation cross-entropy (continuous; patience counts from its last improvement); configuration otherwise identical to
the v7 protocol (d 32, 16 modes, 2 layers, J 4, dropout 0.2, lr 0.002, wd 1e-4, batch 64, ≤ 120 epochs, patience 25,
EMA 0.999, crop 0.8, jitter 0.1, seed 0); five official splits; TEST scored once per split. Both v7 (0.9775 ± 0.0074) and
v8 are reported. *Decision:* v8 replaces v7 as the stated PAM result only if its five-split mean is higher; a clear win
needs every split ahead of MTM's 0.975 or a mean margin beyond our split spread.

**Current execution (curie, 9 Oct 20:54 UTC):** splits 4 and 3 use `race_irts_v9.py`, whose model, augmentation, optimizer, EMA and selection are identical to v8. The change adds exact epoch-boundary restart state and a one-time TEST reservation. The interrupted split-4 v8 run ended after epoch 23 without scoring TEST; its weights-only checkpoint is preserved, and seed 0 is restarted under a fresh tag. The recovery numerical contract checks forward parity and exact resumed optimizer/EMA/RNG updates (zero error). Five-split headline replacement still follows the rule above; AWS owns splits 0–2.

**Curie recovery update, 10 Oct 02:05 UTC:** v9 split 4 completed at TEST accuracy 0.966292, F1 0.967775, selected epoch 46/72, 17,367 s. Source-bound v8/v9 model/update equivalence is retained; the exact v9 checkpoint and one-time TEST reservation are preserved. Split 3 is live in the same admitted pipeline. This single completed split does not replace the leading five-split v7 result; the registered five-split v8 mean rule remains the decision. AWS retains splits 0–2.

**AWS PAM v8 split 0 (10 Oct 08:15 UTC, pre-registered v8 NLL-selection/EMA recipe, TEST once).** Selected epoch 111 of 120 by validation NLL, 46,316 parameters, 4.9 h on one CPU thread: TEST accuracy **0.979401**, precision 0.985073, recall 0.981133, F1 **0.982940**, NLL 0.0905 (MTM 0.975 / 0.976). Identical to v7 split 0 to every digit: training is deterministic and the two runs share the trajectory epoch by epoch; v7 selected epoch 117 by accuracy, v8 epoch 111 by NLL, and both checkpoints classify all 534 TEST recordings identically. v8 targets the early tied-epoch freeze seen on split 4, which split 0 never had. AWS splits 1 and 2 follow on slot 1; curie owns splits 3 (running) and 4 (completed, 0.966292).

**AWS PAM v8 split 1 (10 Oct 11:20 UTC, registered recipe, TEST once).** Selected epoch 45 of 71 by validation NLL (patience stop), 46,316 parameters, 3.0 h: TEST accuracy **0.970037**, precision 0.977043, recall 0.969893, F1 0.972834, NLL 0.0995. v7 on the same split: 0.981273 (accuracy selection, epoch 73 of 99). NLL selection stopped this split early rather than repairing a tied-epoch freeze. Running comparison over the completed v8 splits: split 0 0.9794 (= v7), split 1 0.9700 (v7 0.9813), split 4 0.9663 (v7 0.964); sum −0.009 against v7. The registered decision (v8 replaces v7 only if its five-split mean is higher) awaits splits 2 (AWS, running) and 3 (curie).

