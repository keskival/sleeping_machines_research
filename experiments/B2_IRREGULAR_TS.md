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
