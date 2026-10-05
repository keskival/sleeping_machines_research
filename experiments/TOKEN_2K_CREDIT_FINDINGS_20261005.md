# Small integrated credit comparison and selection repair

CPU, seed6, GPT-2 FineWeb,2048training tokens and8160presentations,
1024development targets, P16/D2/H2/U4, B8/T16. Same frequency initialization,
data and64update budget. All fitting jobs used unique one-job queues/run_safe,
one thread,500000KiB RSS/5000000KiB VMS caps and8192MiB reserve. Larger8K fits
remain the user-directed AWS work; this separate2K diagnostic is not a repeat.

| Credit | Initial dev NLL | Best trained dev NLL | Final train NLL | Measured training targets/s |
|---|---:|---:|---:|---:|
| Local value |8.906910|9.131615|4.788195|404.09|
| Full suffix |8.906910|9.130419|4.778686|342.83|
| Four sampled positions |8.906910|9.130868|4.777276|351.84|

**All trained arms lose to their initial frequency model on development.**
Training fits improve while held-out likelihood worsens: this coordinate
overfits the small corpus. Full suffix beats local credit by0.001196NLL at
the best trained checkpoint. Four-position credit loses to full scoring by
0.000449NLL and measures2.63%higher throughput in this single run. These small
differences neither select a benchmark member nor justify extrapolating the
synthetic14.05%arithmetic saving to measured hardware speed. Historical result
files retain every curve, state-use metric, gradient norm and teacher audit.

The full teacher credits one cycling depth/head at the first token per chunk.
It does not sample all event positions or estimate every writer's future
utility. More complete site exposure, horizon and estimator strength remain
selectable learning coordinates. Future credit should be compared at an
adequate-data point before attributing this small-data overfit to the family.
The already prepared8K AWS arms supply the next data-scale test. Promotion must
beat initialization, not merely another overfitted trained checkpoint.

## Checkpoint-selection failure and repair

The existing driver initializes best=inf and considers only trained
checkpoints. Therefore its `best_dev_nll` excludes a better initial model in
these runs. Their reported numbers are preserved; the explicit correction is
`results/diagnostics/curie_integrated_2k_credit_comparison_20261005_v1.json`.
Initial weights were not saved in these historical fits; no unavailable
checkpoint is claimed.

`selected_token_language_lab.py` saves initial weights and emits a separate
selection record choosing the best initial/trained development checkpoint.
It delegates the exact learning trajectory to the pinned sampled driver,
preserves the raw result and binds the selection wrapper on resume. A guarded
actual6-update comparison matches model, AdamW, memory, all RNGs and cursor
exactly. Selected loss equals the minimum observed development loss and the
selected artifact exists. Evidence:
`results/diagnostics/curie_selected_token_parity_20261005_v1.json`.
Existing immutable AWS sources stay unchanged; their results must also be
checked against the initial score before promotion. Future recipes use the
corrected selection entry point.
