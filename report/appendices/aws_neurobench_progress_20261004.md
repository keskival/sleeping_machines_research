# AWS official benchmark progress — 4 October 2026

Mackey–Glass r1: **10 of 30 official repeats completed**. Fixed primary mix8
mean sMAPE is **13.598096** on those first ten repeats. The remaining repeats
are pending; this partial score does not establish a full-protocol benchmark win.
Argmax19.052840 and sampled18.374131 are reporting-only variants, not alternative
primary selections. Repeat windows overlap, so seed/repeat spread is not an
independent-sample confidence interval. Eight stream predictions and their
state traffic are paid; inference work and energy remain unmeasured.

The [immutable aggregate](../../experiments/results/diagnostics/aws_mg_official_r1_summary_n10_20261004T184500Z.json)
binds completed JSONs, data, sources and settings. The automated publisher will
create separate20/30-repeat aggregates and a completed-protocol appendix.

Primate reaching: all six official data files are downloaded and verified
against vendor MD5s, with SHA256 and URL provenance retained. Existing frozen
r1 queues are prepared for five untouched sessions first and the previously
inspected development session last. Numerical confirmation has not started.
The future report must show both six-session and five-untouched-session means.

Depth8 private replay and teacher experiments continue independently with exact
checkpoint recovery. Their partial training checkpoints are not held-out final
language quality or a public benchmark result.
