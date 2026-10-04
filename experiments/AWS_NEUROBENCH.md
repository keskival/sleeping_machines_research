# AWS: NeuroBench primate reaching protocol run (r1)

Target: the NeuroBench non-human-primate motor prediction leaderboard (experiments/SOTA_TARGETS.md target 2): AEGRU R²
0.71, GRU-t1 0.707, bigRSNN 0.698, tinyRSNN 0.66; the leaderboard reports the mean over the six official sessions.

Six one-job queues `queue/aws_primate_r1_<session>_20261004T053500Z.txt`, one session each, so the guarded slots can run
them in parallel. Admit after the running 90M language arms when slots are free; contracts:
`-m pytest -q tests/test_mackey_glass_native.py tests/test_sparse_inference.py tests/test_sparse_training.py`. Data:
`data/neurobench/primate_reaching/*.mat` from https://zenodo.org/record/583331 (MD5s in the vendored neurobench loader;
about 4.3 GB). The driver uses the vendored official neurobench 2.3.0 loader (train_ratio .5, 4 ms bins) and the
official R² formula.

Model (fixed before this run, chosen on development session indy_20170131_02 by validation R²): integrated native core
p32/d2, pool 4 with shared maps and private state, route credit, compiled training, 4,000 steps of 32 × 250-bin windows,
causal leaky readout constant chosen on validation, predictions averaged over 4 race-noise streams. One seed (1). Expected
wall time: about 80–100 min per session on one thread; RSS under 2 GB.

Integrity: the development session's test R² was observed during development (selection used validation only). Report
the six-session mean and the mean over the five untouched sessions. Publish each completed result JSON.
