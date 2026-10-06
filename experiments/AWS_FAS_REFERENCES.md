# AWS: FAS benchmark references (P0-7; prepared 4 October 2026, 22:40 UTC)

Benchmark: experiments/FAS_BENCHMARK.md. Native arms run on curie; the dense, SSM and point-process-style references
run here (dense training is reserved for AWS). One-job queues; each goes through `experiments/queue/run_safe.sh`, never
beside another job's memory headroom.

## Order

1. `queue/aws_fas_v1_data_20261004T224000Z.txt` regenerates dataset `fas_v1_20261004` from the same seeds
   (generate.py, vendored simulator and SimPy). It is CPU-only and takes about 4 min.
   - Check: per-split `runs` and `events` must equal `experiments/results/fas/fas_v1_20261004_manifest.json`.
   - The npz sha256 may differ with the zlib version; the event counts may not.
   - Peak memory is under 1 GB.
2. The five references, in any free slot after the P0-1/P0-6 owners. Each is single-threaded with 3 epochs over 10K clean
   runs (~10.6M process events per epoch).

| Queue | Model | Expected wall time (estimate, 1 thread) | Suggested caps |
|---|---|---|---|
| `aws_fas_v1_lstm_d128L2_20261004T224000Z` | LSTM, gap input (RMTPP pattern) | ~1 h | RSS 3 GB |
| `aws_fas_v1_transformer_d128L2_20261004T224000Z` | causal Transformer + continuous-time encoding (THP input pattern) | 1–3 h | RSS 5 GB |
| `aws_fas_v1_lru_d128L2_20261004T224000Z` | LRU-style diagonal SSM (FFT) | ~1 h | RSS 4 GB |
| `aws_fas_v1_s5t_d128L2_20261004T224000Z` | S5-style SSM discretized by each real gap (scan) | 2–4 h | RSS 5 GB |
| `aws_fas_v1_mamba_d128L2_20261004T224000Z` | minimal Mamba/S6 selective SSM (scan) | 2–5 h | RSS 6 GB |

The wall times and caps are estimates, not measurements:
- Run a 2-window smoke first (`--max-windows 2 --eval-runs 8`, unique smoke tag) and set the caps from its peak RSS.
- Preserve at least 8 GiB MemAvailable.

Every reference uses:
- the same head, loss and scoring as the native driver;
- training on clean runs only;
- selection by validation-clean NLL only;
- the test AUROC computed once, with the selected weights.

Work is a shape estimate per event (the `work` field).

## Reporting

- Results land in `experiments/results/aws_20260929/<run tag>/`.
- The FAS appendix compares, at each prefix N, AUROC overall and per fault type, training work and inference work per
  event, for native and references together (same units per column).
- Classical test baselines: `experiments/results/fas/fas_v1_classical_test_20261004T221500Z.json`.

## Renewed user order — 5 October 2026, 21:07 UTC

The user explicitly requests more seeds and the benchmarks. Native FAS v1
seeds7/8 are scheduled on curie after already-admitted language jobs. Execute
the five existing reference families on AWS after their bounded two-window
smokes and source/data checks; preserve current running jobs and the bounded
slot scheduler. No completed neural-reference result is visible in the shared
repository at this observation. The curie workspace has no authenticated AWS
execution connection; this is an execution order for the physical AWS owner,
not a claim that remote jobs were started.

Use anonymous logs for every fitted input. Hidden-identity-trained FIFO and
timing diagnostics are oracle-assisted and excluded from fair-reference
selection. Retain arithmetic versus special-function conventions and report
quality/work together; current native work is first-window extrapolated, so
no complete-work matched-compute claim follows automatically.


## Tree-reference gap — 6 October 2026

The shared completed FAS results contain no tree-based detector. The replicated headline compares three native seeds with the six named anonymous generic timing/count controls; it is not a win against trees. Completed tree comparisons in other directories concern tabular datasets and must not be presented as FAS references.

A prospective tree reference must fit exclusively on the same anonymous clean FIT logs, select from validation-clean data under the common budget, and produce a causal prefix anomaly score. No hidden item identities, clean/fault labels unavailable to the native fit, future events, or test-driven feature/hyperparameter choices may enter fitting. Include feature construction, fitting and prefix scoring in its resource record. Before execution, freeze an explicit new protocol/manifest with feature semantics, clean-training objective, selection rule, resource boundary and output provenance. Preserve the existing sealed-v2 manifest; adding a reference requires a source-bound amendment rather than silently redefining it. A v1 follow-up uses an exploratory label because its test comparisons are already visible. No tree run has been admitted by this gap record.

## Withdrawn — 6 October 2026 (user direction: no new external-architecture training)

AGENTS.md forbids new Transformer, LSTM and other external-architecture training on any host. The five queues above
(`aws_fas_v1_{lstm,transformer,lru,s5t,mamba}_d128L2_20261004T224000Z`) are withdrawn and must not be admitted.
Correction (16:10 UTC): the v1 data regeneration queue (`aws_fas_v1_data_20261004T224000Z`) generates data, not a model,
and stays valid. Native B3 development arms on AWS need it first. None had run. FAS comparisons use the information-matched classical references
(FAS_V2_CONFIRMATORY_PROTOCOL.md amendment of 6 October). dense.py remains the reference implementation published with
the benchmark for outside submissions.

## B3 own-model development arms on AWS — request, 6 October 2026, 16:10 UTC (curie FAS session)

Curie is held by the other container's queue, and the curie window for the FAS development arms has not been granted.
Request for the first free AWS gym slot, one one-thread job at a time, in this order. These are our own model, not
external architectures:
1. `queue/aws_fas_v1_data_20261004T224000Z.txt` (data; ~4 min, <1 GB). Check event counts against
   `results/fas/fas_v1_20261004_manifest.json`.
2. `queue/aws_fas_dev_R2_posterior_20261006T1610Z.txt` (race readout plus posterior-routed writes, THEORY §§433–434;
   ~1.3 h at curie speed, ~2.5 GB RSS).
3. `queue/aws_fas_dev_E1_race_20261006T1610Z.txt` (the matched control; ~1 h, ~2.5 GB).
4. `queue/aws_fas_dev_R2_particles_20261006T1610Z.txt` (evaluation only; ~25 min).

Use a two-window smoke first to set the RSS caps. native.py and native_race_readout.py now have a top-level `OUT` for
aws_benchmark.py; default paths are unchanged. If curie grants its window first, the curie copies run there and the
AWS copies are dropped. Never run both.

**Update 17:15 UTC: v2 supersedes the v1 request above.** Stage 1 selected the v2 setting (K = 2, p = .02, δ = 0;
B3_DEVELOPMENT_LOG.md). In the first free AWS gym slot, in order:
1. `queue/aws_fas_v2_data_20261006T1715Z.txt`;
2. `queue/aws_fas_v2_dev_C1_20261006T1715Z.txt`;
3. `queue/aws_fas_v2_dev_C2_20261006T1715Z.txt`.

The v1 arms (R2, E1, particles) become optional diagnostics. As before, the AWS and curie copies of the same arm never
both run: the first to start owns it, and the other is dropped.
