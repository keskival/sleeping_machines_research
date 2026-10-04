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
