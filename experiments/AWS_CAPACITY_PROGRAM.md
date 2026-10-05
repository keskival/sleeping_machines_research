# AWS: capacity-utilization and weight-decay arms (prepared 5 October 2026, 04:00 UTC)

Theory: THEORY note 155 (§§422–428) and note 59 §§419–421. The small arms run on curie (the FAS recruitment sweep R0–R7
and the grokking testbed G1–G4). The language arms below are larger and go to AWS slots after the current P0 owners
(P0-1, P0-3 and the P0-6 90M arms), in this order. One job per queue; through `run_safe.sh` with the usual caps and at
least 8 GiB MemAvailable.

| Order | Queue | What it decides | Estimate (1 thread) |
|---|---|---|---|
| 0 | `aws_language_wd_smoke_20261005T040000Z` | wrapper smoke (3 windows) | minutes |
| 1 | `aws_language_10M_wd0.01_p64d4_4pass_20261005T040000Z` | §421: does parameter weight decay close part of the local (2–4 character) gap to the tuned LSTM? Compare 1.955 (Adam) and LSTM-384 1.915 at the same ~107 TF | ~3 h (p64 4-pass on curie: ~2.5 h) |
| 2 | `aws_language_10M_wd0.1_p64d4_4pass_20261005T040000Z` | same, stronger decay | ~3 h |
| 3 | `aws_language_10M_x1_p32d4_pool32_tied_sampled_20261005T040000Z` | X1: capacity curve at large pool with shared maps (§422.4); should beat pool 2's 2.370 if capacity is used | ~1–2 h (sampled credit ~flat in pool) |
| 4 | `aws_language_10M_x1_p32d4_pool8_tied_sampled_20261005T040000Z` | X1 midpoint | ~1–2 h |
| 5 | `aws_language_10M_x1_p32d4_pool32_untied_sampled_20261005T040000Z` | X1 untied control; §419 Proposition 3 predicts it is worse than tied | ~1–2 h |

**Notes:**
- The weight-decay wrapper (`experiments/language_wd_benchmark.py`) leaves the pinned base driver's bytes unchanged. It
  swaps Adam for AdamW and records `weight_decay` in the result JSON. Run a `--max-windows 3` smoke under a separate tag
  first, since the wrapper is new.
- The estimates are not measurements. Set the RSS caps from a short smoke; pool 32 holds 16× the slot state.
- Compute is reported as for every native language row: traced fitting work and winner-only inference.
