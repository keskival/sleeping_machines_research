# Protocol asymmetry audit, native vs dense references (4 October 2026, 21:50 UTC)

User question: are we handicapping our model to match the Transformer's limits? Findings, in order of expected impact.
Each item names the evidence, the direction of the bias and the fix. Results land in FINDINGS and the scoreboard.

| # | Asymmetry | Evidence | Who it favours | Fix (status) |
|---|---|---|---|---|
| 1 | **Test-time state.** Saved LSTMs stream the whole 1M test interval with carried state; native and Transformer are scored in T=256 windows with state reset (E64 window protocol; ~2 evaluated positions per scored target). | `e64_lm_baselines.py: score()`; `language_batched_benchmark.py: window_scores()` | LSTM | Native streaming rescoring with the exact winner-only stepper (`experiments/language_stream_rescore.py`), queued first on curie. LSTM-in-T256-windows rescoring stays as the AWS diagnostic (`aws_lstm512_native_windows`) to measure how much of the LSTM lead is context. Both are reported; the Transformer remains window-bound by construction. |
| 2 | **Training context.** Native trains on 128-character segments with state reset; LSTM/Transformer train on 256-character windows. The native model never learns to use memory beyond 128 characters, although persistent memory is its core mechanism. | args `--segment 128`; E64 `--ctx 256` | LSTM, Transformer | Indication: same p96 weights score 1.8886 at T128 and 1.8885 at T256 (no use of longer context yet). Fix: train with state carried across consecutive segments (truncated credit, detached state), same compute per character. Needs a new training module (compiled_episodes.py hashes are frozen by AWS manifests). Proposed next integrated stage. |
| 3 | **Tuning budget.** P0-6 gives each reference budget 4–6 tuned arms (lr, warmup, dropout, validation early stopping); native runs use one recipe (lr .004 cosine, final weights). | TUNED_BASELINES.md; v6 queue | references | Equal tuning budget for native (lr .002/.003 arms already queued on curie; add validation selection of checkpoints). |
| 4 | **Evaluation routing.** Native is scored with sampled races; deterministic (argmax) routing was ~0.01 bpc better on earlier models at no extra cost (4-seed averaging ~0.02, at 4x inference). | language route diagnostics (p32/d8: 2.470 sampled, 2.461 argmax, 2.448 mix4) | references | Deterministic streaming variant queued (`--deterministic`); a deployment declares its routing mode before test. |
| 5 | **Compute conventions.** Native compute is an operator trace of every forward/backward op including elementwise and special functions; references are shape estimates (backward = 2x forward, approximate elementwise). | `lm_training_flops.py` vs the native tracer | unknown, plausibly references | Trace one reference forward/backward window with the same tracer (no training) and report the ratio; until then the conventions-differ note stays. |
| 6 | Not an error: native inference compute is quoted per evaluated position, so the window overhead (~2 positions per target) was never charged in the numbers; streaming makes the deployed per-target number equal to it. | scoreboard notes | — | — |

The headline matched-compute wins against the Transformers are unaffected by item 1 (both windowed). Items 1–4 can only
move native quality up relative to the LSTM rows; item 2 is the structural one and the largest expected gain.

## Measured, 4 October 22:35 UTC

- **Item 1 does not matter on this data.** Streaming equals windowed scoring for both the tuned LSTM-512
  (1.8255 / 1.8255) and native p96 (1.8889 / 1.8885).
- **Item 2 is unlikely to matter on this data.** Both models stop improving after 16–32 characters of history.
- The 0.06–0.07 bpc gap is present from 2–4 characters of history (FINDINGS, 4 October). The levers are local
  modelling and items 3–4.
