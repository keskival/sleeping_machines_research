# Product orders — what we are after (issued 4 October 2026, 19:20 UTC)

Owner: the user. Product owner: curie agent, on the user's instruction. **Every agent (curie, AWS, local) reads this
first and works these orders in priority order.** Definitions of win/loss: experiments/WIN_CRITERIA.md.

## What increases our valuation

Investors buy three things, in this order:

1. **A headline number against the incumbents.** "Our model matches or beats LSTM and Transformer language models at
   N× less compute" (training and inference), on a standard dataset, with the same data and test set. One number,
   reproducible, with a second seed.
2. **A public leaderboard entry.** "First place on a recognized benchmark" (NeuroBench), under its official protocol,
   with its own efficiency columns.
3. **A scaling curve.** The win holds or grows from 10M to 90M characters (and beyond), so it is a trend rather than a
   lucky point.

Everything else (diagnostics, theory notes, new primitives) is valuable only insofar as it produces or protects these.
Every deliverable below has a pass/fail number. Report it as a win or a loss in one line.

## P0 — do these first (all hosts)

| # | Deliverable | Host / queue | Pass criterion | Status |
|---|---|---|---|---|
| P0-1 | **90M matched-compute win** | AWS: `queue/aws_language_90M_r4_p64d4_4pass_linear_20261004T193000Z.txt`, then `..._r4_p96d4_4pass_...` (AWS_NATIVE_LANGUAGE_90M.md rev. 4) | test bpc ≤ 1.661 at ≤ 3.9 PF (beats LSTM-512); ≤ 1.604 at ≤ 8.0 PF (beats Transformer) | queued |
| P0-2 | **10M win against LSTM-512 at matched compute** | curie: p128/d4 4 passes, starts right after the FAS arm (~00:15 UTC 5 Oct); fallback p96/d4 pool 4 with sampled credit, 6 passes | test bpc < 1.799 at ≤ 433 TF; tuned LSTM-512 4.5p is 1.825 at budget A | queued. History curves (4 Oct): the gap to the LSTM is per-event local modelling (present from 2-4 chars of history, constant), not context or routing noise, so width is the direct test |
| P0-3 | **NeuroBench primate leaderboard** | AWS: `queue/aws_primate_admission_20261004T184500Z/manifest.json` (six-session r1) | six-session mean R² > 0.71 (AEGRU) | prepared |
| P0-4 | **NeuroBench Mackey-Glass leaderboard** | AWS r1 (repeats 20–29 running); curie expected-reception development → a pre-declared tau 17 run | 30-repeat sMAPE < 13.37 (LSTM), with a smaller footprint | 20/30: 14.37; third batch unblocked 4 Oct 23:55 (pins match, driver does not import the drifted file) |
| P0-6 | **Tuned dense references at our budgets** (added 4 Oct 21:00 UTC, user-directed) | AWS: ten `queue/aws_tuned_ref_10M_*_20261004T210000Z.txt` first (1.5–3 h each), then four `aws_tuned_ref_90M_*` (TUNED_BASELINES.md) | native beats the validation-selected tuned arm at each budget (A ≤ 352 TF: 1.888; B ≤ 107 TF: 1.955; 90M C/D once P0-1 completes) | running; 5/6 A arms done: LSTM-512 4.5p 1.826 (val-leading) and LSTM-384 1.840 beat native 1.888 (losses, context-independent); tuned TF192x4 2.027, TF128x4 1.996 and TF256x4 lr.002 2.053 are behind native 1.888 budget A 6/6 complete (TF256x4 lr.001 2.199); budget B 4/4 complete: LSTM-384 1.915, LSTM-256 1.934 beat native 1.955; TF192x3 2.315 and TF128x4 2.215 behind; admission waits for windowed LSTM validation rescore (below); curie native lr .003/.006 fairness arms queued after P0-2 |
| P0-7 | **Home-field async wins** (added 4 Oct 22:40 UTC, user-directed) | curie: FAS native arm (v6 queue, after the streaming evaluations), SHD aug/expected p32d4; AWS: FAS data + five references (AWS_FAS_REFERENCES.md) after the P0-1/P0-6 owners | FAS (experiments/FAS_BENCHMARK.md): native AUROC above every reference (LSTM, Transformer, LRU, S5-style, Mamba-style; classical) at N <= 256 process events, at no more training/inference work; SHD: a parameter- or work-matched result against small SNNs (cAdLIF 94.19% at 38.7K params; DCLS 95.07% at 0.2M) | queued |
| P0-5 | **Scoreboard** | report/scoreboard.py: REPORT.md / PDF page 2, generated from result files on every rebuild | every comparison listed as win / loss / efficiency point per WIN_CRITERIA | **done** (5 wins of 13) |

**AWS slot allocation order (revised 4 Oct 21:00 UTC):** P0-1, then P0-6 10M arms, then P0-3 take slots as soon as any
slot frees; P0-6 90M arms and the P1-1 seed `aws_language_10M_r1_p64d4_4pass_linear_s7_20261004T210000Z` follow. Without
P0-6, a matched-compute win against an untuned reference will not survive review, so it ranks with P0-1. The long depth-8 streaming
replay/teacher fits are research diagnostics: checkpoint them at their next milestone and suspend them (do not delete
anything) until P0-1 and P0-3 are running. Resume them afterwards.

**P0-6 blocking step (5 Oct 04:10 UTC): windowed VALIDATION rescore of the tuned LSTM arms.** Budget A is complete (6/6);
budget B is 3/4. `report/tuned_reference_admission.py` correctly refuses cross-architecture validation selection while
LSTM validation is carried-state and Transformer validation is windowed. Owner: curie (text8 + language_stream_rescore
tooling; inference only, minutes per arm). Rescore text8[90M:90.2M] with E64 T256 windows, the same scorer as the
Transformer arms, for A: lstm384_p6, lstm512_p4.5 and B: lstm384_p2.5, lstm256_p5 (+ any later LSTM arm). Save each
as a new result beside the original; never edit the AWS JSON. `reference_window_rescore.py` covers TEST only, so add a
validation interval mode or use language_stream_rescore. Expected outcome, not a substitute for the run: selection
unchanged (LSTM-512 val 1.769 vs best TF 1.950; test context effect 0.0000), so budget A = loss to tuned LSTM-512
1.826, and the Transformer sub-comparison = native win (1.888 vs TF128x4 1.996 at 101% compute).
**Taken by curie (5 Oct 01:15 UTC):**
- Queue: `queue/curie_lstm_val_window_rescore_20261005T011500Z.txt`, covering all four arms with
  `reference_context_curve.py --offset 90000000 --test 200000 --context-curve 0`.
- The scorer is the E64 window convention (first window whole, then second halves), checked against `e64_lm_baselines.score`.
- It runs first after P0-2 finishes (around 04:00 UTC), before taps3 and the native lr arms.
- Results land in `experiments/results/language_stream/curie_lstm_valwin_*`. The AWS JSONs are untouched.

**P0-7 FAS update (5 Oct 06:30):** identity-aware oracle bound .755 at N=256 / .913 at N=512, so the early range is
discriminating and native (pool 2, .600/.742) recovers 39%/59% of it. Curie queues: pool2 seed 7 (replication), pool 8,
pool 32 (per-item slots for 30 interleaved items), `curie_fas_native_p32d4_*_20261005T063500Z`.

## P1 — confirm and widen the wins

| # | Deliverable | Pass criterion |
|---|---|---|
| P1-1 | Second seed of every P0 win (10M: p96 6-pass, p64 4-pass; 90M winners) | win holds on seed 7 |
| P1-2 | Sampled route credit (§416): fitting cost flat in pool size at equal quality | pool 4/8 sampled within .01 bpc of linear at ≤ 60% of its fitting work |
| P1-3 | SHD official test (speaker-held-out selection first) | report accuracy against the cited 96.4% best |
| P1-4 | Update-matched 10M row against Transformer-256×4 (96 lanes) | 6-pass with 4,883 updates still < 1.908 |

## P2 progress

- **Hardware cost model v1 done (4 Oct):** `experiments/hardware_cost_model.py` → `results/diagnostics/hardware_cost_model_20261004.json`.
  Efficiency point: native p96 (1.888) moves 5.8× fewer bytes/char than Transformer-256×4 (1.908) and 1.9× fewer than LSTM-512
  (1.799, better quality), with cached-key winner-only execution (§414). Pool 2→32: 10.5× params for +5% bytes/char.
  (v1 of the model mis-charged key scoring; corrected same day, theory 154.) Next: a quality point at large pool (pool 8
  90M exists in rev. 3) so the flat-traffic capacity claim has a quality number beside it.

## P2 — after the P0/P1 numbers exist

Hardware cost model for the target asynchronous ASIC (event, message-rate, memory-locality/traffic and FLOP counts on one fixed trained model, vs the tuned dense reference; see investment/HARDWARE_THESIS.md;
CPU wall-clock is not evidence since the hardware is simulated); multi-modal or TTT demonstrations;
larger-than-90M scale.

## Stop or defer now

- New theory notes, primitives or diagnostics that do not feed a P0/P1 row (record ideas in a backlog, do not run them).
- DVS large program and the public UCR campaign repeats (results stand as reported).
- Clock-noise micro-diagnostics, unless they block P0-4.
- Investor-deck edits beyond inserting P0/P1 numbers once they are completed.

## Reporting rule

When a P0/P1 run completes: (1) one line in the scoreboard — win, loss or efficiency point with the numbers; (2) commit the
result file; (3) queue the confirming seed if it is a win. No paragraph of caveats: one scope sentence (data, seeds,
compute convention). Losses are reported just as plainly.
