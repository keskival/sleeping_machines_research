# Tuned dense references at the native compute budgets (queued 4 October 2026, 21:00 UTC, user-directed)

## Why

The saved E64 references use one fixed learning rate (Adam 1e-3 Transformer / 2e-3 LSTM), no warmup, seed 0. Their
sizes and pass counts were chosen before the native budgets existed. Several current matched-compute wins compare
against a reference that spends its budget differently: for example, the 1-pass Transformer-256×2 at 2.427 bpc is
undertrained. A skeptical reviewer will retrain a dense model *at our budget* with a small learning-rate/shape choice
and compare against that. These runs do that first, so a win survives review. If they beat us, that is a loss, and we
report it plainly (WIN_CRITERIA.md).

## Protocol (unchanged except the listed knobs)

- Script: `experiments/e64_lm_baselines.py`, run via `aws_benchmark.py` (isolated output per run tag). New options
  `--lr`, `--warmup` (linear warmup then cosine), `--weight_decay`, `--seed`; defaults reproduce the saved runs exactly.
- Same data: text8[0:D], validation text8[90M:90.2M] (best checkpoint by validation), test text8[95M:96M]. Transformers
  use reset T256 windows with T/2 context, matching native `--eval-segment 256`. **Protocol correction:** E64 LSTMs
  retain state across the stream and score 999,999 test positions; native/Transformer score 999,936. The original
  statement that all rows had the same windows was incorrect. Keep existing scores and queues unchanged; rescore
  saved LSTM weights on the native windows before claiming identical context/targets. For selecting among tuned
  arms with different context policies, align validation scoring too, and never select by test.
- Dropout 0.1, warmup 200 steps (10M) / 500 steps (90M), batch 32 × 256.
- **Budget rule:** each run's `training_flops_estimate.total_training_flops` (shape_based_v1, the convention used for every
  saved reference) must be ≤ the native row's traced fitting compute. This was checked when queueing.
- **Selection:** the tuned reference for a budget is the arm with the best **validation** bpc among that budget's arms.
  Its test bpc goes in the scoreboard. Selecting the best arm by test would be a protocol error.

## Queues (one job each; run tag = queue name)

Budget A: ≤ 352.1 TF (native p96/d4 6-pass, 1.888). Budget B: ≤ 107.2 TF (native p64/d4 4-pass, 1.955).

| Queue | Model | Params | Passes | LR | Est. training |
|---|---|---|---|---|---|
| `aws_tuned_ref_10M_A_tf256L4_p1.5_lr0.001_s0_20261004T210000Z` | tf 256×4 | 3.24M | 1.5 | 0.001 | 333 TF |
| `aws_tuned_ref_10M_A_tf256L4_p1.5_lr0.002_s0_20261004T210000Z` | tf 256×4 | 3.24M | 1.5 | 0.002 | 333 TF |
| `aws_tuned_ref_10M_A_tf192L4_p2.5_lr0.002_s0_20261004T210000Z` | tf 192×4 | 1.84M | 2.5 | 0.002 | 328 TF |
| `aws_tuned_ref_10M_A_tf128L4_p5.4_lr0.003_s0_20261004T210000Z` | tf 128×4 | 0.83M | 5.4 | 0.003 | 347 TF |
| `aws_tuned_ref_10M_A_lstm512_p4.5_lr0.002_s0_20261004T210000Z` | lstm 512 | 1.20M | 4.5 | 0.002 | 324 TF |
| `aws_tuned_ref_10M_A_lstm384_p6_lr0.003_s0_20261004T210000Z` | lstm 384 | 0.70M | 6 | 0.003 | 254 TF |
| `aws_tuned_ref_10M_B_tf192L3_p1_lr0.002_s0_20261004T210000Z` | tf 192×3 | 1.39M | 1 | 0.002 | 99 TF |
| `aws_tuned_ref_10M_B_tf128L4_p1.6_lr0.003_s0_20261004T210000Z` | tf 128×4 | 0.83M | 1.6 | 0.003 | 103 TF |
| `aws_tuned_ref_10M_B_lstm384_p2.5_lr0.003_s0_20261004T210000Z` | lstm 384 | 0.70M | 2.5 | 0.003 | 106 TF |
| `aws_tuned_ref_10M_B_lstm256_p5_lr0.003_s0_20261004T210000Z` | lstm 256 | 0.34M | 5 | 0.003 | 102 TF |

90M budgets, matching the pending native rev. 4 arms. C: ≤ 0.96 PF (p64/d4 4-pass); D: ≤ 2.1 PF (p96/d4 4-pass).

| Queue | Model | Params | Passes | LR | Est. training |
|---|---|---|---|---|---|
| `aws_tuned_ref_90M_C_tf192L4_p0.8_lr0.002_s0_20261004T210000Z` | tf 192×4 | 1.84M | 0.8 | 0.002 | 0.95 PF |
| `aws_tuned_ref_90M_C_lstm512_p1.4_lr0.002_s0_20261004T210000Z` | lstm 512 | 1.20M | 1.4 | 0.002 | 0.91 PF |
| `aws_tuned_ref_90M_D_tf256L4_p1_lr0.001_s0_20261004T210000Z` | tf 256×4 | 3.24M | 1 | 0.001 | 2.00 PF |
| `aws_tuned_ref_90M_D_tf192L4_p1.75_lr0.002_s0_20261004T210000Z` | tf 192×4 | 1.84M | 1.75 | 0.002 | 2.07 PF |

Also queued: `aws_language_10M_r1_p64d4_4pass_linear_s7_20261004T210000Z`, the P1-1 second seed of the 10M p64/d4 4-pass
win (identical settings, `--seed 7`). The p96/d4 6-pass seed 7 is already queued on curie (`curie_language_v6_..._s7`).

## Resources (from the saved runs on this AWS host, one thread)

Transformer-256×4 ran at about 3.2K tokens/s and LSTM-512 at about 14K tokens/s. Budget-A Transformer arms take about
1.5–3 h, and the 90M D arms about 8 h. Suggested settings: 10M `JOB_TIMEOUT_S=28800 MEM_CAP_RSS_KB=4000000`; 90M
`JOB_TIMEOUT_S=64800 MEM_CAP_RSS_KB=6000000` (the 90M Transformer needed the rss6g cap). Keep the 8 GiB floor.

## Reporting

Add scoreboard rows "10M vs tuned dense at ≤ budget A/B" (and 90M C/D once the native arms complete): win, loss or
efficiency point. Keep the existing rows against the saved references, labelled "saved reference". If a tuned
reference beats the native model, the corresponding earlier win is still historical evidence against that saved
reference, but it is no longer the headline claim. Say so beside it.

## Efficiency metric

Wall time is not an evidence axis: the target asynchronous hardware does not exist yet, and all native runs are
simulated on CPU. Compare counted work (fitting FLOPs, inference FLOPs/position, and where available state/memory
traffic) under the stated conventions. Do not report CPU wall-clock as an efficiency result for either side.

## Independent queue audit (4 October, review workspace)

All 14 queued arms fit their declared budgets under the saved shape estimate; A/B use completed native budgets,
C/D use projected four-pass native work and must be rechecked at completion. Five arms use continuous-state LSTM
scoring and need the context correction above. [Audit record](results/diagnostics/tuned_reference_budgets_stdlib_20261004T213500Z.json).
The first source-bound, inference-only LSTM-512/10M rescore is [prepared, unrun](queue/aws_lstm512_native_windows_20261004T213000Z/README.md).
It does not consume a training slot here or displace the active P0 owners. [Protocol audit](HEADLINE_PROTOCOL_AUDIT.md).
The scoreboard now waits for all six A / four B arms with completed provenance and checked actual budgets, then
requires aligned validation/test contexts. A first finished arm cannot become the headline tuned reference.

## Completed arms

| Arm | Budget | Est. training | Best valid | Test | vs native in budget |
|---|---|---|---|---|---|
| `aws_tuned_ref_10M_A_lstm384_p6_lr0.003_s0_20261004T210000Z` | A | 254 TF | 1.778 | **1.840** | native p96/d4 6-pass 1.888 at 352 TF: **loss** for native (context resolved: streamed 1.8255 = windows 1.8255, curie 23003f5d) |
| `aws_tuned_ref_10M_A_lstm512_p4.5_lr0.002_s0_20261004T210000Z` | A | 324 TF | **1.769** (leading) | **1.826** | **loss** for native by .062; nearly the saved 6-pass LSTM-512 (1.799) at 75% of its compute |
| `aws_tuned_ref_10M_A_tf192L4_p2.5_lr0.002_s0_20261004T210000Z` | A | 328 TF | 1.982 | **2.027** | native better by .139, but native used 352 TF (107%): not a formal matched-compute win; still improving at its last step |
| `aws_tuned_ref_10M_A_tf128L4_p5.4_lr0.003_s0_20261004T210000Z` | A | 347 TF | 1.950 (best TF so far) | **1.996** | native better by .108; native used 352 TF (101%) |
| `aws_tuned_ref_10M_A_tf256L4_p1.5_lr0.002_s0_20261004T210000Z` | A | 333 TF | 2.011 | **2.053** | native better by .165; native used 106% |
| `aws_tuned_ref_10M_B_lstm384_p2.5_lr0.003_s0_20261004T210000Z` | B | 106 TF | 1.852 | **1.915** | native p64/d4 4-pass 1.955 at 107 TF: **loss** for native by .040 |
| `aws_tuned_ref_10M_A_tf256L4_p1.5_lr0.001_s0_20261004T210000Z` | A | 333 TF | 2.175 | **2.199** | native better by .311 (lr .001 too low at 1.5 passes) |
| `aws_tuned_ref_10M_B_tf192L3_p1_lr0.002_s0_20261004T210000Z` | B | 99 TF | 2.303 | **2.315** | native better by .360 (native used 108%) |
| `aws_tuned_ref_10M_B_lstm256_p5_lr0.003_s0_20261004T210000Z` | B | 102 TF | 1.872 | **1.934** | **loss** for native by .021 |
| `aws_tuned_ref_10M_B_tf128L4_p1.6_lr0.003_s0_20261004T210000Z` | B | 103 TF | 2.191 | **2.215** | native 1.955: better by 0.260 |

The validation curve was still improving at the final step (1.785 → 1.778 over the last 1,500 steps), so this budget is
not saturated for the LSTM. Group selection waits for all six A arms (HEADLINE_PROTOCOL_AUDIT.md). Whatever is selected
can only be at least this strong on validation. Fairness: native configurations were not learning-rate tuned. Matching
native lr arms are queued (`curie_language_native_lr_*_20261005T000500Z`).

**Inference view (5 Oct, two arms):** per-character inference is about 2 × params: LSTM-384 ≈ 1.4 MFLOPs, LSTM-512 ≈ 2.4.
Native p96 winner-only is 1.3 MFLOPs. At roughly equal inference compute, LSTM-384 is .048 bpc better. On 10M
characters, tuned LSTMs currently dominate the native model in quality at similar inference cost. The native model's
remaining measured advantages are bytes per character (no KV cache; capacity added for +5% traffic) and the Transformer
comparisons, pending the tuned Transformer arms.

**Context and mechanism (5 Oct 01:40, curie 23003f5d / 1b4fa655):** scoring context is irrelevant here (streamed = windowed
for both LSTM and native). The LSTM's 0.06–0.07 bpc lead is already present at 2–4 characters of history and stays
constant after that. It is local character modelling, not longer memory. Deterministic routing (1.8884) equals sampled routing
(1.8889), so race noise is not the lever. The native gap is in short-range modelling capacity.

**Transformer arms, 2 of 4 (5 Oct 02:30):** the best tuned Transformer so far (tf128x4, 5.4 passes) scores 1.996 against native 1.888 at
essentially equal training compute (347 vs 352 TF). Tuning narrowed the gap to the saved 4-pass TF256x4 (1.908 at 889 TF)
only partly at this budget: on 10M characters the shape-estimated Transformer pays heavily for T256 attention at small width.
If the two TF256x4 arms do not beat 1.888, the Transformer win survives tuning. The LSTM loss also stands.
Checked objection, "T256 attention handicaps the small Transformers": with the shape estimator, a ctx-128 TF256x4 costs 206 TF per
10M-character pass versus 222 TF at ctx 256 (7% less). The projection/FFN term dominates at these widths, so a shorter
training context would buy under 0.1 extra pass. No additional arm queued.

## Final verdict (5 Oct 04:50 UTC; scoreboard generated from result files)

Validation selection was confirmed with windowed LSTM validation (curie aac6d082; equals carried-state).
`report/tuned_reference_admission.py` now accepts a checkpoint-hash-bound windowed rescore.
- Budget A (≤ 352 TF): tuned LSTM-512 4.5p **1.825** vs native 1.888: **loss**. Best tuned Transformer (tf128×4 5.4p) 1.996:
  native better at 102% of its compute (not a formal matched win).
- Budget B (≤ 107 TF): tuned LSTM-384 2.5p **1.915** vs native 1.955: **loss**. Best tuned Transformer (tf128×4 1.6p) 2.215:
  native better at 104% of its compute.
- P0-2 (native p128/d4 4-pass, 1.907 at 411 TF) is also a loss. At 10M the native model sits between tuned Transformers and
  tuned LSTMs. The gap to LSTMs is local (2–4 character) modelling (FINDINGS). Queued levers: content taps, weight decay,
  native lr.
