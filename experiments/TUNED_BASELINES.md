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
| `aws_tuned_ref_10M_A_lstm384_p6_lr0.003_s0_20261004T210000Z` | A | 254 TF | 1.778 | **1.840** | native p96/d4 6-pass 1.888 at 352 TF: **loss** for native (stateful LSTM context; aligned rescore pending) |

The validation curve was still improving at the final step (1.785 → 1.778 over the last 1,500 steps), so this budget is
not saturated for the LSTM. Group selection waits for all six A arms (HEADLINE_PROTOCOL_AUDIT.md). Whatever is selected
can only be at least this strong on validation. Fairness: native configurations were not learning-rate tuned. Matching
native lr arms are queued (`curie_language_native_lr_*_20261005T000500Z`).
