# B10 AWS: equal-compute LSTM, Transformer and native market models

Owner: AWS engineering queue. Authorized by the founder on 14 October 2026.
Decision: does the integrated native market model improve next-event likelihood
against information-matched LSTM and Transformer models at equal fitting work?
This is a B10-specific exception to the prohibition on new external-model fits;
it does not authorize language controls or a general baseline grid.

## Queue and admission

Work order: `experiments/queue/aws_b10_matched_controls_20261014_v1/work_order.json`.
Status: queued for implementation and accounting gates; no training job admitted
or running by creation of this protocol. The existing language acquisitions and
Curie D16→B10 growth pipeline finish untouched. AWS takes this at the next free
engineering opportunity; admit its validated contracts/resource smoke first,
then immutable one-job addenda through the six-slot coordinator. No shell queue
may point at an unimplemented driver. No dependency on a Curie model checkpoint.

## One exact forecasting problem

Reuse B10_MARKET.md: Aug1–24 TRAIN, Aug25–31 DEV, September TEST sealed.
Use the exact 801 TRAIN and 756 DEV windows, 1024 events/window, stride20/10,
respectively; score positions128..1023 (677376 DEV targets). Train on the same
1023 next-event targets per TRAIN window as the saved native driver. Inputs are
past six-class marks and physical timestamp gaps only. Do not give any arm prices,
quantities, extra history, a prior-day evidence bank or target-dependent features.
Reset numerical state at each window; preserve chronological order inside it.
Float64, CPU, one thread, same AWS hardware/software and minibatch size16.

The portable TRAIN/DEV artifact can supply these windows after hash and exact
source-window parity checks. Export/loading costs are recorded. No model
checkpoint is required for a fresh model. Save new checkpoints for recovery;
resume must reproduce the next loss, update, optimizer and RNG state.

Every model predicts the same normalized joint law:
log[S(g)-S(g+1e-6)] + log pi_k(g+0.5e-6).
Use the existing native race clock/mark output head for all three encoders,
with identical number of clock components, TRAIN-only initialization, stable
interval evaluation and zero-gap treatment. LSTM and Transformer use this common
head as temporal-point-process references, not a claim about every possible
LSTM/Transformer density head. Report encoder and head parameter/work counts.

LSTM: two layers, width32, mark embedding32; causal elapsed-time input
log1p(g / TRAIN-median-positive-gap). Transformer: two layers, width32,
four heads, feed-forward width128, causal attention, mark embedding plus past
elapsed-time encoding and within-window position; no dropout. Timestamp features
at position i may use t_i, never t_(i+1). Pin the precise encoding before any fit.
Native: existing two-stage width32 Race member (18164 parameters), retaining
its temporal memories and clock races. These are fixed initial configurations,
not parameter-matched competitors. Later eight-stage results remain separately
scoped until the same resource protocol includes their complete acquisition.

## Equal compute, not equal epochs

Primary resource boundary is complete fitting arithmetic: encoder, common head,
backward, optimizer, initialization/calibration and every candidate/smoke/teacher
fit. Report multiply/add FLOPs, special-function evaluations and comparisons in
separate columns with one common operator ledger. Reject unsupported operators;
never apply a blanket backward multiplier only to one architecture. Include
attention scoring, losing routes/values, native memory and any audit work.

Before quality fitting, derive and validate the native two-stage budget B from
30 complete TRAIN epochs at batch16 with the saved training-target protocol.
Record B numerically in a source-bound accounting artifact; the old 5750.7s
Curie wall time is not an AWS FLOP budget. Validate the ledger on representative
forward/backward/AdamW batches of all three models, including tied timestamps.
No quality selection during calibration. Cap every family/seed at the same B;
stop before a batch would exceed it. Report unused budget; require <=1% slack
or reduce the common budget to a feasible matched boundary. Validation arithmetic
is measured separately and included in a second total-development-work table.
A claim about matched total development work requires that second boundary too.

Seeds0,1,2 are fixed. Give each family/seed two learning-rate candidates
(0.001 and0.003) with 10% of B each. Select on full DEV NLL, then continue that
candidate for the remaining80% of B, retaining optimizer/RNG; charge the discarded
candidate. Use AdamW, weight decay1e-4, global gradient clip1.0 for all arms.
Score at fixed candidate endpoints and 25/50/75/100% of continuation; charge
those evaluations separately. Select final checkpoint by DEV NLL. No patience
shortcut or uncharged retry. Smokes consume the corresponding family's budget;
no architecture gets hidden prior acquisition or learned guidance for free.

Also record process CPU seconds, wall seconds, peak RSS, acquired targets/passes,
and fitting/inference work per target. Equal-FLOP fits are not equal CPU-time
fits. If accounting cannot be validated, halt arithmetic-matched admission;
a separately labelled equal-CPU-seconds protocol requires an explicit recorded
amendment before fitting, not a post-result change of denominator.

## Gates and execution

1. Validate portable TRAIN/DEV identities against original B10 windows, all day
   boundaries and scored target IDs; reject September data. Record source hashes.
2. Contract: future marks/timestamps cannot change earlier scores or gradients;
   zero-gap scores and gradients finite; normalized cell probability and mark law;
   common-head parity from identical hidden features; chunk/reset semantics;
   exact saved-native score replay (6.882244527423017 NLL within1e-8).
3. Validate full training ledger, batch-budget stop and checkpoint-resume parity;
   representative full1024-length update smoke for each model, measured RSS/CPU.
4. Freeze numerical B, driver hashes, initialized configurations, output paths,
   hardware and seed list. Each resulting run names B10 and its decision.
5. AWS host preflight: inspect current manifest/addenda/results, running jobs,
   memory and GPU occupancy. CPU only. Use run_safe_aws6 through the six-slot
   coordinator, pinned physical-core slots, <=3000000KiB RSS/job and >=8GiB host
   MemAvailable. Choose timeout from measured throughput and charged B with
   evaluation headroom. Long coordinator in tmux; never bypass host/slot locks.
6. Admit unique per-model/per-seed jobs, without displacing active acquisitions.
   Serialize publication and preserve all results, logs, checkpoints and failures.

## Evidence and investor publication

Report all nine family/seed endpoints, mean/sampleSD, per-day and per-mark NLL,
time/mark decomposition, tied/positive-gap bins and complete resource tables.
Paired day analysis uses shared days, not windows treated as independent days.
Development selection on DEV is explicit; three seeds are not nine independent
market samples. No benchmark win on sealed TEST from these DEV fits.

Retain three separate evidence cohorts: old E120/E123 prefix screen
(native3.823226 vs Transformer4.207972, unequal evidence-bank inputs); full B10
native6.882245 vs Hawkes9.117119; and online-credit16384-event NLL5.737391.
Do not compare their absolute NLLs or replace one with another. Add the completed
matched B10 table to the investor evidence PDF with parameters, total fitting work,
quality and a single scope statement. Report wins and losses plainly. The frozen
B10 seven-day/three-seed/volatility rule remains unchanged; September evaluation
requires the existing sealed-test admission after development selection. No P&L
claim follows from likelihood alone.
