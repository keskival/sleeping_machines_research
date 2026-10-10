# B4 numerical learning stability

The frozen Github guarded runs skip the same long TRAIN batch: the loss is
finite but the gradient norm is NaN. Recovering those updates is the first
development decision. Frozen benchmark scores and checkpoints remain evidence;
this pipeline reads TRAIN only and does not rescore TEST.

First isolate the clock-head backward on the longest original TRAIN batch at
the saved validation-selected checkpoint, with dropout disabled. Anomaly
detection identifies the first invalid backward operation. This is a diagnostic,
not a fit or a benchmark result. A passing isolated head would require a
full-chain reproduction before selecting a repair.

Candidate numerical repairs must preserve the represented real-valued
likelihood, temporal computation, persistent memory, clock races, addressed
mark state and silence compensation. Log-domain algebra can avoid overflowing
intermediate values whose final likelihood remains finite; rate clipping would
change the model and requires a separate development rationale. No such repair
is adopted on static inspection alone.

Admission gates for a repair: reproduce the failure; compare finite-region
values and gradients against the original; recover finite gradients on the
failing TRAIN batch; measure execution and memory; then conduct an explicitly
labelled TRAIN/DEV development fit. Charge the full backward work. No new
external architecture or sealed test score is part of this pipeline.

Initial source-bound job: `aws_b4_backward_diagnosis_20261010T1958Z`, AWS slot 3,
900-second timeout, 4 GB RSS watchdog, existing scheduler memory floor.

Completed diagnosis: the original isolated head has finite loss -3.74237 but
11,550 infinite sigma values. Autograd identifies ExpBackward0 at v19 line177,
the recording-cell spread floor exp(-mu), with minimum mu -1440.83. This is
an intermediate overflow, not evidence that the recording-cell likelihood
itself must be non-finite. Result: aws_b4_backward_diagnosis_20261010T1958Z.

Repair stores log spread = logaddexp(log softplus(raw), max(log(c)-mu,
log(.005))); raw Gaussian terms use exp(-log spread) and log spread directly.
Exponential survival uses exp(log rate + log weight). Logistic edges are also
represented by their logarithms. The log-softplus negative tail below -30 uses
the asymptote x (relative error below 5e-14); no delay/rate/gradient clipping
is introduced. Existing parameters and architecture remain unchanged.

The first window-contract fixture used invalid negative endpoints; its failed
job/log stays recorded. Corrected positive endpoints run under
aws_b4_logspread_check_v2_20261010T2002Z. Completed: maximum finite-region value
error 9.09e-13, normalized gradient error 1.64e-16; head and full-chain backward
both finite on all 9,114 TRAIN events. Full loss -3.75014, 116.42 seconds with
anomaly detection enabled, peak RSS 2,948,080 KB. Diagnostic time is not an
inference or optimized training throughput measurement.

Completed five-pass pilot `aws_b4_logspread_pilot_20261010T2003Z`: **zero
skipped updates**,225.30s, DEV sequence loss -76.7246 (time -166.5078,
marks89.7832), best epoch4. Source-bound update decision passes. This is
first-pass numerical development, not a benchmark win or a family verdict.

The independent padding audit (`diag_b4_padding_nan.py`) finds the overflow
at masked tail positions, where fake events keep accumulating addressed state.
Single sequences remain finite. The guarded original skips the entire padded
batch, not just one long sequence: Github splits1–3 lose59–68% of TRAIN events
in55–60 epochs. Frozen verdicts already carry this correction. The log-spread
repair recovers the pilot updates without changing valid-event computation.
Explicitly stopping padded state writes is a further execution repair requiring
score/gradient contracts; it is not silently included in this fit.

Next admitted on AWS slot2: `aws_b4_logspread_develop_20261010T2112Z`, after
existing admitted work, fixed Github split1/seed0 TRAIN/DEV. Original d32,
2-layer model and AdamW0.003,100-pass ceiling and original30-pass DEV patience;
no parameter grid, TEST interface or frozen-score replacement. Five-pass
source-bound zero-skip gate is required at entry. Any further nonfinite update
stops the fit for diagnosis. Save parameter/Adam/RNG and batch-order checkpoints
after each pass. Record data/source hashes, training passes/targets, fitting
and evaluation time, RSS and selected-checkpoint DEV forward latency. Full
fitting FLOPs are unmeasured and cannot support an efficiency claim.

Queue timeout10,800s,4GB RSS watchdog,8GB VMS and scheduler host floor8GiB;
measured diagnostic full backward peak2.95GB. Model wall budget7,200s checked
between passes. No active source or existing job was replaced.
