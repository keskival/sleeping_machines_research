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

Admitted slot3 pilot aws_b4_logspread_pilot_20261010T2003Z: fresh seed0, original
settings, five TRAIN passes including the first original failure epoch; DEV
only. Full-gradient/source gate checked again in the driver. Decision: whether
the original problematic batch receives finite updates throughout the pilot.
The guarded driver records any other skipped updates. Quality is development
evidence, and the frozen benchmark result stays unchanged. 2,400-second queue
timeout, 4 GB RSS watchdog, 1,800-second between-epoch wall budget.
