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
