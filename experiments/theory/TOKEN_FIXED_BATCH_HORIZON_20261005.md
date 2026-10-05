# Independent credit window and optimizer batch

Frozen trained2K actual-write audit: at sites(0,0,0),(7,0,0),(15,0,0),(7,1,1),
81.7–97.1%of absolute per-position loss perturbation over128positions lies
after position15. Alternative utility signs can change beyond the window.
This is consequence coverage on training lanes, not missing-gradient fraction
or held-out quality. EOS122on one lane is retained. Boundary-only v1 and
broader v2 files/producers are preserved.

The16-token learner keeps numeric memory but stops continuous state credit
and future-choice utility at its update boundary. Increasing chunk length also
reduces AdamW update count, confounding a horizon comparison. Separate the
optimizer batch from the credit window.

`horizon_token_language_engine.py` processes a fixed optimizer batch in credit
segments. It detaches carried state between segments without resetting facts;
readout loss spans the entire optimizer batch, with one update. The uniform
teacher samples sites over the whole batch, then scores only causal positions
up to the selected segment boundary. Its importance factor is

    inverse_site_probability * (credit_end - token) / optimizer_batch_length.

This estimates the sum of conditional choice contributions within the chosen
credit boundary. Full-batch credit removes those internal detaches and extends
choice utility to batch end. First-site/local control remains available.
Read-old/write-new causality, races, separate keys/values, timestamps, numeric
memory and observed queries are retained. Credit is still finite and declared.
Within reset-mode diagnostics, reset occurs once at the optimizer batch start,
not at every credit segment. EOS resets remain event-dependent per lane.

Inference features and assigned RNG must match across partitions. Learning
gradients intentionally differ: a final-target loss can reach initial memory
only when its credit path crosses the intervening boundary. Model/optimizer
update count, input presentations and evaluation cadence are held fixed in the
paired horizon study. Longer graphs increase activation memory; short-window
key rebuilds and actual replay also cost work. Charge both rather than assuming
equal runtime. The existing immutable AWS engines remain unchanged.

Required contracts: identical numeric features/state/RNG across windows,
explicit gradient cut versus retained path, actual resume inside a multi-window
optimizer batch, and complete resource/quality comparison before promotion.
Use matched64-token optimizer batches with16versus64credit windows, then choose
the next horizon by measured consequence coverage and trained quality.

## Completed contracts and queued comparison

Numerical v2 completed18:51:13 UTC,2tests passed: numeric features/state/RNG
match, and short-window initial-memory gradient is zero while full-window
gradient is nonzero. The failed v1 assertion expected None instead of zero;
its test producer is archived and its log/queue retained. The numerical
learning contract is zero credit, not a particular autograd representation.
Actual4internal windows per optimizer batch interruption/resume completed
18:51:59 with exact model/AdamW/state/all RNGs/quality/192targets parity.

New immutable packet `addenda/aws_horizon_tokens_20261005T185300Z.json` queues
the paired8K study: batch64,credit16or64,32updates,16,368targets, evaluation
every8updates, same initial/decoder/core/seed. Both include initialization in
selection and publish selected/final checkpoint artifacts. Graph/replay cost
can differ; RSS cap1,200,000KiB, VMS6,000,000KiB, timeout300s, host reserve8GiB.
These are pending fits, not evidence that longer credit improves quality.
