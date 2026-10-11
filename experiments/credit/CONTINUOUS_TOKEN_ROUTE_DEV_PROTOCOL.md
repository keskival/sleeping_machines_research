# R1 continuous route policy v1: fixed development comparison

Registered before fitting on11 October2026. Decision: does reference-forward
credit produce a better route policy than zero/static credit on fresh prefixes?
Use available two-layer4,194,304-token source checkpoint without changes.
All new data use properly tokenized FineWeb TRAIN-tail; public scoring untouched.

Training: seed98410, one64-token context and64 consecutive eight-target windows,
offset4,194,304+32,768. RouteController width64/29,057 parameters; Adam1e-4;
lexical Adam.01;0.9softmax+0.1uniform; signed native-write amplitude.25.
Three arms zero/current-snapshot static/reference, same initialization and
sampling seed98410+1000+window. Reference no-op independent of factual choice.
No clipping, checkpoint selection or tuning: final64 updates only.

Scoring:8 private held-out prefixes at4,194,304+65,536+256*i,64 context/eight
future targets each. Evaluate each issued route with an independent exact
lexical Adam learner; fresh key/lexical episode state. Score issued stochastic
expected NLL exactly and greedy separately. Actors see prefix only. All costs
shared across actors; evaluate initial actor, uniform/no-op and exhaustive best
route to diagnose headroom. DEV labels never enter training.

Primary engineering comparison: mean reference-over-zero expected NLL gain,
paired10,000-draw prefix bootstrap95%. Additional reference-over-static isolates
benefit of reference adaptation. Positive lower bound supports this single-seed
private-prefix endpoint; eight prefixes do not establish dataset generalization.
Negative/unclear results diagnose this variant, not the family. Analyze headroom,
policy changes, gradient scales and train/evaluation state mismatch before a
new registered development iteration.

Smoke: same code, four TRAIN windows/two private prefixes, no selection.
Preserve results; full fit may reuse those private prefixes only as disclosed
nonsealed development observations. Memory cap1.5GB RSS, one CPU thread, at least
9GiB available reserve,600s timeout through unique run_safe queue. All fitting,
exhaustive scoring, copies, vocabulary normalization and discovery counted.
