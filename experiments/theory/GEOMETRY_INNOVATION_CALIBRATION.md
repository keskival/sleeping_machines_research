# Local credit must learn target innovations,not the mean

16 October 2026. This revision does not modify active three-arm sources.

## Completed conditioning evidence

The frozen native decoder control reduces root DEV reconstruction to0.203460,
but a per-level constant fitted on TRAIN achieves0.052045. Root reconstruction
skill1-L/L_constant is-2.9093. Most higher-level frozen-decoder skills hover
near zero despite the large reduction from untrained heads. That decrease is
therefore not evidence of newly learned representations. The current joint
and continued-native arms still determine the actual quality comparison.

Zero-frequency geometry channels correctly encode counts, but constant root
counts contribute a large nonzero mean. Random decoder outputs have to learn
that mean before resolving low-variance geometry. At the root,TRAIN target
variance is0.049281,versus an initial frozen decoder DEV loss0.835437. Raw loss
is thus badly calibrated as a measure of geometric information learned.

## Fixed TRAIN-only innovation target

For each level fit mu_l=E_TRAIN[q_l] and sigma_l=sqrt(E_TRAIN||q_l-mu_l||²/d),
with the existing equal-scene/equal-object weighting. Use

    z_l=(q_l-mu_l)/sigma_l.

Now the zero predictor has unit TRAIN error for nondegenerate levels, and
constant signed counts do not become the dominant representation-learning job.
This is scalar per-level normalization,not per-channel whitening;zero/dead
channels are not inflated individually. A1e-12variance floor handles degenerate
levels. Freeze the statistics before acquisition;DEV/TEST never fit them.
Training-only decoder bias starts at zero and weights have small nonzero scale
0.02/sqrt(native width),giving live native derivatives without stacked zero
paths. Native inference,keys/values,races,memory and emitted views are unchanged.
The new class is separate and is not swapped into active fitting.

Centering does not destroy the exact teacher composition law. Substitute
q_child=sigma_child*z_child+mu_child into the existing role-rotation/count sum,
then subtract mu_parent and divide by sigma_parent. The centered law is affine
with known count/pose-dependent offsets. It is still a fixed observable target,
not a semantic or universal information guarantee.

## Separate decoder optimization from encoder information

Reuse the frozen arm's saved TRAIN/DEV root features. Fit the same linear
32-output geometry decoder by closed-form ridge0.01 after TRAIN feature
standardization (std floor1e-6,matching the native binary-readout convention).
The unpenalized bias can fit the target mean immediately. Evaluate DEV once
at the fixed setting. This is an audit of the existing auxiliary decoder,not
new native acquisition,an external competing architecture or a tuned reference.
Compare it with the64-update Adam decoder and constant predictor. A large gap
identifies decoder conditioning;failure against the constant at this ridge is
a diagnostic of readable information,not a family impossibility theorem.

The preprocessing job saves TRAIN-only means/variances/directions with source
and input hashes. Contracts check ridge stationarity,bias-only recovery,unit
TRAIN innovation loss,zero-variance handling,native forward parity and live
all-level gradients. No centered quality fit is admitted automatically. First
review the completed joint/native/frozen comparison and this audit,then run a
real resource/update gate if normalization is the selected repair.
