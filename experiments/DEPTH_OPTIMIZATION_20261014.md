# R1 useful depth: diagnose, repair, win

Depth should add transformations that retain useful incoming information, integrate persistent memory and earn their discovery, teaching and update work. More stages are valuable only when their trained composition predicts better. Nonzero gradients alone answer none of those questions.

## Analytical diagnosis

For a frozen route sequence, write each stage as h[d+1]=F[d](h[d],m[d],t[d]). Content perturbations propagate through J[D-1]...J[0], with J[d]=partial F[d]/partial h[d]. Products can contract useful directions, amplify noise or become ill-conditioned even when every stage has nonzero gradients. Race changes add discontinuities; this conditional Jacobian does not characterize global route learning. Memory adds temporal paths whose credit can be truncated independently of numerical persistence.

A residual stage h'=h+alpha*f(h,m,t) has J=I+alpha*A on a frozen smooth branch. Its norm is at most1+|alpha|*||A|| and minimum singular value at least1-|alpha|*||A||. When |alpha|*||A||<1, products have corresponding multiplicative upper/lower bounds. This gives a conditioning design target, not a proof that our native stage is an identity residual or that arbitrary gates preserve clocks. Time transport, key/value addressing and old prediction must be contracted if a preservation repair is introduced. The packed core presently stores one uniform unit gain; per-stage gains require a new representation rather than silently changing frozen code.

Distinguish four failures: poor forward transport; delayed/weak consequence credit; insufficient acquisition; inefficient execution. Current D8 improves zero-feature offsets0.010855 but loses feature-conditioned gain0.125952, net0.115097 versus basicD2. That directs us toward feature paths/learning; it does not identify which stage causes the deficit.

## Empirical sequence

1. Existing D8 final checkpoint: stage message cuts with identical8192reusedDEV/1024history protocol, exact baseline replay, stage input/output/residual RMS and per-cut loss change. Queue aws_native_d8_stage_paths_20261014_v1 in guardedslot3 after existing work; no new training. Zeroing messages shifts distribution and is a necessity diagnostic, not a retrained-depth comparison.
2. Finish admitted D16/8M and D8/32M, and banked/localD8 acquisitions. Compare against both completedD2quality/timefrontier points. Preserve source-bound evidence.
3. New continuous-lane D8/256credit program tests removal of short consequence and reset restrictions. Changed layout/horizon recorded. Completed useful quality and full work decide adoption.
4. Based on stage results, build a prediction-preserving depth extension or stage-conditioning repair. Retain actual final-stage output, races, temporal transport/private memory/key-value/counterfactual work. Require old prediction/gradient/state/clock contracts and measured integrated resource smoke before an acquisition. Compare with ordinary continued parent fitting, because additional training already explains most Taxi successor gain.
5. Native bank fragments and shared long-horizon critic address learning cost and cross-update consequences after version-safe integration. Measure useful later prediction per complete paid work, not critic MSE alone.

Optimize quality at declared acquisition and total work to quality. “Optimal” requires choosing an operating objective; report quality/time/work frontiers rather than an unsupported universal optimal depth. No new Transformer fitting, no public validation tuning, no extra host slot.
