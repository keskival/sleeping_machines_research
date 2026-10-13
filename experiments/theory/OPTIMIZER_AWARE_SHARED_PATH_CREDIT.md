# Shared-path credit must be economical in update space

13 October 2026. B10 continuation after exact four-direction raw-gradient transport and actual one-event Adam parity. The saved actor, oscillator clocks, addressed memory and numerical inference are retained. This is an analytical design note, not a new quality result or claim of a novel constrained least-squares theorem.

A raw-gradient projection contract verifies the temporal information path. It does not show that the recovered information creates better updates. In particular, global gradient clipping couples every coordinate and Adam updates its second moment using the supplied gradient. Even an exactly recovered head gradient can produce a different head step when other coordinates differ.

Let g_hat be a baseline gradient, and y=D^T g be exact transported directional information, with D full-column-rank. A correction delta must satisfy D^T delta=y-D^T g_hat. If local update sensitivity is a fixed matrix P and update-distance is measured by positive-definite M, define W=P^T M P, assumed positive definite. The minimum update-distance correction is

    delta = W^-1 D (D^T W^-1 D)^-1 (y-D^T g_hat).

This follows by solving the Lagrange conditions for minimizing delta^T W delta subject to the directional constraint. The Euclidean raw-gradient repair currently implemented is the special case W=I and orthonormal D. Its exact projection proof remains valid, but it need not be the cheapest correction in actual update space.

A previous-step diagonal Adam scale can define an economical surrogate P. It is not the exact sensitivity of the next Adam step: new moments, clipping and the epsilon term matter. A singular or nearly singular metric requires a declared regularization; inversion must not silently amplify unsupported coordinates. The four-direction Gram solve and metric maintenance count as learning work. Validate the proposed rule by actual optimizer replay, not just this linear surrogate.

Direction selection should maximize future predictive improvement per complete cost, rather than maximize agreement with a historical BPTT gradient. At a window boundary let candidate transported directions expose context, shared-leg identities and native memory. After outcomes arrive, compare their actual update consequences using the declared delayed utility protocol. The native critic can learn which directions and residuals are worth activating from this causal evidence. Fixed teacher audits can check derivative fidelity; they are neither free nor the training objective by necessity.

Do not let current utility feedback change an already scored forecast, and do not change a sensitivity basis retroactively within a history window. Select from earlier information, retain a spanning bank, or reset sensitivities. Shared fragment representations permit transfer across compatible routes; branch context still distinguishes opposing effects of the same producer.

The current development experiment isolates the unlearned transport contribution before adding the critic: exact BPTT, event-local, own-leg traces and four-direction shared-path correction use common acquired actor/moments and actual clipped Adam. Report quality and full cost whether they improve or fail. A useful correction with a slow correctness backend warrants execution engineering; a fast inaccurate correction needs information/credit redesign. Neither outcome is a verdict on the full family.
