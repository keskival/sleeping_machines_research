# A contained contextual decoder for the deepest temporal state

14 October 2026 · R1 prepared capability, native checkpoint contracts pending.

## Failure being addressed

The active native model predicts from its deepest selected48-dimensional event feature. Its top2000 token logits are affine in that feature; rare-token branches project it to24/12/6 dimensions. Saved8M error analysis puts70.7% of targets and52.5% of total loss in the top2000 group. Increasing rare-tail capacity alone cannot address most of that loss. These counts concern the saved D2 baseline, not a causal diagnosis of the running D8 member.

The completed temporally aligned96-dimensional interface improves long-history DEV6.160183→6.099337 at8M targets, after losing at512k and2M. Training7734.249→8657.820seconds (+11.94%). It is evidence that acquisition can change which interface is useful. It does not show that more linear dimensions, or direct earlier-layer access, is the right repair for the deepest temporal path.

The prepared candidate enriches the decoder of that deepest path rather than bypassing it. It addresses restricted contextual prediction; it preserves temporal computation, private addressed memory, separate keys/values, selected writes and counterfactual route learning. It supplies no information erased by the temporal state, and does not replace the native core with a dense carrier.

## Construction and containment

For deep feature z, the existing frequent-token/branch head is b+Bz. Replace its affine head by

    logits(z) = b+Bz + V tanh(Az+a),  V_initial=0.

All adaptive branch normalization and rare-tail heads are retained. Therefore initial normalized probabilities equal the learned predictor exactly. The feature cotangent and existing raw parameter gradients are retained at initialization. The added hidden-weight gradient is initially zero, but V generally receives a nonzero gradient; once V moves, the hidden features can learn. This is a known residual nonlinear head, not a new temporal primitive.

Construction preserves global actor RNG using an isolated CPU RNG scope. Existing base parameters remain the same objects, so their Adam moments can be retained while new parameters start with fresh state. This is not optimizer-step containment: extra V gradient coordinates change the global gradient norm and can change clipping of existing parameters from the first update. A fitting comparison must account for that and explicitly declare any optimizer restart or moment migration.

With width48, the lift adds98,496parameters to the current2003-output head. It evaluates the extra head whenever prediction needs that head and charges its optimizer work. Stored capacity, routing discovery and decoder work stay separate. No full-fit economy is inferred from the parameter increment.

## What the spectrum can and cannot tell us

For N contexts and C classes, center feature rows X and head rows B. Double-centered logits are X_c B_c^T; biases and the per-context softmax normalizer cancel. Their rank is at most min(N-1,C-1,P). The new logits add Phi_c V_c^T, permitting rank at most min(N-1,C-1,P+M), where M is the lift width. These are upper bounds on a probability interface, not a bound on intelligence or proof that its rank limits this task.

To measure nonzero singular values without an N-by-C logit matrix, take thin QR factorizations X_c=Q_x R_x and B_c=Q_b R_b. The nonzero singular values of X_c B_c^T equal those of R_x R_b^T because the Q columns are orthonormal. The implementation handles rectangular factors and constant contexts. It records energy-effective rank, direction energies, class/weight coordinates scanned and wall time on512frozen TRAIN contexts. Adaptive tail spectra are centered within each branch; branch-dependent normalizers cancel there, not across all vocabulary groups together.

The spectrum measures contextual prediction variation. It is not conditional information, an optimization response eigenvalue, a causal depth effect, or attainable language quality. Full available class weights are scanned during this diagnostic; avoiding a large intermediate matrix does not make that scan free.

## Execution decision

The current D8/8M→D16/8M→D8/32M program stays unchanged. Its pending native replay diagnostics measure per-branch spectra and test contained decoder loss/feature-cotangent/RNG parity on the actual checkpoint. They charge both readout evaluations. The separate longer-horizon route-credit acquisition remains the next admitted fit.

Use completed quality, spectrum, content-cut effects and learning-cost measurements to choose the subsequent decoder comparison. Do not select a width from this diagnostic or call high rank a diagnosis. A controlled fitting comparison must demonstrate lower loss and charge the expanded decoder, teacher, clipping and optimizer. Public validation remains sealed; investor claims follow completed evidence.

Sources: `results/diagnostics/aws_multidepth_8m_error_buckets_20261013_v1.json`, both source-bound8M result packets; implementation `sleeping_machines/contained_nonlinear_readout.py` and `sleeping_machines/contextual_logit_spectrum.py`.
