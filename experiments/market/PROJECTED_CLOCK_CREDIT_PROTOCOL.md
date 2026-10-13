# B10 projected shared-path clock credit

13 October 2026. Decision: can bounded state sensitivities recover selected cross-layer history credit while preserving the saved native market actor's forward computation?

Reuse the saved 18,164-parameter model and existing TRAIN recording-cell likelihood protocol. No fit or DEV/TEST selection. Warm 128 events from August 24; score 1, 8, 32 and 128 subsequent targets. All parameters stay fixed inside each window, matching the detached warm-state full-BPTT reference.

Four orthonormal parameter directions cover each layer's mean log-decay and mean frequency. Carry the complete 88-coordinate state sensitivity in these four directions through the actual gated oscillator, read map, LayerNorm, exact GELU, upper-layer writes and addressed mark memory. The analytic observer uses NumPy and preserves the original PyTorch numerical forward. It is a correctness backend, not the optimized training implementation.

At each event, one backward obtains local parameter gradients, new intermediate-state cotangents for existing own-leg traces and independent old-state leaf cotangents for shared-path transport. Repair the four gradient projections without counting own-history contributions twice. Release the event graph. Reset/version sensitivities after parameter updates; the current contract performs no updates.

Pass: every score difference <1e-12; every selected gradient projection difference <1e-10; addressed-memory/readout gradient difference <1e-10; all gradients exact for the one-event case. Report the residual complete-gradient discrepancy and all stored traces. Four sensitivities require 352 floats plus a 352-float previous-state snapshot and operator workspace; existing own/history traces each require 7,280 floats, with numerical state, parameters and gradient buffers additional.

After this gate: verify actual Adam steps and learning trajectories under declared parameter-version resets; compare projected-only and native learned remote residual credit against the repaired exact baseline on development data. Complete trace, local derivative, critic, teacher and optimizer cost counts. Raw gradient projection parity does not imply Adam update projection parity or a learning-quality advantage.

Completed run: curie_b10_projected_clock_20261013_v1_contract, uniquely named one-job run_safe queue. Fixed four directions; no learned direction selection claim.
