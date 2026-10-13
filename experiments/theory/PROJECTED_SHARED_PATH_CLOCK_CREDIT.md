# Exact projected shared-path credit with learned residuals

13 October 2026. B10 implementation direction, derived from the completed integrated market eligibility contract. This is a proposed extension, not a measured quality or efficiency result. It retains the native numerical forward computation and exact local derivatives.

The own-leg traces preserve addressed-memory/readout gradients through 128 events, but omit historical changes to lower-layer inputs and upper-layer writes. At 32 events the largest missing gradient is a clock-frequency coordinate (0.227606). Rather than approximate every parameter equally, carry a bounded set of sensitivities through the complete shared state transition.

Let s_t = F_theta(s_(t-1), x_t), with fixed theta inside the declared credit window, and loss l_t(theta, s_(t-1), x_t). Choose a parameter direction matrix D with r orthonormal columns before the window. Define S_t = (d s_t/d theta) D. Then

    S_t = A_t S_(t-1) + B_t D,
    A_t = dF_t/ds_(t-1), B_t = dF_t/dtheta.

Initialize S_0=0 when the warm state is treated as constant, matching truncated BPTT. Carry the complete state sensitivity, including upper layers and addressed memory: only the parameter-direction dimension is restricted. The recurrence is exact for these directions; it does not discard cross-layer paths within them.

For the current event graph, detach the old numerical state and make its components independent leaves. Obtain local parameter gradient g_local and old-state cotangent c_old in one backward call. The exact directional gradient is

    D^T g_full = D^T g_local + S_(t-1)^T c_old.

Use the OLD-state cotangent here. Dotting a total state sensitivity against cotangents of dependent new intermediate states can count shared paths twice. The current-event parameter effects already appear in g_local.

If an existing own-leg eligibility method supplies historical parameter contribution h_own, retain it and repair just the chosen subspace:

    g_repaired = g_local + h_own
                 + D [S_(t-1)^T c_old - D^T h_own].

Because D^T D=I, D^T g_repaired equals the full window gradient projection. The orthogonal residual still requires credit. This equality concerns the raw gradient; Adam's coordinatewise preconditioning need not preserve the same update projection. Test actual optimizer steps separately.

For the saved two-layer market actor, a first declared basis consists of four disjoint unit directions: each layer's mean log-decay and mean frequency. The state has 88 scalar coordinates, so these projected sensitivities require 352 floats independent of history length. Parameter snapshots, own traces, tangent workspaces, current-event graph and gradient buffers are additional. Maintaining r directions costs tangent propagation through the active operators; this is not a free substitute for backward computation or a hardware-energy claim.

An economical implementation propagates analytic tangents through oscillator rotations, gated writes, read maps, LayerNorm, the actor's actual GELU MLP, and addressed memory. For y=LayerNorm(x), the directional derivative is

    dy = gamma / sqrt(v+eps) *
         [dx - mean(dx) - (x-mean(x))*mean((x-mean(x))*dx)/(v+eps)].

The GELU derivative for the exact Gaussian definition is Phi(x)+x phi(x). The zero-gap case retains identity temporal transport and current write derivatives. No layer replacement or activation substitution is authorized by this derivation.

Fixed D defines the proven contract. A future native critic can choose directions using earlier context and delayed utility, with its selection/exploration and tangent costs charged. Changing D does not retroactively provide old sensitivities for newly chosen directions. Reset at a window boundary, retain a spanning basis, or explicitly transport a basis change lying in the stored span. Parameter updates also require declared trace resets/versioning; unchanged numerical memory does not validate derivatives computed at an old parameter version.

The next integrated contract must preserve every forecast, reproduce four full-BPTT gradient projections at 1/8/32/128 TRAIN events, retain exact addressed-memory/readout derivatives, and report the remaining whole-gradient discrepancy. Only then compare actual learning updates and total work against the repaired exact baseline. Learned residual credit is taught by future predictive utility, not assumed correct from this exact directional identity.
