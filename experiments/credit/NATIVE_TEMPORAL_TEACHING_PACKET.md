# Joint content and elapsed-time teaching packet

B1/R1 AWS exact execution reference. The established input-only packet held elapsed times fixed. Native autodiff already differentiates time; the new packet makes that information available through the same reusable teaching/credit interface. No actor architecture or active training source is changed.

For complex persistent state z_i=exp((-rate+i*frequency)*dt_i)z_(i-1)+g_i*w_i, the elapsed-time forcing is (-rate+i*frequency)*exp((-rate+i*frequency)*dt_i)z_(i-1). The packet reconstructs this from its existing decay/rotation/write caches without another model forward pass. Two mode vectors per event store real/imaginary forcing. They add2*B*L*n floats; shared parameters, inputs/queries/output and any parameter graph are additional.

`apply_with_time(v,w)` carries joint input/time tangents through temporal recurrence, readout, normalization and GELU. `transpose_with_time(c)` returns both input and elapsed-time cotangents, including future recurrence effects. The reciprocal identity is

    <c, J_input v + J_time w> = <J_input^T c,v> + <J_time^T c,w>.

Version/context/parameter checks include clock changes. Time coordinates are the layer's scaled elapsed intervals. Physical seconds or upstream trainable delays require their producer chain rule and declared normalization; race selection and route counterfactuals still need their own credit mechanism. Eval/no-dropout and first-order derivatives only. This is not a higher-order optimizer-transport or hard-route derivative implementation.

## Completed contract

aws_native_temporal_teaching_packet_20261011_v1: one native layer, random content and real Taxi TRAIN gap patterns,4/8/32 events with the initial zero gap. Maximum JVP error3.33e-15, VJP1.78e-15, reciprocity1.25e-14; zero time directions reproduce input-only responses exactly. Updated clock frequency invalidates the packet. Cache280/560/2240 floats. No learned-credit utility, model quality, whole-fit or speed claim.

Timing sensitivity is now an explicit teaching target rather than an assumption of fixed clocks. Active B4 models remain unchanged. Next integration belongs with the compact-credit owner; preserve exact local derivatives as the reference and charge capture, added state, queries and audits.
