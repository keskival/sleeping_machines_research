# Joint content and elapsed-time teaching packet

B1/R1 AWS exact execution reference. The established input-only packet held elapsed times fixed. Native autodiff already differentiates time; the new packet makes that information available through the same reusable teaching/credit interface. No actor architecture or active training source is changed.

For complex persistent state z_i=exp((-rate+i*frequency)*dt_i)z_(i-1)+g_i*w_i, the elapsed-time forcing is (-rate+i*frequency)*exp((-rate+i*frequency)*dt_i)z_(i-1). The packet reconstructs this from its existing decay/rotation/write caches without another model forward pass. Two mode vectors per event store real/imaginary forcing. They add2*B*L*n floats; shared parameters, inputs/queries/output and any parameter graph are additional.

`apply_with_time(v,w)` carries joint input/time tangents through temporal recurrence, readout, normalization and GELU. `transpose_with_time(c)` returns both input and elapsed-time cotangents, including future recurrence effects. The reciprocal identity is

    <c, J_input v + J_time w> = <J_input^T c,v> + <J_time^T c,w>.

Version/context/parameter checks include clock changes. Time coordinates are the layer's scaled elapsed intervals. Physical seconds or upstream trainable delays require their producer chain rule and declared normalization; race selection and route counterfactuals still need their own credit mechanism. Eval/no-dropout and first-order derivatives only. This is not a higher-order optimizer-transport or hard-route derivative implementation.

## Completed contract

aws_native_temporal_teaching_packet_20261011_v1: one native layer, random content and real Taxi TRAIN gap patterns,4/8/32 events with the initial zero gap. Maximum JVP error3.33e-15, VJP1.78e-15, reciprocity1.25e-14; zero time directions reproduce input-only responses exactly. Updated clock frequency invalidates the packet. Cache280/560/2240 floats. No learned-credit utility, model quality, whole-fit or speed claim.

Timing sensitivity is now an explicit teaching target rather than an assumption of fixed clocks. Active B4 models remain unchanged. Next integration belongs with the compact-credit owner; preserve exact local derivatives as the reference and charge capture, added state, queries and audits.

## Producer training integration

The fixed-time teaching boundary omitted upstream delay-producer credit. A diagnostic positive internal-tempo producer checks the chain rule through both packet outputs while retaining the native layer parameter graph. Physical addressed-memory aging and race-density time remain unchanged; this fixture is not an active architectural substitution.

aws_timed_producer_actor_20261011_v1 passes four Adam updates on a real four-event Taxi TRAIN prefix. Maximum gradient difference4.17e-17, Adam-state difference5.64e-18 and parameter difference8.68e-19; nonzero tempo gradients range0.0632–0.0677. All producer/layer/head gradients agree with ordinary autodiff. Whole numerical gate0.619s/387MiB. This validates exact temporal credit delivery, not learned-backward quality or efficiency. Next: connect content/time teaching targets to compact-credit integration and evaluate future loss per charged work.

## Shared-clock depth integration plan

The single-layer contract leaves shared-clock overlap untested. For h_l=F_l(h_(l-1),tau), define A_l=partial_h F_l and B_l=partial_tau F_l. Forward teaching carries v_l=A_l v_(l-1)+B_l w. Reverse transport carries c_(l-1)=A_l^T c_l and shared elapsed credit g_tau=sum_l B_l^T c_l. Every layer contribution enters once; no discount belongs in this chain rule. A learned credit approximation may approximate this sum, but its target and sparse residual audit must use the full shared-producer gradient.

The new native_deep_temporal_packet composes existing local packets. It retains physical race and addressed-memory clocks and changes no active model fits. Registered next gate: depths2/4/8,4/32-event joint JVP/VJP and reciprocity, suffix invariance, intermediate-clock invalidation; four real Taxi eight-event Adam updates per depth reconnect all producer/layer/head gradients. First-order dense-write execution diagnostic, not a learned-credit or depth-quality claim. Only after this gate passes: measure whole actor plus joint teaching acquisition, retained cache, producer/optimizer work and no-query overhead before choosing any integration.

## Completed shared-clock depth and fused cost gates

aws_deep_temporal_packet_20261011_v1 passes native depths2/4/8,4/32events: JVP<=5.78e-15,VJP<=6.67e-15,reciprocity<=3.02e-14, exact suffix invariance and stale intermediate-clock rejection. Four eight-event Adam updates per depth reconnect every parameter: gradient<=2.45e-15,Adam<=3.34e-16,parameters<=1.12e-16.

The unfused joint path repeats recurrence/readout work for content and timing. native_fused_temporal_packet removes that duplication without changing the primal, temporal mechanism or active fits; fused joint depth/Adam gates pass. aws_fused_deep_temporal_teaching_cost_20261011_v1 measures complete real Taxi batch4/32-event actor/loss/backward/Adam plus joint queries. At depths2/4/8/32, one-query speedups1.616/1.652/1.606/1.592;16-query speedups1.748/1.759/1.757/1.718, versus native joint forward AD. All gradient/Adam/response checks pass. Whole cost gate21.724s/500.5MiB, retained cache70,144/140,288/280,576/1,122,304floats, excluding shared parameters and autograd graph. Zero-query speedups0.810/0.762/0.741/0.731: capture is slower for ordinary fitting and stays conditional on teaching work. No critic fitting/use, task quality or complete learning-efficiency claim.

## Learned credit and actual-loss transfer

The differentiable joint_forward_credit loss trains predictions while freezing measured responses, probe directions and terminal loss cotangents. At native depths2/4/8 its1716parameter neural-estimator gradient matches exact cut/shared-producer regression within3.56e-15. Three fixed local acquisition seeds learn a1152parameter joint transpose map from forward labels and generalize to64fresh terminal directions per depth, with residual-energy reduction99.856–100% versus zero. These are artificial receivers at one frozen4-event TRAIN context, not new task data or whole-learning gains.

Saved field/actor bundles restore without refitting. On real observed-loss shared-producer credit, learned fields beat identity in9/9same-context and9/9one-Adam-update cases;16/18new-context cases improve,2worsen atdepth8/seed1. Missing context conditioning is the concrete failure; expired fields remain approximations requiring revalidation/audits. State-conditioned risk is developed from forward residual observations, followed by a fixed64/256/1024context acquisition ladder with new128-context validation. No active architecture substitution, future task learning or full-gradient learning-efficiency claim.
