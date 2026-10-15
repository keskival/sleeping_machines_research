# G-HIER current findings — 15 October 2026

## Native multi-memory composition learns a usable FIT signal

Eight abstraction levels, four native race stages per level, 169,675 parameters.
Persistent indexed objects integrate child events and query responses from all
active lower learned memories in their support. The final round has no raw-input
shortcut. Reads preserve child state; root publication is generated at a native
race time. Query selection, ownership and arming policies remain explicit.

On 12 distinct 32-event prefixes from one official FIT recording, 64 batch-one
updates reduce paired stochastic NLL 2.547975 → 2.482746. Two common noise tapes
are shared across all episodes, with bitwise train/evaluation forward parity.
Label association rises −0.013494 → +0.037650; it is positive on both tapes.
Raw accuracy reaches the majority predictor's 16.67%; raw NLL remains above the
empirical constant-prior 2.369382. This prompted a frozen readout diagnosis,
not another unchanged backbone fit.

| Frozen construction | Raw NLL | Prior-centred FIT NLL | FIT accuracy |
| --- | ---: | ---: | ---: |
| Initial core and head | 2.547975 | 2.369382 | 16.67% |
| Trained core and head | 2.482746 | **1.925817** | **37.50%** |
| Trained, query contents zeroed | 2.452741 | 2.369382 | 16.67% |
| Initial core, trained head | 2.541522 | 2.369382 | 16.67% |
| Trained core, initial head | 2.461518 | **1.926302** | **37.50%** |

Scope: one FIT recording, two stochastic tapes, no DEV/TEST access. Prior,
logit centre and one nonnegative scalar gain are fitted and scored on these same
FIT logits. These are optimistic learnability results, not generalization or a
semantic hierarchy win. Swaps measure these fixed decoders, not a basis-invariant
ranking. They locate a useful learned signal in the native core for this probe;
removing lower-memory content removes that gain. No per-item noise identifier is
provided. The calibrated artifact changes only the external linear readout.

The loss identity L = H(p) + mean KL(p || q) − A explains why erasing useful
query content can improve raw loss: content adds 0.051704 label association and
0.081709 label-independent penalty. Prior-centred logits expose the signal.
For q(g)=softmax(log p + g(z−mean z)), convex FIT loss has derivative −A at g=0.
This guarantees a positive-gain FIT improvement when A>0, not held-out transfer.

## Native learning and execution costs

64 training presentations contain 2,048 raw event presentations; unique input
contains 384 events. Fitting plus all scheduled scoring took 961.276 CPU seconds,
peak RSS 2,074,196 KiB. Frozen five-way diagnosis took 164.565 seconds and
297,376 KiB RSS. These are implementation diagnostics, not hardware estimates.
For the measured first episode: 79 objects, 78 lower-memory queries, 334 delivered
messages/publications, 10,688 candidate scores and 5,344 selected native updates.
Learning evaluates 10,688 counterfactual values; selected-only evaluation does
not. Two timer candidates are scored. Full fitting FLOPs, per-target arithmetic,
traffic and hardware energy remain unaudited. State tensor counters exclude
graphs, input records, queues, aliases, parameters and Python metadata.

## Formal and implementation advances

State-ownership contracts protect lower published states from a changed private
upper hypothesis. This is inference-state independence, not independence from
shared parameter learning. Native coupled value/time credit passes an integrated
expectation identity for affine selected-value losses; nonlinear value losses
retain a local surrogate. Do not modify its timing derivative in isolation.

`VarianceReducedNativeComposition` preserves native parameters, random draws,
forward outputs, publication times and event work. Only pure emission-timer
credit changes: average the winner indicator over its conditional race law,
removing winner variance from the interior timing gradient. Receipt/value races
retain their coupled teacher. This does not repair event-order discontinuities
or establish a training-quality improvement. See the linked mathematical notes.

## Completed synchronous diagnostic and interpretation

On12 FIT episodes,64updates at batch2 (128 sampled training presentations), the older cached synchronous model's argmax-route
NLL rises2.494155→2.520648; accuracy stays1/12. A frozen diagnosis using three
fixed stochastic race seeds finds average NLL improves by0.010006 instead;
argmax worsens by0.026493. Therefore some stochastic-objective learning occurred,
but it did not yield useful classification or transfer to argmax execution.
Do not describe this as zero learning, successful hierarchy or a family loss.
Root feature standard deviation rises0.10513→0.14717 under argmax; total feature
collapse is not supported by this statistic. It does not measure class separation.

Removing the final model's top raw-read source gives2.520060, an improvement of
0.000588. Of11 top raw reads,9 hit empty input. Static geometry explains the
restricted access: at level7 on128x128 input, source0 stencil offsets are-128,
0,+128 around the origin; only(0,0) is valid. Thus retaining memory was not enough:
this candidate-discovery construction failed to offer broad raw detail at the
highest level. This does not identify the sole cause of the learning failure;
other source reads and credit variance require separate diagnosis.
Sources: `results/hierarchy/aws_g_hier_cached_20261015_v2_probe.json` and
`results/hierarchy/aws_g_hier_cached_frozen_diagnosis_20261015_v1.json`.

## Coalesced native composition: measured resource advance

An eager adjacent-level read-round construction exceeds the3,000,000KiB slot
limit: watchdog stops at3,202,220KiB. Its intermediate publications repeatedly
cascade upwards. The new coalesced construction preserves native local writes
and indexed publication, but separates upward transmission: every active child
arms a native race and emits one evolved view per explicit composition round.
Parents integrate those views before becoming the next frontier.

| Work on the same32-event TRAIN scene/acquired weights | All-level terminal reader | Coalesced child emissions |
| --- | ---: | ---: |
| Addressed objects |94|94|
| Delivered messages / publications |349|125|
| Native candidate scores |11,168|4,000|
| Selected native updates |5,584|2,000|
| Counterfactual value evaluations |11,168|4,000|
| Timer candidate scores |2|188|
| Timer emissions |1|94|
| Indexed query calls |93|0|
| Native state tensor bytes |93,040|82,760|

Deliveries and native candidate work fall2.792x, with a changed information path
and more timer work. Both constructions receive the same observations and have
identical acquired parameters; they need not predict equally well. All eight
levels have finite nonzero content gradients. No optimizer or task-quality score
is used in this resource gate; the old11-class gesture head remains solely for
numerical gradient checks. Software forward/backward time10.286s→4.231s; measured
process peak RSS rises to1,350,556KiB across the sequential pair and is not an
isolated per-arm memory comparison. Complete FLOPs/traffic remain unaudited.
Evidence: `results/hierarchy/aws_g_hier_coalesced_preflight_20261015_v2.json`;
failed gate: `aws_g_hier_frontier_resource_stop_20261015_v1.json`.

The task generator now implements three genuine nested arrangement levels with
eight recurring four-event motifs. Each split contains all child codes, balanced
labels, both relation parities and both root orientations. TRAIN128/DEV64/TEST64
scenes have disjoint child combinations and nuisance tapes. A single complete
child plus root orientation is label-balanced; two children plus their relation
determine the target. Internal identities are audit metadata only. This is a
controlled composition diagnostic; eight object levels do not imply eight
verified semantic levels, and a flat expressive model could also solve the task.

## Completed held-out composition development gate

The128-update coalesced whole-label fit completes,169,378parameters, with exact
model/Adam/RNG checkpoints. It fails unseen-combination DEV: raw NLL1.163353,
TRAIN-fitted scalar-calibrated NLL2.508095, accuracy50%, versus constant prior
0.693147/50%. Initial raw DEV was0.708274; initial calibration selects gain0 and
returns the prior. Final TRAIN calibration hits gain cap256, improving TRAIN
NLL by only0.000025 (0.693122). Its DEV amplification is harmful; no gain from
this fit is claimed. This is one short development attempt, not a family verdict.

A frozen full affine-head audit with fixed TRAIN-standardized ridge.01 does not
recover transfer: initial/final TRAIN NLL0.615729/0.658176 but DEV551.710/216.153,
both50%accuracy. This is a severe out-of-distribution readout failure. It does
not prove absence of all useful information in every native memory. No native
backbone is changed; feature caches are retained for analysis.

Cached root-feature analysis identifies nuisance domination. Mean per-coordinate
feature standard deviation is0.000525/0.000559 on TRAIN versus0.031882/0.033106
on DEV for initial/final cores. Final DEV mean shift is298 TRAIN standard
deviations RMS (coordinatewise with1e-6scale floor). For paired scenes holding
arrangement/label fixed, nuisance changes move final root features by0.494892;
flipping the root relation and label moves them by0.001403, about353x smaller.
Both timing/order and translation vary in that nuisance pair; this statistic
alone cannot separate them or identify one causal failure.

Geometric occupancy is94objects on both TRAIN nuisance tapes,127and106 on DEV.
Small global translations therefore change the number of native emitters and
state updates despite identical label/part multisets. Combined with changed
timing, this motivates a frozen factorial audit before architectural repair.
The task has unseen combinations and nuisance realizations; local motif-shape
instance transfer is a separate future gate. Eight object levels include upper
unary consolidation and are not eight verified semantic abstractions.

Fitting plus initial/final scoring545.072CPU seconds, peak RSS684,948KiB.
Training128presentations/4,096raw events charges512,000native candidate scores,
256,000selected updates,512,000counterfactual value evaluations,24,064timer
scores and16,000deliveries. Frozen affine audit145.009seconds. Phase counters
sum state/object exposures; they are not peak capacity or complete FLOPs.
No hardware energy/latency or matched-quality efficiency claim follows.

Evidence: `aws_g_hier_coalesced_learning_20261015_v1.json`,
`aws_g_hier_coalesced_readout_20261015_v1.json`,
`aws_g_hier_coalesced_transfer_diagnosis_20261015_v1.json` under
`experiments/results/hierarchy/`.

## Factorial diagnosis and observed-frame repair

The frozen2x2TRAIN audit separates coordinates from timing/order. On four child
pairs/two root orientations, final root displacement is0.416171 for translation
only,0.000228 for timing only,0.416184 together; label-changing root relation
moves0.000562. The coordinate nuisance dominates on this one-tape fixture.

`FrameNativeComposition` keeps every raw world pixel's persistent native memory,
then commits a scene frame from already revealed pixel bounds at composition
time. Upper ownership/pose uses coordinates relative to that frame. No sensor
event is replayed, no future observation/part ID used, and parameter keys/shapes
are unchanged. Pixel indices remain world-addressed; upper memory uses the
committed frame. Ordinary coordinate anchoring is a known primitive, not a
novel geometric claim. Learned moving-object ownership remains future work.

On the same acquired final weights/factorial scenes, pure translation produces
exactly identical upper features. Timing-only and combined displacement are both
0.003158; the root-relation displacement is0.001153. Combined nuisance falls
about132x, while timing sensitivity increases versus the predecessor. This
removes the measured dominant coordinate failure; it does not prove learned
semantic transfer or full nuisance invariance. Common-tape sensitivity includes
changed stochastic schedules.

The one32-event backward uses119native objects,150deliveries/publications,
4,800candidate scores,2,400selected updates,4,800counterfactual value evaluations,
238timer scores and119timer emissions;104,320native state tensor bytes. Frame
work adds62bound comparisons and128coordinate subtractions. All eight levels
have finite nonzero content gradients. Peak process RSS645,096KiB across the
gate;21.443seconds including32factorial evaluations and backward.76combined
contracts pass. Model169,378parameters; no optimizer or quality score in gate.
Compared with the coalesced predecessor, this adds25deliveries and800native
candidate scores on the measured scene. Full arithmetic/traffic remain unaudited.

Evidence: `aws_g_hier_nuisance_factorial_20261015_v1.json` and
`aws_g_hier_frame_preflight_20261015_v1.json` under `experiments/results/hierarchy/`.
[Derivation and policies](../theory/OBSERVED_FRAME_NATIVE_COMPOSITION.md).

## Completed observed-frame learning and next decision

`aws_g_hier_frame_learning_20261015_v1` completes128whole-label TRAIN updates,
using the same acquired gesture core/fresh2class initialization, data/order and
optimizer as the failed coalesced fit. Finite checks pass; final raw DEV NLL
1.297693 versus initial0.716158. TRAIN-only scalar calibration selects gain0
both before and after training: DEV0.693147/50%accuracy, equal to the prior.
The fixed TRAIN-ridge.01 affine readout also fails transfer: initial/final DEV
0.732992/0.743684, accuracy42.19%/46.88%; TRAIN0.666768/0.669694. No emitted-content
intervention is triggered because the quality gate fails. No TEST model scoring.

The coordinate mechanism repair is demonstrated; learned part-combination
transfer still requires a representation/credit repair. These short fit outcomes
are evidence about the current construction and pipeline, not a family verdict.
No larger unchanged acquisition or depth grid is admitted. Reuse cached initial/
final TRAIN/DEV root features to diagnose part/role binding and credit, keeping
coordinate invariance explicit and existing native races/memory intact.

Fitting plus scheduled scoring and readout diagnostics632.316CPU seconds,
peak743,168KiB RSS,169,378parameters. Training128presentations/4,096raw events:
614,400native candidate scores,307,200selected updates,614,400counterfactual
value evaluations,30,464timer scores,19,200deliveries and15,232timer emissions.
State/object counters summed over scenes are exposures, not peak capacity.
Full arithmetic/traffic and hardware costs remain unaudited. Exact model/Adam/
race/order checkpoints every8updates and initial/final feature caches persist.
Evidence: `experiments/results/hierarchy/aws_g_hier_frame_learning_20261015_v1.json`.

The emission-order boundary audit separately identifies a learning term missing
from pure interior derivatives. Two deterministic integral witnesses pass; this
is a teacher-design target, not integrated exact credit. [Derivation](../theory/EMISSION_ORDER_BOUNDARY_CREDIT.md).

## Role-memory and interaction-depth construction

The new parent caches the latest child view/time per(source abstraction,quadrant),
then aligns views with native protected/rotating transport at parent admission.
Role-conditioned maps preserve which part filled which role; ordered bilinear
maps expose cross-child interactions. The original native core still consumes
every accepted message, preserving hard races,separate keys/values,persistent
state,local counterfactual learning and race-generated views. Sparse occupied
role/pair maps are evaluated; full optimizer/parameter work remains paid.
Pixel objects receive no redundant relation module. Ownership is geometric;
this is not learned arbitrary object discovery or exact future-order credit.

The algebraic contract realizes an eight-way product through three composition
operators on all256sign assignments, with live derivatives to every input.
This proves an available interaction path, not a trained native semantic result.
Nonzero variance-scaled operators are used at every parent; prediction changes
are explicit. Native checkpoint parameters migrate exactly, and the original
constructor/race RNG is preserved. [Theory](../theory/ROLE_MEMORY_AND_INTERACTION_DEPTH.md).

One real-size two-scene optimizer gate per arm:169378→255394parameters, unchanged
9600native candidates/4800selected updates/9600losing-value evaluations and
300deliveries. The new module adds811008forward projection MAdds,2944pair
products,1248native transport-vector calls and32096logical role-view tensor
bytes across the pair. These counters exclude relation backward/optimizer and
complete native arithmetic; full FLOPs/traffic remain unaudited. Paired process
update time7.420→7.912s; cumulative process RSS reaches761580KiB. These are
implementation measurements, not equal-quality or hardware efficiency results.
All eight levels receive finite nonzero native content gradients; top pair-map
gradient norm0.10751. Completed gate:
`experiments/results/hierarchy/aws_g_hier_relational_preflight_20261015_v1.json`.

**Paired role-memory result.** frame: NLL0.728851, accuracy46.88%; relational: NLL0.694245, accuracy51.56%. No arm clears the transfer gate. Reuse cached features and diagnosed interaction/support/credit paths;no unchanged larger fit or depth sweep. Scope: one native seed,64paired updates/128scene presentations per arm,TRAIN-only ridge.01 affine head,64reused synthetic DEV scenes and observed-geometry clustered uncertainty;TEST not model-scored. Equal presentations do not mean equal parameters/arithmetic. Source: `experiments/results/hierarchy/aws_g_hier_paired_decision_20261015_v1.json`.

Endpoint attribution includes zero emitted values,fixed lower arm-to-emit delays,
and both together,under the same native race tape. Addresses/availability and
internal computational clocks remain; fixed delays also change transport.
These are frozen diagnostics,not flagship restrictions. The amendment precedes
both completed endpoints;primary quality/actors are unchanged. [Channel audit](../theory/NATIVE_EMISSION_CHANNEL_AUDIT.md).


**G-HIER interaction-credit repair,15October:** completed16TRAIN-scene audit finds comparable role-value median norms0.915–1.270 but median gradient-times-value norm0.004035 at root versus5.072e-8 at level1 (~79556x). Local first-order sensitivity under acquired joint head/native teacher,not causal attribution. New carried-role actor retains latest ordered child interaction as version-bound persistent state and transports it into indexed/generated views alongside native transformed memory;all32native stages/races retained. Four new contracts/eight combined pass,including exact zero-strength parity and private-future isolation. One guarded paired-update gate compares sensitivity/resources from acquired weights;no DEV/TEST or automatic larger fit. Admission restriction text inherited the16-scene audit count;the gate adaptation,driver and result correctly use one pair/two scenes. [Theory](theory/CARRIED_ROLE_REPRESENTATIONS.md).


**G-HIER carried-view learning admitted,15October:** the one-TRAIN-pair integrated gate improves mean local interaction sensitivity at every parent level (2.73x lowest,9.30x root), with finite all-level native gradients. Paired update 8.002→8.473s; cumulative peak792460KiB,not isolated memory or hardware efficiency. Persistent composed views are bound to published versions;private future views cannot leak. Slots1/6 now compare continued acquired relational actor versus carried actor(initial gain0.5),64additional paired updates/128presentations each. Both reuse the acquired raw native/composer/head weights with fresh equal AdamW/order seed239;no prior optimizer reuse. Existing TRAIN-only ridge.01 affine endpoint and synthetic DEV protocol;TEST closed. `g_hier_carried_decision_v1` waits both fits,admits frozen value/timing cuts,updates report/findings and stops. No larger fit automatically. Source: `experiments/results/hierarchy/aws_g_hier_carried_role_gate_20261015_v1.json`.

**Carried role-memory result.** continued: NLL0.686609, accuracy54.69%; carried: NLL0.716414, accuracy45.31%. No arm clears the transfer gate. Reuse cached features and diagnosed interaction/support/credit paths;no unchanged larger fit or depth sweep. Scope: one native seed,64paired updates/128scene presentations per arm,TRAIN-only ridge.01 affine head,64reused synthetic DEV scenes and observed-geometry clustered uncertainty;TEST not model-scored. Equal presentations do not mean equal parameters/arithmetic. Source: `experiments/results/hierarchy/aws_g_hier_carried_decision_20261015_v1.json`.


**G-HIER predictive responsibility,15October:** implemented rate-independent multiscale residual access and gate-independent per-level whole-support credit inside the carried native hierarchy.13contracts pass. Full-resolution one-TRAIN-pair update gate completes:263923parameters,9.238s,810724KiB peak;finite content gradients at all8levels. Initial allocation63.23% to level0,23.26% to level1,8.56% to level2;prediction-loss gate derivatives +0.051737 at0,-0.035359 at1,-0.017724 at2,so descent favors1/2 over0 without a depth schedule. This is one-update learning-path evidence,not semantic transfer. All transmissions remain enabled;238extra output reads/paired update and full output work charged. Mean spatial summaries are a restricted access diagnostic;native role-aware higher composition retained. Common prediction time also contains higher scheduling history,so allocated source level is not evidence provenance. No new quality fit admitted:current matched carried-view fits finish unchanged before selection. [Theory](theory/PREDICTIVE_RESPONSIBILITY_TRANSFER.md).
