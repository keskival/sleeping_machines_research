# Deep representational composition: features, parts and objects

15 October 2026. User direction: the missing capability is trainable hierarchical representations, exemplified by event-camera evidence → texture/parts → cat. Sequence length, slower cadence and working two-layer models are not substitutes. This note defines the new architecture and falsifiable claims before admission.

## Objective and diagnosis

A semantic hierarchy should make useful relations of lower features available to higher features. Its depth is the number and nature of learned composition steps. Observation duration is held fixed. The previous language/time-depth program and the new visual project answer different questions. Current experiments do not establish that our deeper stages acquire this capability.

The local event-camera adapter used 4×4 spatial cells. It cannot support a claim about fine texture discarded by that adapter. The new adapter retains 128×128 locations, polarity and eight fixed time bins over the same first second. All depth variants see exactly that evidence. Event-time integration remains a native input operation; the main new computation is spatial composition. DVS Gesture supplies gesture labels, not cats or fur. It is an available engineering development dataset, not proof of the example's semantics. Object/part evidence needs appropriate data and a separate declared protocol.

## Three useful formal statements

**Arrangement obstruction.** If two inputs have the same unordered multiset of feature vectors, any representation that depends only on their pooled mean/histogram is identical. If the target distinguishes their arrangements, every downstream function of that representation must give the same answer. For a balanced two-example witness this forces at least one error. A position-sensitive learned composition can distinguish the pair. This is a restriction of that interface, not a limit of vectors, neural networks or the family. Contextual lower vectors could already preserve arrangement.

**Local composition support.** In an acyclic circuit whose output sites each depend on at most f preceding sites per stage, with no unrestricted global skip or normalization, a depth-D site's support contains at most f^D input sites. A function depending on N independent input sites requires D≥ceil(log_f N) under those restrictions. Having sufficient support is necessary, not sufficient for semantic learning. A dense global operator removes this bound. The implementation therefore uses per-site channel LayerNorm, never spatially global normalization in early blocks. Its learned 3×3 composition and strided transitions progressively combine neighborhoods.

**Exact local race teaching.** At a selected spatial site/group, fix all other site values and the current linear mixer/readout. Let candidate values v_i have route probabilities pi_i=softmax(s)_i and direct stage losses L_i. The conditional local risk is R=sum_i pi_i L_i. Then

    dR/ds_i = pi_i (L_i - sum_j pi_j L_j),
    dR/dv_i = pi_i dL_i/dv_i.

Every losing candidate can receive credit. If the head reads mean(y), replacing a site's selected group value v_w with v_i changes its input by alpha*M_g(v_i-v_w)/(H*W). This gives exact counterfactual local logits without replaying the whole spatial map. Mixer and head are held fixed for this teacher; global factual differentiation trains them separately. This is not an exact derivative of the complete downstream stochastic network risk or an alternative persistent-state-write consequence. The code and tests explicitly preserve that boundary.

The 1/(H*W) factor also exposes a possible weakness: global labels can give weak local routing signals at fine resolution. Log gradient scales, learn measured useful spatial composition first, then introduce local relation/part targets or downstream residual audits if required. Do not pretend a nonzero gradient proves good conditioning.

## Construction

`sleeping_machines/spatial_race_hierarchy.py` implements 32 learned feature-composition blocks, four blocks at each of eight spatial scales, with widths16/24/32/48/64/80/96/128. Each block performs a position-sensitive local composition, separate key scores, two parallel feature races over four candidates, selected value delivery and a residual feature update. Seven learned strided transitions increase the spatial scope; global readout is after the deepest representation. Both local transformations and cross-scale compositions are trainable. Early stage readouts are training/probing aids, not claims that early stages already represent objects.

Each group selects one candidate using an exponential race during training; predefined greedy evaluation uses its highest score. Parallel groups retain complementary candidate values rather than forcing one global winner. The tensor reference backend evaluates all spatial sites and all candidate proposals. Selected activity is bounded but total work is not falsely described as event sparse. The event adapter's learned decay/rotation weights are applied to fixed-bin log counts; this is a declared approximation, not raw-spike exact continuous-time transport.

Global supervised loss differentiates through selected value paths across all32 blocks. Auxiliary stage readouts and exact conditional local candidate enumeration at one rotating spatial audit site per block teach all32 key maps and losing candidates. No part labels, hidden item IDs or future inputs enter the model. Intermediate semantic roles are learned, not assigned by naming a stage “fur”. Stage maps remain separate so spatial arrangement and receptive-field composition can be interrogated.

## Admission and scope

Battle G-HIER extends track G under the user's explicit representational-hierarchy instruction. It is owned architecture development, not a new leaderboard claim or external CNN/Transformer training. Standard convolution and normalization are local support operations inside our hard feature-race construction. Source-bound existing trainers finish unchanged. This new module does not displace the stronger acquired language model or convert a visual diagnostic into a family verdict.

Fixed-record classification protocol: official DVS TRAIN list only, first1s, eight temporal bins,128×128. Fit users1..16; select users17..19; held-out development users20..23; official TEST unopened. The split differs from older fit1..19 controls: do not compare their quality as matched results. Parameters adapt only in fitting; sample states reset between distinct gesture episodes. It is not a chronological online-forecast benchmark. We preserve raw input locations rather than expanding the sequence horizon.

Mechanisms retained: temporal input kernels, separate selection/value paths, hard feature races, parallel selected feature channels, global deep value credit, explicit losing-candidate local credit. Missing integrations: raw asynchronous per-pixel dispatch, persistent per-node write state across event chunks, independent timer emissions, full downstream alternative-write credit, object/part labelled validation and a deployed chip backend. These are concrete integration work; no hidden disabling or claim of full flagship coverage.

First run: contract/prepare/resource smoke → one32-block fit. Resource smoke is not a quality veto. It must show finite integrated updates, composition and key gradients at every block, measured128-resolution RSS/throughput and an economical timeout. Save exact epoch-boundary learner/optimizer and RNG checkpoints; failed partial-epoch work must be charged separately. No automatic depth/width sweep. Native16/32 comparisons become useful only with unchanged evidence/split and a declared total-work boundary; no two-layer proxy stands in for this project.

## What success must demonstrate

1. Deep representation changes improve selected/held-out prediction at fixed observation evidence; training loss alone is insufficient.
2. All stages get useful learning signals. Per-block composition/key gradients and update magnitudes are diagnostics; stage interventions and held-out prediction determine usefulness.
3. Arrangement interventions affect deeper object decisions; spatially averaged stage replacements on a predefined DEV subset identify which learned composition the fitted model actually uses. They do not measure the best possible shallower predictor.
4. Learned feature reuse and part/object semantics require data and tests that identify those concepts: nuisance transformations, new combinations and held-out instances. Gesture classification is a first engineering fit, not a cat-recognition result.
5. Report quality alongside full fitting/per-target/inference work, candidate scoring/proposal costs, selected deliveries, tensor state and traffic, and actual dependency span. Current analytical route counts are not a complete FLOP ledger; measured CPU/RSS are implementation diagnostics.

Architectural contribution remains a hypothesis until this pipeline produces completed evidence. A failed first fit triggers error analysis of interface, composition, candidate exposure, conditioning and supervision; it is not a model-family loss.

## Prior work and attribution

Spatial receptive-field hierarchies, residual paths, convolution, normalization and deep supervision are established primitives. Event-vision precedents include [HATS](https://arxiv.org/abs/1803.07913), [asynchronous sparse convolution](https://arxiv.org/abs/2003.09148), and [AEGNN](https://arxiv.org/abs/2203.17149). The construction here integrates our temporal input/race selection and local counterfactual feature teaching into a deep spatial representation learner. No novelty or state-of-the-art claim follows from the integration alone.
