# An innovation-driven, multirate embodied temporal hierarchy

14 October 2026 · Architectural proposal within the temporal/sparse family. No implementation, benchmark win, robot deployment or new training admission is claimed. Existing language acquisitions and isolated repairs continue. This proposal addresses the user's requirement for an architectural hierarchy with genuinely different event and action cadences.

## Failure being addressed

Current native token members execute every stage for every token. Changing their optimizer horizons does not change that computational graph. In embodied sensing, frequent pixel/encoder events should not force task-level memory writes or task planning. Deeper persistent representations should be updated when additional information matters, while fast action heads remain responsive.

Retain computational delays/races, sparse addressed writes, separate keys/values, content/private-memory mixing, counterfactual alternatives and silence-aware supervision. The architectural change is selective event propagation between persistent levels, top-down predictions/intent and separately scheduled output heads. It replaces all-events-through-all-stages execution, not temporal computation with a dense carrier. The original coalescing theory30 supplies exact affine-state aggregation and its nonlinear downstream boundary; it does not justify discarding arbitrary nonlinear messages.

## Practical robot stack

Illustrative service deadlines are design targets, not measurements or forced internal clocks:

| Level | Persistent representation | Example outgoing event | Output head |
|---|---|---|---|
| Local sensor/motor | Per-patch motion/contrast memory; joint/contact dynamics | Unexpected slip, contact impulse, changed local motion | Fast stabilization/impedance correction, roughly1–5ms service |
| Body/object | Object pose/velocity, contact mode, grasp state, pose uncertainty | Grasp acquired/lost; obstacle trajectory changed | End-effector or locomotion target, roughly10–100ms service |
| Skill/scene | Tool/object relations, skill phase, local map | Insertion aligned, doorway reached, current skill blocked | Skill selection/trajectory parameters, roughly0.1–1s service |
| Goal/episode | Task constraints, progress, unresolved goals | Subgoal completed or violated | Task/subgoal commands on transitions, often seconds apart |

These are different physical units and different output responsibilities, not competing torque predictors. High levels set goals and constraints; lower controllers own actuator commands and can report or override an invalid goal. Commands carry their reference frame, issuance time, validity/deadline and goal version. A trajectory/feedback law can run between expensive policy decisions. Fast control reads its own current state and the latest valid higher-level intent; it does not wait for a full stack pass.

## State, communication and scheduling

Each level has addressed private modal memories and an independent event queue. A message contains a destination/key, timestamp, payload and optional confidence/validity metadata. On an admitted event, a race selects a small receiver set, combines incoming content with decayed/rotated memory, writes selected state and optionally emits a higher-level event.

Between events, linear modal evolution is evaluated lazily at the next write, query or scheduled deadline. Numerical state persistence does not mean frozen belief. No periodic scan of all levels or all memories is required by this design. Every index lookup, selected read and scheduled query remains paid.

Three cadences must stay separate:

1. Sensor arrival rate: irregular, potentially very high for event cameras.
2. Internal write/communication rate: learned/task dependent and different by level.
3. Output service rate: set by control/query requirements, potentially fast even when high-level memory is quiet.

There is no hard rule that a deeper level must always be slower. Sudden hazards can propagate immediately through the hierarchy. Nor is every event camera event promoted to a semantic packet: pixel addresses select local spatial memories/regions cheaply, with explicit candidate discovery, and summaries are transmitted only under a contracted aggregation/innovation rule. Arbitrary fixed frame conversion is not the defining input interface.

## What makes an event worth transmitting

Higher-level state predicts lower-level evolution; motor commands supply an efference copy for predicting self-induced sensor motion. An innovation is observed evidence beyond that prediction, measured in a learned task/uncertainty metric. It is not simply a large Euclidean residual: a tiny impending slip may be more valuable than a large predictable illumination transient.

Use a hard emit/hold race with supported alternatives. Its intended decision compares expected later loss reduction with incremental discovery/write/communication cost. A learned future-utility predictor may eventually amortize that estimate, while supported forward audits teach missed/optional routes. The existing finite-horizon route-credit contracts are reusable ingredients; they do not already implement this multirate policy.

Silence can mean stability, a missing sensor, occlusion or a failed expected transition. Timer/deadline events and event-survival supervision keep those cases distinguishable. Rare higher-level paths must still receive counterfactual credit; selecting training sites only among emitted events would hide useful suppressed events. A target-periodic clock alone is not a learned abstraction boundary.

## Why this is an architectural scaling test

Let lambda_l be actual event arrivals to level l, p_l its probability of emitting one upward event, and mu_l extra timer/top-down arrivals. In the restricted one-message case,

    lambda_(l+1) <= p_l lambda_l + mu_l.

Expected work per second includes

    sum_l lambda_l*(candidate_discovery_l + selected_update_l + delivery_l)
      + sum_heads query_rate_h*head_work_h + timer/index/learning work.

Decreasing event rates can permit more levels/capacity without multiplying full work per raw sensor event. This is a hypothesis to measure, not a guarantee: high entropy, bursts, dense key search, repeated replay or expensive output heads can erase it. Charge losing-route and silent-event learning as well as inference. Deep active communication paths must remain available; theory20's disconnected-path obstruction still applies.

The decisive comparison is an always-on and a selective-propagation version of the same temporal family with identical modalities, available capacity, targets and action service deadlines. Test task quality versus total work and burst-tail latency; a lower event count alone is not success. Fixed periodic subsampling and random thinning are diagnostic own-model controls, not the main model.

## Learning the hierarchy

Use local prediction/observation losses where targets exist, action losses at each labelled output cadence, and delayed task outcomes. Normalize/weight objectives deliberately so plentiful low-level targets do not numerically erase rare high-level learning. Do not impose oracle skill boundaries at inference. TRAIN labels can supervise those boundaries, but evaluation must use only causal sensor/goal inputs.

Reuse acquired joint temporal operators and preserve parent update state where a continuation is defined. Neutral new branches require the full clock/context containment contract, not gain0 alone. Cross-credit feature seeding is a possible response when internal features fail to create output-error correlation; its cost/noise assessment and numerical gates precede any fit.

Internal counterfactual replay on recorded sensor streams can evaluate alternative computation/write choices. It cannot establish the physical consequences of alternative robot actions: those change subsequent observations. Action-level claims need closed-loop rollouts or explicitly justified dynamics/coverage assumptions.

## Benchmarks checked against primary sources

- **REASSEMBLE** is the strongest initial offline candidate for the complete sensing/skill hierarchy: event arrays with separate timestamps, force/torque, proprioception, RGB and two annotation levels. Use native sensor timing and causal prediction of skill transitions/failures; compare sensor ablations and report actual data/missing-sensor protocol. Recorded demonstrations alone do not establish closed-loop policy success. Official archive: https://researchdata.tuwien.ac.at/records/0ewrv-8cb44 ; project/labels: https://tuwien-asl.github.io/REASSEMBLE_page/ .
- **CEAR** targets fast mobile perception: event camera, RGB-D, IMU, LiDAR and joint encoders on a Mini Cheetah. Useful for event-to-body-state fusion and perception under agile motion, not by itself a complete hierarchical control benchmark. Primary paper: https://arxiv.org/abs/2404.04698 .
- **DSEC** provides public event-camera optical-flow/disparity benchmarks. Useful for local/mid-level temporal vision and comparable public metrics; it does not test robot action heads. https://dsec.ifi.uzh.ch/ .
- **Closed-loop simulation:** robosuite exposes action-control frequency separately; ManiSkill exposes simulation and control frequency. A multirate event-sensor/control setup requires explicit high-rate observations or an event renderer, not merely reporting a fast physics timestep while the policy sees slow observations. Official docs: https://robosuite.ai/docs/simulation/environment.html ; https://github.com/mani-skill/ManiSkill/blob/main/docs/source/user_guide/tutorials/custom_tasks/advanced.md .
- **Longer-horizon follow-on:** CALVIN tests language-conditioned skill sequences, and BEHAVIOR-1K tests long household activities. They address goal/skill memory but are not stand-alone millisecond event-camera tests. https://arxiv.org/abs/2112.03227 ; https://behavior.stanford.edu/ .
- **SEBVS** is a smaller event-based navigation/manipulation candidate using Gazebo and video-to-event synthesis. Its sensor-generation assumptions must be separated from a real-event claim. https://eventbasedvision.github.io/SEBVS/ .

First proposed decisive benchmark: causal multisensor contact/skill-transition prediction on a bounded REASSEMBLE split, with separate fast and slow heads; always-on native versus innovation-propagating native, same sensor information, reporting hierarchical task quality, each level's write/event rate, complete fit/inference work and reaction-latency distribution. Then closed-loop contact-rich control to test action consequences. Dataset/license/split audit, implementation contracts and battle admission are required before training. No new dataset download or robotics fitting has been launched by this proposal.
