# Sleeping Machines: a universal trainable computing substrate

Sleeping Machines aims to make intelligence composable within **one trainable computing substrate**. Language and reasoning, perception and world models, embodied action, typed tables and interleaved process streams become different forms of experience available to the same learner. The ambition extends through continual learning, communication and self-design to the hardware that executes them.

An observation enters through an interface that respects its meaning: a token, a sensory event, a typed comparison or an action. It becomes an addressed message interacting with persistent memory. Learned delays and temporal races decide which evidence meets and which computation happens. Small messages carry new evidence into deeper state; separate keys and values distinguish where evidence goes from what it says. Counterfactual credit teaches the routes and writes that could have happened.

The larger promise is that these are **reusable learned computations**. Comparison, binding, retention, prediction and planning learned from one form of experience can support another through shared representations and end-to-end credit. Perception can inform reasoning; reasoning can guide action; process traces and tables can contribute to the same world model. Joint training must measure that transfer and the retention of existing skills.

This also connects the learner to its execution. Time performs computation, including useful sleeps; local events can advance it without a mandatory global clock. Inference and learning are designed around the same temporal, addressed substrate. Available memory and skill can grow beyond the activity recruited for one observation, with work tuned from small data and constrained devices to large datasets and datacenters. Candidate discovery and alternative learning work count in that budget.

The project is therefore a proposal for **a common foundation for learning, representing and executing intelligence**. The experiments test pieces of that construction. Mixed-type tabular learning is a decisive bridge: meaningful numeric, ordinal, categorical and missingness comparisons should feed the same temporal core used for sequences and asynchronous events. Strong results across small and large data would demonstrate the breadth of the unification. Current CPU implementations and separate-domain fits support specific mechanisms; joint transfer, scalable typed learning and clockless learning hardware have concrete tests ahead.

The ambition has distinct design axes. Modality is not the same question as temporal computation; temporal computation does not itself require a global clock; clockless execution does not itself supply a learning rule; and none of these automatically supplies a scaling advantage. Their combination is the construction we develop and test.

| Design axis | Target |
| --- | --- |
| Data and modality | One composable event interface for dense/sparse, synchronous/asynchronous, typed tabular and sequential observations; interleaved processes need not arrive with their identities supplied. Type-aware comparisons can precede neural mixing. |
| Computation through time | Sleeps, delays, temporal races, decay and rotation perform transformations and determine which persistent evidence interacts. Waiting is part of the computation, not only an instruction to skip work. |
| Globally clockless execution | Local arrivals and completion events advance computation without a mandatory global tick or layer barrier. Local timing and numerical clock precision remain meaningful. |
| Learning in the execution substrate | Inference and learning use the same addressed state, temporal programs and asynchronous event semantics. Counterfactual credit teaches unrealized routes and writes. Learning may require additional work; sharing the architecture does not mean identical operation counts. |
| Capacity beyond activity | Available representations, receivers and memories can grow beyond the work recruited for one observation. Candidate discovery, scored keys, selected updates, delivered values and alternative learning work are accounted for separately. |
| Resource and data adaptation | Tune useful computation, memory, depth, routing and learning effort for datacenters, small devices and small-data problems. Seek better quality at equal complete resources, improved scaling, and useful low-resource operating points. |

These are separable design questions with real interactions: learning needs credit and discovery; timing needs precision; capacity needs useful access; low-resource execution needs measured budgets. We aim to preserve the complete construction while learning which allocations serve each regime.

The architectural continuity is explicit: computational delays and races, sparse addressed persistent updates, small messages, distinct keys and values, deep credit to unrealized alternatives and silence-aware supervision where applicable. Dense local operations support this substrate; they do not define its execution schedule.

This document states the ambition. Current CPU drivers use bounded batches and gradient horizons; they are not globally clockless hardware or a completed fully asynchronous learning runtime. Completed quality/work results, engineering contracts and proposed capabilities are identified separately in the report. Scaling superiority, universal transfer and small-device benefits are experimental targets, not conclusions drawn from the ambition.

## Proof of transfer

The first transfer comparison must train a shared core through two domain interfaces and compare it with separately trained cores under the same total fitting/tuning work and target-domain adaptation budget. Check that both losses credit the shared temporal/state/routing parameters, then measure recipient quality, adaptation data/work and retention in the donor domain. Include an unrelated-donor control and report negative transfer. Keep interface, parameter-sharing and persistent-state boundaries explicit.

All participating methods receive the same permitted inputs and supervision. In anonymous FAS, a language interface or auxiliary label must not reveal hidden process identities or identity-derived timing statistics. Selection uses development data; reserved public language validation and sealed FAS confirmation remain protected. Separate language/FAS/tabular benchmarks provide mechanism evidence today; the joint experiment is a new proof, not a reinterpretation of them. It follows the already-admitted quality/resource and headline work, without redirecting the main tokenized language program.
