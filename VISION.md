# Sleeping Machines: a universal trainable computing substrate

Sleeping Machines aims to unify learning and computation across language and reasoning, multimodal world models, embodiment, heterogeneous tables, synchronous samples, asynchronous events, anonymous interleaved process traces, token sequences, continual learning, communication, self-design and hardware. One task is evidence for one front, not the definition of the project.

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
