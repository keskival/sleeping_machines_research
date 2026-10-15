# Exact suffix replay makes higher-level composition cheaper to teach

15 October 2026. G-HIER implementation stage, no new trained quality claim.

## Failure and construction

The addressable parent architecture gives high concepts direct access to all
lower memories, but its categorical teacher reruns all32 cells for every
alternative. Most repeated lower work cannot depend on a late decision.
This is an implementation cost, not an architectural ceiling on credit.

At a fixed parameter vector, model state and exogenous race tape, write the
computation as prefix P_r followed by suffix S_r. Changing the winner at cell r
cannot change P_r. Cache the input state, candidate bank, old feedback, committed
lower memories and random-generator state immediately before r. Force the
alternative at r and execute the rest. Future query scores and winners are
recomputed; they are not held fixed. By induction over downstream operations,
this gives exactly the full-replay alternative final logits.

Factual pathwise differentiation remains on the original graph. Alternative
losses are detached when used in the categorical surrogate. Reusing prefix
tensors in the detached replay therefore changes neither objective nor declared
parameter derivative. Cache use is valid only before parameters/state change;
it is not a reusable approximate teacher across optimizer steps.

`sleeping_machines/cached_parent_hierarchy.py` implements this while retaining
all-level memory access, role races, source/pose metadata, relation composition,
content transport, persistent state, delayed top-down conditioning and cell-
balanced audit selection. The existing source-bound files remain unchanged.

## Work consequences and contracts

With C cells, an alternative at r (zero-based) executes C-r rather than C cells.
This is an exact dispatch count, not an equal-cost FLOP or wall-time ratio:
coarse cells generally have fewer sites and more candidate sources. For the
last cell of a32-cell construction, each alternative executes one cell rather
than32; its lower-memory gathers/keys/values still count. Cache memory and
keeping the factual graph for backward must be measured. Earlier alternatives
still require nearly full replays; losing-candidate learning is not free.

32 combined numerical tests pass. New contracts enumerate every valid
alternative at every cell in a six-cell fixture and require bitwise full/suffix
logit equality; require bitwise objective and every parameter-gradient equality;
and check retained-state feedback and controlled source masking. These support
an exact engineering substitution, not semantic generalization or hardware
latency/energy claims.

## Turning this into learning evidence

The new source-bound128-resolution preflight reuses one official FIT recording,
checks real-input forced suffix/full parity and gradients, measures peak RSS and
an update time, and saves that recording for reuse. It gates a64-update learning
probe with32 cells/eight abstraction levels, native final-task credit and
all-level memory. Probe state resets between distinct gesture episodes (the
task is episode classification), but no state is reset between internal
refinements. Every route audit uses its true cell-balanced probability without
importance clipping. Alternative values receive sampled factual path gradients.

The probe asks whether the full construction can learn a bounded FIT dataset;
it accesses no selection, DEV or TEST labels. Initial/final FIT loss and a
controlled top-level raw-read intervention diagnose learning and raw bypass.
This is a memorization gate, not an investor benchmark or evidence of reusable
parts. Exact model/optimizer/torch/audit-RNG checkpoints every8 updates permit
resume without resetting numerical state or selection. Resource/watchdog
admission remains through the six-slot coordinator; no active trainer is killed.
The older unstarted full-replay/no-fit preflight is explicitly superseded by
this stronger exact implementation; old evidence and active jobs are preserved.

After the probe: if it learns, test held-out arrangements and learned source
contributions at fixed observed evidence before scale. If it fails, inspect
importance-weight variance, role collisions and state transport. Adding more
unchanged depth is not the next action. Dense candidate discovery, dense state
writes, raw event scheduling and a full arithmetic/traffic ledger remain work.
