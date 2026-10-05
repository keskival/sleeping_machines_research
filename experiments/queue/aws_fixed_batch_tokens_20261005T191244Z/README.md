# Fixed-batch 8K continuation

The completed 2K batch64 recipe improves development NLL by 0.087–0.094
in both seeds. Retain credit16 and credit64: their selected ranking changes
between seeds. This packet runs both seeds at 8K using the existing contracted
horizon engine, identical 16,368 fitting targets, 32 optimizer updates and
evaluation every eight updates. Initialization remains eligible for selection.
GPT-2 FineWeb, persistent sparse temporal state and actual alternative-write
credit are retained. No architectural substitution.

Slot1 loads this packet after its existing allocation packet, whose verified
data and actual-driver resume stages are prerequisites. Three occupied slots
continue unchanged. Each fit uses run_safe.sh, one CPU thread, RSS 1,200,000KiB,
VMS 6,000,000KiB, timeout300s and the mandatory8GiB host reserve. Prior small
fits used700,000KiB caps; this larger reservation follows the original8K packet.

Original slot3 addenda are preserved. Their alphabetical load order places
capacity/event/horizon before their producers; the frozen scheduler marks
missing prerequisites blocked rather than deferring them. These new names
provide an independently runnable fixed-batch comparison after slot1 setup.
This receipt is not physical training admission. Select from completed dev
trajectories; require at least0.02NLL gain against initialization for the pilot
promotion gate. Compare horizon ranks across seeds. Before scaling, audit
whole-fit and per-target work, inference work and context contributions.
No pending quality, FLOP advantage or benchmark win is asserted.
