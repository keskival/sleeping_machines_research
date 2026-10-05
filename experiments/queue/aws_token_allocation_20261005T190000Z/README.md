# Paired token allocation experiment — seed 7

The selected approach is small integrated diagnostics before scaling. This
packet occupies slot1 only after the healthy p96 fit finishes; the other
agent's seed6 packet remains on slot3. No incumbent baseline is retrained.

Seven one-job stages: pinned data; exact actual-driver resume contract;
private/local, private/future, shared/local, shared/future fits; stdlib audit.
The four fits use identical GPT2 FineWeb TRAIN8192, DEV2048 starting20971520,
P16/D2/H2/U4, 128 updates, lanes8, chunk16 and evaluations every32 steps,
seed7. Public benchmark validation first10485760 is never scored here.
Sharing ties input/output/gate/control maps while keys and state stay private.
Future credit uses one bounded alternative-write suffix replay per chunk;
factual winner and sampled local-value proposals remain sparse.

This 2x2 tests an interaction: does sharing make actual future-write credit
more useful? Same presentations are a mechanism comparison, not isoFLOP
supremacy. Replay, all key scoring, readout, backward and AdamW remain in
measured fitting wall; exact FLOP accounting is a prerequisite for scaling
promotion. Width/depth/readout/time cadence remain selectable family choices.

Readouts: initial/final/best DEV NLL, train gap, receiver usage and gradient
exposure, entropy, effective memory rank, capacity split, fitting and entire
stage wall, RSS. The audit requires matching source/data/settings, complete
trajectories and equal development-selection cadence. A .02NLL improvement
is a preregistered practical pilot gate, not a statistical significance claim.
No automatic promotion: independent seed, complete fitting FLOPs and a
larger-data capacity comparison remain explicit gates.

The wrapper preserves the full final recovery checkpoint and embeds the
validation-selected model in it, so automatic publication saves both exact
continuation and the weights associated with reported best DEV.

Conditional next stage: choose a coherent member using both seed packets;
measure complete fitting work; then cross width/data on64K,256K,1M tokens
with fixed passes and cadence, retaining a reserved4M point. Pool/activity
sweeps remain separate from width. No coefficients or scaling win are
claimed by this packet.
