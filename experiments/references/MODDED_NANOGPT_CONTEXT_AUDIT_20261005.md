# Pinned published Transformer context and target protocol

Source: complete source embedded in `modded_nanogpt_20250126_0bdd5ee9.txt`,
SHA25662a81a281be714d90e183c95b6190643940626f4dd9a9ba37b02e6ed704118ab.
Repository revision4ea6b937337a4889b8cfe3f38a93d120048d8f71.
No model/data loader from this log was executed. This audit reads its source.

## Exact declared scoring boundary

Validation resets its loader on each evaluation and scores10,485,760targets.
With8ranks and262,144positions per rank sequence, that is5batches,40fresh
sequences. The first validation shard supplies inputs at positions0through
10,485,759 and next-token targets at positions1through10,485,760inclusive.
Sequence state is not carried across rank sequences or validation batches.
The native benchmark evaluator must reproduce these sequence starts, targets
and target count, resetting numeric state at each sequence start.

Document IDs are cumulative counts of observed EOS50256inside each sequence.
An observed EOS starts a new document before its content is processed. The
last preceding token may still predict EOS using its preceding context. Native
EOS-before-observed-input resets match this information boundary. Every target
is scored; no route decides the validation denominator.

## Attention support and actual architecture

Attention blocks are128tokens. Sliding long-window count increases over the
1770update schedule, finishing at1792tokens=14blocks; the short window uses
7blocks. Within the current block, token causality still applies. Thus14/7
blocks are support bounds, not exact token lookbacks at every query.

The source declares12model blocks,6heads,width768,vocabulary50,257. Block7
has no attention; it still has its local MLP. Encoder mask order is
long/short/short/short/long/short, then reversed for the decoder. Consequently
the11attention blocks comprise3long and8short windows. Layers compound
receptive fields;1792is not a whole-network context limit. In an uninterrupted
document, a derived block-graph upper bound reaches87earlier blocks plus the
current causal block: up to11,264input positions, bounded by sequence start and
EOS. This is a support derivation, not measured attention utilization.

The native comparison should share the available sequence and document
boundary; it need not copy the Transformer’s internal attention policy or cap
its persistent representation at1792positions. Its actual history/state work
and query policy must be declared and charged.

## Data/presentations and score reuse

Train batch size is8×49,152=393,216targets.1770updates present695,992,320targets.
The loader moves to the next100M-token shard when the next full batch plus
one lookahead token would exceed it:254batches=99,876,864targets per full shard.
Six full shards and246batches of the seventh give the total above. A native
same-data comparison must follow the same admitted training population;
repeated fitting of a different single shard is a different data recipe.

Published final validation NLL3.2774 is reused. Intermediate logged scores
have their then-current window and optimization schedule; treating them as
separately tuned smaller-budget baselines would be incorrect. The log declares
`save_checkpoint=False`: this record supplies a published result/source, not
released weights to rescore. Its validation is a public benchmark validation
score, not an untouched official test score. Development here uses a separate
interval; public targets remain reserved until a configuration is selected.

The log does not record token-file content hashes. Current native data files
are pinned to the published FineWeb/GPT-2 distribution and their SHA256 values;
historical byte equality cannot be inferred from matching filenames alone.
Preserve this provenance distinction in the final comparison.
