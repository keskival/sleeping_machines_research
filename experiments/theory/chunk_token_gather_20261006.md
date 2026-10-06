# Credit-window gather: retain temporal learning, remove repeated lexical buffers

The completed64K/P24 fitting ledger records16,382 calls to
`embedding_dense_backward`, producing39,518,888,352 floating output elements.
The separate timing replay puts58.25% of whole fitting wall in backward. The
operator ledger identifies repeated vocabulary-sized derivative buffers;
these logical tensor sizes are not measured memory traffic or energy.

`token_features` currently gathers each event's input column separately. The
lookup is independent of mutable state and random routing. Gathering the
existing token-ID matrix once per credit window, then consuming columnk at
logical eventk, preserves causality and the same forward values. Future lookup
results are never supplied to an earlier event. State, EOS resets, transport,
clocks, races, scored keys, sparse writes, separate keys/values and actual
alternative-write continuation credit retain their original operations.

Standard embedding backward sums the same selected input-column adjoints.
The accumulation order can change floating rounding; whole-model gradient and
AdamW-step contracts therefore precede fitting. This uses ordinary autograd
and does not impose the custom rotation probe's higher-derivative restriction.
No dropout or random draw is hoisted. Inference still reads the same observed
content; lookup/read movement is not declared zero because arithmetic is zero.

The proposed change addresses implementation work, not a model-family limit.
It can reduce the number of full vocabulary-gradient outputs from one per
step to one per credit window. Prediction quality, ordinary fitting throughput
and complete arithmetic savings require executed evidence. Dense readout,
candidate discovery, factual/alternative continuation and optimizer work all
remain charged. Current source kernels and the activeP32 fit are unchanged.

An isolated source clone changes only the lookup location. The guarded probe
uses the selected64K/P24 checkpoint and64 FIT-derived tokens with duplicate IDs
and EOS. It checks output/state/forced-write risk, every parameter gradient and
AdamW update; counts embedding backward calls and output elements; and times
five interleaved forward/continuation/backward repetitions per implementation.
A full small-fit replay and cost record follow only after contracted improvement.
