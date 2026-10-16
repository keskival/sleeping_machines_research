# A coherent local teacher and a three-arm representation decision

16 October 2026. Continue the selected native role-memory actor.

## Exact composition of observable targets

For a child signature c normalized by sqrt(N_c), a parent signature normalized
by sqrt(N_p) obeys

    phi_parent = sum_c sqrt(N_c/N_p) R(omega dot (b_c-b_p)) phi_child.

R rotates each [cos,sin] pair by the observed child-to-parent origin offset,
separately for the two polarity channels. This follows directly from the angle
addition formulas. Empty children contribute zero. Counts recover the original
sum normalization. It is exact for the fixed feature map, not an approximation
or a result about learned native state. The directions include zero frequency,
so the signed counts are retained. Contracts compare direct parent observation
encoding with merged child signatures to1e-12;swapping unequal child signatures
between roles changes the parent;all nonempty child derivatives are live.

Thus every local native memory has a consistent parts-to-whole target, instead
of unrelated auxiliary tasks at different levels. The signature is still an
additive spatial sketch, not a semantic concept or universal sufficient statistic.
Its information retention and downstream usefulness must be measured. Successful
reconstruction alone cannot establish hierarchical task learning. Geometry-based
ownership and missing learned emission/query policies remain explicit gaps.

## Decisive learning comparison

Reuse the best-point acquired continued-relational raw weights (DEV0.686609),
not the carried extension (0.716414). Both prior arms failed the clustered
transfer gate. All three new arms start from identical native/head weights and
identical geometry decoder initialization:

- joint: whole-label CE +0.1fixed local reconstruction;native parameters learn.
- continued: whole-label CE only;native parameters learn;decoders dormant.
- decoder_only: native parameters frozen;only existing final head and geometry
  decoders learn under CE +0.1local reconstruction.

Each gets64paired updates/128new presentations,common TRAIN pair order seed953,
paired common-race tapes,fresh AdamW1e-3/wd.01/clip1,and exact8-update recovery.
Independent scenes reset numerical state;all32raw observations,32native stages,
indexed persistent state,child-role interactions,native temporal races,timers
and local losing-route credit are retained in trainable arms. The frozen encoder
needs no counterfactual parameter credit and uses selected-only evaluation
forward with the same validated stochastic law. This saves paid work,reported
rather than artificially suppressed. These are stationary scenes,not an online
benchmark. No external model is trained.

The primary quality endpoint remains the existing binary affine head fitted
on64TRAIN features at fixed ridge0.01,scored on64reused DEV scenes. Its solver
is not an added decoder architecture. Joint/scalar heads and local reconstruction
are diagnostics. Initial/final feature packets are saved;frozen-encoder final
features must equal initial features exactly. TEST is never model-scored.
Per-level DEV reconstruction compares joint learning with frozen-decoder fitting;
whole-label DEV and native source cuts determine whether preserved information
helps useful composition. Prior gain and joint-versus-continued gain use16
observed-geometry bootstrap groups. A qualified development result requires
positive confidence bounds and emitted-channel dependence,then a fresh
independent composition confirmation. No automatic larger acquisition follows.

The real grounding gate used19.197s and980532KiB for one pair,including two
extra diagnostic gradient traversals per scene. Learning jobs omit those two
traversals;use a3600s watchdog and3MKiB RSS per arm rather than claiming measured
training throughput in advance. Three separate AWS gym slots1/4/6,one thread
per physical core,8GiB host reserve. Charge native candidate/selected/losing
work,all target row tests,phase/trig work,indexed reads,decoder MAdds,optimizer
state and scoring. Equal presentations do not imply equal work.
