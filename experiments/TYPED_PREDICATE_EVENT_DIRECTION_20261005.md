# Typed predicate events for heterogeneous tables — 5 October 2026

User direction: heterogeneous columns should pass learned logical comparisons before unrestricted neural mixing. This is a proposed integrated extension, not a completed benchmark result or a replacement for the current language/FAS jobs.

The current native_tabular_model.py admits feature-ID/value/missingness events and canonicalizes static feature presentation. native_tabular_data.py supplies numeric datasets and FIT-only normalization. Neither implements categorical membership or a schema-constrained predicate front end. Current numeric tabular evidence cannot validate the mixed-type proposal.

## Construction

A schema declares numeric/ordinal/categorical/Boolean/time fields, units and missing/unknown categories. Initial predicates consume one named field: numeric thresholds or intervals, ordinal cuts, categorical equality or membership, Boolean tests, explicit missing/unknown tests. Cross-field comparisons require an explicitly compatible relation, such as elapsed duration or two quantities with the same units. Category identifiers have no arithmetic ordering. FIT data alone determine threshold candidates, category sets and preprocessing.

Predicate outcomes become addressed events carrying field/predicate identity, outcome and optionally a type-specific residual. Early selection does not sum raw values from unrelated fields. Path state composes conjunctions/disjunctions and directs selected events into persistent receiver banks. Separate keys and values allow logical selection to govern which content is delivered. Deeper local neural processing combines representations after meaningful selection; it may also use explicitly typed conditional residuals rather than discarding all within-bin information.

Temporal races implement competing computation/routes, rather than being replaced with a conventional tree. For static rows these are computational delays, not measured physical arrival times. Independent rows reset state. Any order dependence must be declared as computation policy or eliminated; column presentation order is not a physical signal. Retain sparse addressed updates, small messages, persistent state during a row, depth and actual counterfactual route/write credit. The available predicate/receiver bank can exceed the selected path.

Hard comparisons alone do not give threshold gradients. Learn route alternatives by actual counterfactual suffix utility; learn threshold/category proposals using FIT-only candidates and evaluated alternatives, or an explicit randomized margin model with a derived estimator. Do not imply that a hard Boolean threshold is differentiated normally, or that a surrogate necessarily matches hard-route learning. Charge candidate discovery, rejected alternatives, threshold search, optimizer and complete readout; a sparse inference path is not automatically sparse fitting.

## Required evidence before promotion

Numerical contracts: category-ID permutation invariance, missing versus observed zero, unknown categories, unit/threshold transformation equivalence, selected-path support, consistent inference/learning state updates, finite deep credit to unchosen predicate/routes, independent row reset and declared feature-order policy. Integrated small fits must exercise both typed predicates and temporal sparse memory. XOR/conjunction, ordered intervals and category interactions are distinct witnesses; no witness alone establishes general tabular advantage.

Compare the integrated typed extension with the existing numeric event adapter, a type-aware dense neural control and strong tree/boosting controls under identical FIT/DEV/test rows, tuning opportunity and resource boundaries. Use genuinely mixed-type data; banknote and wine numeric columns alone do not test categorical semantics. Report quality, complete fitting work, inference work, available predicates/state and selected comparisons/updates separately. Tree superiority is an empirical comparison, not guaranteed by a weighted-sum argument.

The concrete failure addressed is premature unrestricted mixing of heterogeneous raw fields and the lack of a typed predicate interface. The proposed inductive bias favors featurewise discontinuities and conditional interactions while retaining the event substrate. Thresholding may lose useful continuous information; typed residual delivery is the explicit comparison for that tradeoff. No existing benchmark/source is substituted and no long run is admitted by this note.

## Implemented interface foundation

`typed_predicate_interface.py` now supplies named numeric thresholds, nominal
set membership and Boolean comparisons, explicit missing/unknown channels,
and selected predicate evaluation without reading unselected fields. It maps
comparison margins into two computational delays; a representable strict
ordering repairs exponent rounding near zero, with equality selecting false.
This is a fixed interface, not a learned selector or integrated model. Caller
selection/discovery work is explicitly outside its own comparison count.

Eight stdlib semantic contracts pass, including unit transformations, category
relabeling, missing versus zero, unknown values, type rejection, selected-path
field access and race ordering (including a tiny positive margin). These take
no model fit or numerical training resources. Next implement learned predicate
selection and actual counterfactual suffix credit in the integrated event
model, then run the small fit described above before allocating a benchmark.
No current FAS/language source or job is changed.

## Unification proof and data-scale bridge

The headline question is whether this same temporal/sparse construction learns
usefully from heterogeneous tables as well as token sequences and asynchronous
logs. It tests a common representation and learning substrate across domains.
A tabular win alone is one anchor; independently trained instances do not prove
shared skills.

First integrate the existing typed interface with addressed memory, computational
races and actual alternative-write suffix credit. Use bounded mixed-type
interaction witnesses to verify that all these mechanisms participate in the
fit, including missingness and category relabeling. Then freeze a genuinely
mixed-type dataset protocol and compare trees, a type-aware dense control and
the integrated model on identical splits and tuning budgets. Preserve the
existing numeric-only losses as a separate comparison.

For the small-to-large claim, use nested FIT subsets with fixed DEV/test and
FIT-only preprocessing. Report quality and complete fitting/inference work at
each size, available predicates and memory, selected comparisons and writes,
and seeds. Tune using DEV under a stated equal budget; do not choose an apparent
scaling curve from test results. The next transfer experiment connects typed
comparisons and sequential reasoning through shared trained core parameters,
compares separate-core controls and checks donor retention under matched total
work. This is an explicit research sequence, not a completed result.

## Integrated stage-one implementation (numerical execution pending)

`typed_temporal_model.py` converts fixed comparisons into event IDs by their
computational race winner, then uses the same PackedTokenCore and sparse
counterfactual event kernel as the language member. Receiver selection is
learned; persistent memory, transport delays/rotation, depth, separate keys and
values and small messages are retained. Each independent row resets state.
Canonical full predicate presentation is an explicit static-row policy.

One outcome-independent event/depth/head site is sampled per batch. A nonwinner
is proposed with positive support and its actual write replayed with common
future race noise to the terminal classification loss. Conditional utility
credit trains route probabilities; it does not provide alternative-value
derivatives. All predicates are compared, all receiver keys are scored and
replay learning is extra work. Thresholds and predicate discovery remain fixed
in this first integration; no sparse predicate-reading claim is made.

The bounded pilot tests category relabeling, column presentation, missingness,
independent rows, deterministic learning/inference parity, finite key credit
and a real alternative-write risk change, then fits a synthetic mixed-type
interaction. Queue `curie_typed_temporal_p8d2_smoke_20261005_v1.txt`: 64 updates,
64 FIT/256 DEV rows, one seed, 600s,800MBRSS,8GiB physical memory reserve. It
waits behind the admitted FAS sequence. Static compilation passed; numerical
contracts and learning result are pending. No tabular win or transfer result
is inferred. The fixed P32/256K language cell follows independently.

## Completed first integration; additional exposure queued (6 October)

The64-update witness completed in4.11s. Column-order/category-relabel
invariance, missingness, row reset, deterministic train/inference parity,
finite nonzero key credit and actual alternative-write risk change all pass.
It has6,370 learned parameters, three fixed predicates and four events per row;
each event scores16 receiver keys and performs four selected state updates.
DEV NLL0.758640→0.705243; accuracy remains54.296875%. Integration is verified;
this run has not learned the interaction sufficiently for a tabular headline.

Queuecurie_typed_temporal_p8d2_1024steps_20261006_v1 keeps the construction,
FIT/DEV rows, seed, optimizer and teacher fixed and increases exposure to
1,024 updates. It saves final weights for diagnosis. The first64-update curve
can be compared to the completed pilot because this optimizer has no
step-count-dependent schedule. It waits the admitted language/rotation/utility
sequence and9504MiB headroom;600s/800MBRSS/8GiB reserve. No numerical result
from that pending fit is claimed and no sparse predicate-discovery or learned
threshold claim is promoted.


## Completed witness and typed-composition theory — 6 October

The 1,024-update fixed-predicate fit completed: 100% on 256 synthetic DEV rows from 64 FIT rows, NLL0.0000898102,6,370parameters,52.705s. Its first64updates reproduce the prior pilot and all integration contracts pass. Preserve the earlier pending entries as historical state; this is fixed-predicate integration, not learned threshold discovery or a real-tree win.

[TYPED_SEMANTICS_AND_RACE_COMPOSITION_20261006.md](theory/TYPED_SEMANTICS_AND_RACE_COMPOSITION_20261006.md) formalizes type-preserving event composition, a restricted raw-nominal affine obstruction and the distinction between inference and learning invariance. It preserves joint winner/time and direct factual credit and assigns predicate discovery/threshold learning their own estimator contracts.39stdlib mathematical/interface witnesses pass with maximum finite-difference error1.94e-10. This theory feeds the authorized typed-learning program and investor explanation; it changes no live kernel or queue.
