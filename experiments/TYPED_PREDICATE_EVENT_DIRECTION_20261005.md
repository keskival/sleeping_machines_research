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
