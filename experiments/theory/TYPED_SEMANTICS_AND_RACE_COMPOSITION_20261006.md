# Typed semantics, legal composition and temporal learning

6 October 2026. Proposed theory and derived contracts, extending the existing
[typed event direction](../TYPED_PREDICATE_EVENT_DIRECTION_20261005.md) and
[joint winner/time credit](typed_threshold_joint_credit_20261005.md).
No live language/FAS kernel is changed; no new training is admitted.

## The thesis

**Respect each field's meaning before composing evidence.** A typed interface
can preserve the conditional structure that makes trees effective while
delivering evidence to deep, persistent temporal representations. Weighted
sums become appropriate after values are mapped into a declared common space.

The failure being addressed is premature identification of heterogeneous raw
encodings with interchangeable coordinates of one vector. A category code,
temperature, duration and missing-value sentinel do not inherit compatible
addition, order or distance merely because they are stored as numbers.

Calling all neural sums “illegal” is too broad. A coefficient can convert units;
one-hot or categorical embeddings can give nominal data a valid representation;
feature-specific normalization and maps can produce compatible latent vectors.
Dense models can learn these functions. The question is which semantic
constraints are guaranteed by construction and which the learner must discover
from data. Numerical validity and semantic justification are different tests.

## 1. Data lives in a product of typed spaces

A row belongs to X = X_1 × ... × X_d, extended with explicit missing/unknown
states. Each field has admissible operations and changes of representation:

| Field | Meaningful primitive | Representation change to preserve |
| --- | --- | --- |
| Numeric quantity | Within-field threshold, interval, normalized residual | Positive affine unit change, with threshold/scale transformed |
| Ordinal | Order cut, rank interval | Strictly increasing relabeling; magnitude is not implied |
| Nominal category | Equality, set membership, declared learned embedding | Bijective relabeling with schema/membership/embedding rows relabeled |
| Boolean | Logical test | Boolean semantics, no numerical category coercion |
| Missing/unknown | Presence or explicit unknown branch | Remains distinct from observed zero and known categories |
| Compatible quantities | Explicit typed relation, such as duration difference | Common units or a declared conversion |

Not every mathematical invariance is desirable. Celsius/Fahrenheit is a change
of representation; changing a sensor timestamp relative to another sensor can
change causal meaning. A schema must declare the distinction. Nominal category
renaming preserves meaning only when the associated semantic maps are renamed.

## 2. A concrete obstruction for raw nominal sums

Take a raw affine gate h(c,z) = w*c + b(z), where c is an arbitrary numerical code
for a nominal category and z is held fixed. Require invariance under every
permutation of category codes, without changing this gate. Swapping two distinct
codes c_1,c_2 gives w*(c_1-c_2)=0, hence w=0. The invariant raw affine gate must
ignore the category. It cannot both use that category and satisfy this contract.

This is a restricted proposition about a fixed raw affine gate, not a theorem
against MLPs. A relabeled embedding table preserves nonconstant category
information; a nonlinear model can learn a categorical function from codes.
The design advantage is avoiding arbitrary code geometry as the default bias.

Likewise, adding 3 degrees to category ID 17 has no declared semantic result.
Adding learned vectors representing their evidence can be meaningful. The
interface supplies that interpretation rather than pretending the raw columns
already share it. Even incompatible physical quantities can enter a legal
learned sum through coefficients with the required units.

## 3. Composition theorem for a typed event substrate

Let g be an admissible fieldwise representation change, and gθ its action on
interface parameters. Let adapter A satisfy A(gx;gθ)=A(x;θ), including event
identity, content and computational timing. Consider a causal event core with
the same initial state, semantic event order and random draws. If its transitions
depend only on those events and its own state, its entire state trajectory,
selected writes, output and inference arithmetic trace are identical.

**Proof:** the initial states agree. Equal next events, equal preceding state and
equal random draws produce equal races, selected destinations, messages and next
state. Induct over events and depth. The argument includes clocks and persistence;
it is not restricted to a feed-forward classifier.

For numeric predicates m=(x-t)/s, s>0, transformation x'=a*x+b, t'=a*t+b,
s'=a*s for a>0 preserves m. Thus delays exp(-m), exp(m), their ordering and any
margin-dependent payload are unchanged in exact arithmetic. The implemented
fixed predicate interface has finite-precision contracts, including a strict
ordering repair when exponentials round to equal values.

For arbitrary strictly increasing ordinal changes, the threshold outcome is
preserved but the numerical margin generally is not. Use order/rank evidence
when that stronger invariance is required. Do not extend the affine result to
arbitrary monotone transformations of margin-sensitive clocks.

**Learning corollary, with additional conditions:** equivalent training
trajectories require equivalent preprocessing, parameter initialization, route
noise, credit and optimizer coordinates. Raw threshold gradient scales as 1/a,
so an unchanged raw-unit SGD step is not unit-equivariant. Normalized coordinates
or transformed optimizer rules are required. An inference symmetry alone does
not prove training symmetry.

## 4. Trees, race decisions and deep state

A decision tree composes named predicates along a conditional path. A temporal
member can express that computation if it can store node identity, choose the
node's typed predicate, route to a child and stop or retain an absorbing leaf.
Induction over tree depth gives the same output. This conditional construction
explains tree-compatible expressivity; those capabilities must be implemented
and tested before claiming tree-like sparse feature access.

The temporal construction can also deliver a typed residual and update shared
persistent representations. It can combine a condition with text, earlier events
or action state rather than restricting all downstream reasoning to leaf lookup.
Its promise is conditional selection plus deep reusable state, not superiority
over every tree on every typed problem.

Predicate-only compression can destroy useful information: x=-2 and x=-1 both
produce “below zero”, so a bit-only adapter cannot reconstruct a target equal
to x. Type-specific residuals or finer predicates are necessary for such targets.
Smooth compatible linear functions can favor a dense map. Conditional, irregular,
featurewise functions motivate the comparison-first bias. Schema correctness
does not by itself determine the best statistical model.

## 5. Credit must include computational time

For a stochastic numeric comparison use rates r_true=exp(m), r_false=exp(-m),
Λ=r_true+r_false. Winner W and first time T have joint density
r_W*exp(-Λ*T). Existing derivation gives

    d/dm log p(W,T) = sign(W) - T*(r_true-r_false)
    d/dt log p(W,T) = -[sign(W)-T*(r_true-r_false)]/s.

For the closed continuation witness loss=A_W+B*T,

    R = p*A_true + (1-p)*A_false + B/Λ
    dR/dm = 2*p*(1-p)*(A_true-A_false)
             - B*(r_true-r_false)/Λ²,
    p = r_true/Λ.

The first term credits branch choice; the second credits computation time. When
branch losses are equal, choice credit vanishes while time credit can remain.
Actual alternative suffix evaluations with matched future noise can estimate
choice utility. They do not remove time credit or any direct factual derivative.
Use one explicitly derived estimator boundary; do not double-count the same
time derivative. Deterministic threshold inference is a separate policy from
stochastic training and needs its own integrated comparison.

Selecting which predicate or category subset to evaluate is another discrete
choice. Threshold learning, predicate discovery and receiver-route learning are
different mechanisms. Our completed typed witness learns receiver routes with
actual alternative writes; its predicates and thresholds remain fixed.

## 6. Capacity and work contracts

K available predicates with a selected path of k comparisons can require k
semantic field reads, but discovering that path may still cost O(K). Dense
key scoring, discarded candidates, alternative continuation replays and optimizer
updates remain charged. No zero discovery or sparse fitting follows from hard
selection. Report available predicates, inspected fields, scored keys, selected
writes, replayed events and complete fitting/inference work separately.

## 7. Position among established approaches

[Grinsztajn et al.](https://arxiv.org/abs/2207.08815) find strong tree performance
in their medium-sized tabular benchmark and investigate irrelevant features,
feature orientation and irregular functions. These are empirical inductive-bias
results, not a proof that heterogeneous addition explains every gap.

[FT-Transformer](https://arxiv.org/html/2106.11959v2) already uses per-feature
numerical maps and categorical embedding lookups before attention; its study
finds no universally superior neural/tree solution. It is a meaningful typed
neural control, not an example of arithmetic on raw category IDs.

[CatBoost](https://arxiv.org/abs/1706.09516) explicitly treats categorical data
and training leakage. [NODE](https://arxiv.org/abs/1909.06312) combines tree-inspired
decisions with deep representation learning. Comparison layers and neural/tree
hybrids are established ideas. Our proposed contribution is their composition
with computational races, persistent addressed state, deep actual alternative
credit and a common event interface spanning tables, sequences and asynchronous
experience. General novelty or benchmark dominance is not inferred here.

## 8. Evidence and falsifiable next tests

The integrated synthetic witness reaches 100% on 256 DEV rows from 64 FIT rows,
with 6,370 parameters, one seed and fixed predicates. Category relabeling, column
presentation, missingness, row reset, train/inference path, nonzero key credit
and actual alternative-write risk contracts pass. Existing numeric banknote
comparisons favor trees. FAS tree references have not been fitted.

1. Check semantic reparameterization at inference and during learning separately.
2. Implement and enumerate one learned predicate/threshold intervention while
   retaining the computational-time and factual-path credit terms.
3. Compare fixed-bit, typed-residual and learned-predicate adapters on irregular
   interactions and smooth targets; include type-aware MLP/FT and tree controls.
4. Freeze real mixed-type splits, FIT-only preprocessing and equal tuning budgets;
   use nested FIT sizes and full quality/work records.
5. Share the trained core across typed decisions and sequence/event tasks, measuring
   transfer and retention against separately trained controls at matched work.

The accompanying stdlib contracts check the interface invariances, restricted
raw-affine obstruction, legal embedding remedy and joint-time derivatives.
They are mathematical/interface witnesses, not new integrated learning results.
