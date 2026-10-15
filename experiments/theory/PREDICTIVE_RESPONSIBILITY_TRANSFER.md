# Hierarchy grows by transferring predictive responsibility

15 October 2026. Founder hypothesis: early output relies on frequent low-level
messages; learned higher memories progressively replace redundant detail.

## Three distinct decisions

An output may need (1) direct access to lower memories, (2) a contribution from
higher composition, and (3) fewer lower transmissions. These are separate
parameters and measurements. A fast bypass repairs learning access but does
not prove useful hierarchy. A higher sender may merely relay raw detail.
Persistent lower state stays addressable after emission becomes rare.

Let v_l(t) be a typed, currently published view at level l, transported to the
prediction time. A conservative first construction uses

    z = sum_l g_l W_l v_l,   g = softmax(s),   p = softmax(H z).

This is feature fusion of potentially complementary evidence, not a mixture of
independent probabilities. The implemented gate uses a learned level bias plus
a query from the latest available abstraction. Initial biases favor low levels;
no prescribed annealing or monotonic-depth constraint is applied. At a given
prediction time one summary is read per available level. Lower activity counts
do not replicate gate candidates. The current spatial summary is an arithmetic
mean of views within the root support: this is a restricted access diagnostic,
not a claim that spatial roles are preserved by the bypass. Native role-aware
composition at higher levels remains intact. A learned spatial/part query is
needed before treating the bypass as a general-purpose output architecture.

Writing e = dL/dz and u_l=W_l v_l, the direct score derivative is

    dL/ds_l = g_l e dot (u_l-z).

Thus a higher view gains allocation when its contribution reduces predictive
loss relative to the current combination. Equal transformed views give zero
allocation gradient. A rise in mean gated source level is a prediction of the
hypothesis, not a training objective. Gates can correctly favor low levels for
fine details, or combine levels indefinitely when evidence is complementary.
Content-dependent query derivatives are additional paths, not part of this
partial derivative holding the candidate views fixed.

## Prevent bypass starvation without inventing hidden labels

Main-loss view derivatives contain g_l and can vanish as a gate closes. Add a
small direct supervised loss on each available level's whole-support predictor,
using the same supplied target and shared output head:

    L_total = L_fused + lambda/|A| sum_{l in A} CE(H W_l v_l, y).

The auxiliary derivative does not contain the allocation g_l. This gives a
trainable direct path, not a guarantee of useful abstraction. A level whose
summary discards a necessary relation cannot recover that relation through
more credit. Whole-label auxiliary learning is allowed anonymous supervision;
latent part identities are not provided. The existing parity task has no
single-child label signal, so isolated-child CE would be the wrong objective.
Here each auxiliary predictor sees its complete root support. Even that can
lose bindings through mean pooling; its representational limitation must be
measured rather than blamed on depth. Contextual cross-child prediction and
frozen role/pose reconstruction are next alternatives if that limitation binds.

## Earn silence only after learning value

For a frozen predictor measure conditional lower-source removal

    U_l = E[L(y,p_without_l) - L(y,p_full)].

The implemented value cut holds other gate weights fixed, replacing that
source's fused contribution by zero. It does not rescore routing or remove
clock/address paths. A separate rerouted cut answers a different question.
Correlated levels can each have small unique value; removing both together can
still be harmful. Pair/group cuts must therefore precede suppression. Marginal
utility cannot be summed as independent information or Shapley attribution.

A future transmission controller should compare paid prediction improvement
with message/read work and a service deadline, with coverage of silence and
losing alternatives in learning. No ideal utility oracle is implemented here.
Traffic is not suppressed in the first prototype, so no bandwidth saving is
claimed. A source can be silent while its last published state remains useful.
Expected source level is allocation, not semantic depth or evidence provenance.

## Native implementation and decisive admission

ResponsibilityNativeComposition keeps the complete carried native hierarchy,
its receipt races, indexed state, role interactions, temporal transport and
autonomous generated views. After causal composition it reads published
memories within the root support at the same prediction time. All reads,
projection/head/query MAdds and native carrier transport are charged. No raw
sensor shortcut or future private version is exposed. This first output uses
all-level fusion rather than an unvalidated sparse output race; candidate
selection and actual transmission thinning are missing integrations.

Contracts test absence of count-driven migration, utility-driven upward
allocation, direct credit under tiny allocation, missing-level masks, frozen
source cuts, and all-level native gradients. One real paired optimizer gate
will measure full-resolution resources and learning paths, using only TRAIN
and acquired weights; no DEV/TEST selection or quality claim. Existing paired
carried-view fits finish unchanged. Quality fitting is a later decision after
that comparison and this gate, not an automatic repeated depth sweep.

Success requires better prediction followed by successful conditional/group
lower-source removal, fewer measured transmissions at preserved quality, and
rising *causally useful* high-level contribution. A rising gate-depth plot by
itself is insufficient investor evidence.
