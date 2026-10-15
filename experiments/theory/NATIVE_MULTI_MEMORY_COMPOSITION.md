# Native multi-memory composition and a faithful learning gate

15 October2026. Implemented extension; learning outcomes pending.

## From notification streams to deliberate composition

The native hierarchy already integrates lower emissions into persistent parent
state. A higher concept must also be able to gather several retained lower
representations after the notification stream has quieted. Replacing this with
one more linear pooling network would discard the event-native advantage.

`NativeMemoryComposition` extends the same object primitive with one bounded
read round. Each active lower-level memory in the root's geometric support
answers an indexed query. Its answer contains the same native representation
at query time, with source level, relative pose and query type. The root consumes
these answers through its own persistent native races. It then generates the
answer representation at a native clock's firing time. There is no additional
encoder/attention model and no direct raw-input shortcut in this read round.

All active lower memories are queried in this development protocol. This is
strong access coverage, not a learned sparse query policy. Discovery/reads,
parent updates and losing-value learning are charged. Learning which subset to
query and when to stop remains a concrete engineering objective.

## Why this can compose rather than just retrieve

For persistent parent state H and typed lower views V_j(t), native receipt makes
H_(j+1)=F(H_j,V_j(t),time_j,race_j), followed by generated view R(H_final,t_emit).
The parent can combine different lower memories and prior evidence through
nonlinear gates, temporal memory and independent head races. Values retain source
and pose relations; reads can cover opposite spatial corners without the old
fixed raw stencil. The same rule parameters are shared across spatial objects.

None of this proves semantic abstraction. The parent might rely on one low
level, memorize training prefixes or exploit timing idiosyncrasies. Useful
hierarchy requires learned constituent reuse and held-out composition, plus
source interventions that show intermediate concepts have unique value.

The receipt ordering in this first read round is canonical source-level/address
order. It is computational order, not a fabricated observed chronology; message
kind explicitly marks query responses. Ordered composition is available, but
permutation invariance is not claimed. No new sensor observation is introduced.

## Race semantics and clocks

The synchronous model's failed learning gate improved0.010006 NLL under fixed
stochastic races but worsened0.026493 under argmax. Therefore the native gate
keeps exponential races in both training and evaluation. Training materializes
losing proposals for its local counterfactual teacher; evaluation computes only
the selected value. Under the same exogenous tape, complete32-stage indexed-
pooling/readout logits and completion time match bitwise in the contract and
must match on the real preflight input before fit admission.

The new wrapper also fixes a scheduler precision defect: converting a Python
float through default float32 can make an event at the current float64 time
appear to arrive in the past. Queue keys now explicitly preserve float64. This
is a numerical-causality repair, not a relaxation of the no-past-events rule.

Terminal readout uses the memory view at the actual timer tensor, preserving
its interior clock derivative. The result carries a virtual availability time.
The reference scheduler remains chronological CPU execution; virtual times are
not physical hardware latency. Native local losing-route credit is not exact
credit through all alternative event orders or message counts.

## Learning admission and what success means

Data: the first32 raw events in the first second of every annotated gesture in
one official FIT recording, selected without DEV/TEST. Times are relative to
each gesture start; absolute recording times and class IDs are absent from
model inputs. Prefixes are stably ordered by event time. Input hashes exclude
labels and detect identical inputs assigned different labels. Distinct hashes
are necessary but do not prove that useful class features survive the model.
This is a deliberately bounded memorization/learnability gate, not a full
DVS Gesture benchmark or object-part semantics result.

The real preflight verifies finite gradients, stochastic train/eval bitwise
parity, exact intended batch1 RSS/time, and prefix identifiability. Only then
admit64 AdamW updates on the complete native32-stage composition, including
multi-memory queries and generated readout. Event states persist throughout each
episode; independent gesture episodes reset. Global race RNG and separate
sampling RNG, optimizer/model and step are checkpointed every4 updates.

Two fixed evaluation tapes are shared across all episodes, not assigned as
per-item noise identities. Evaluation restores training RNG afterward. Report
initial/final FIT loss, accuracy, label entropy/majority predictors and association:

A = E_i E_(Y~empirical label marginal)[-log p(Y|X_i)]
    - E_i[-log p(y_i|X_i)].

A is zero for an input-independent predictor. Positive A shows training-label
association beyond merely learning marginal class frequencies; it is not
held-out evidence. Lower loss alone is insufficient to declare feature learning.
Work includes fitting and both scoring passes; full arithmetic/traffic/optimizer
accounting remains distinct from native event counters.

After this gate, either investigate transport/credit failure or test source
contributions and held-out compositional reuse. Do not launch an unchanged depth
sweep or interpret one restricted diagnostic as a model-family verdict.
