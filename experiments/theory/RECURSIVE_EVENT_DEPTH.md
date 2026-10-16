# Recursive event depth: lower-level memories as asynchronous causal data for deeper levels

15 October 2026, Curie. User question: how do deeper levels use lower-level memories as asynchronous causal data, so that a
deep level is as capable with that data as our shallow models are with raw input? Builds on
CAUSAL_ASYNCHRONOUS_DEPTH_SYNTHESIS.md (event transducers, availability vs physical time, I(Y;W|Z), silence as
information; AWS) and PART_WHOLE_COMPOSITION.md (observed part-whole levels, receptive fields, tree read).

## 1. The principle

Our shallow models are strongest on marked event streams: they model *when* the next event comes and *what* it is with an
exact likelihood, decaying/rotating memories clocked by elapsed time, races, identity-addressed slots and silence-aware
supervision. They win where those are the right inductive bias (EasyTPP, TGB, FAS sealed binding, B10 online market) and
the keyed member's identity-addressed read is its decisive language mechanism (6.193 → 6.009 at 1M tokens).

Recursive event depth makes every level an instance of that same learner, applied to the event stream emitted by the
level below. Depth then inherits, by construction, the competence the family has on raw events, instead of asking a deep
layer to do something different from what our models do well.

## 2. Derived streams

**Definition 1 (derived causal stream).** Level k−1 emits events e_i = (a_i, c_i, v_i, τ_i): availability time a_i
(observation time of the triggering input plus computation delay δ ≥ 0), a discrete mark c_i ∈ V_k (identity), a
residual vector v_i and the physical gap τ_i to its previous emission. Causality: (c_i, v_i, τ_i) is measurable with
respect to the input filtration F at the triggering observation time, and the level-k state at query time q depends only on
events with a_i ≤ q. Silence (no emission up to q) is itself observed at q.

**Proposition 1 (closure).** If level k−1 emits a derived causal stream and level k is a causal event transducer on it,
level k's emissions are a derived causal stream with respect to the original filtration F. Hence any depth of such levels
is causal and needs no global clock: each level runs at its own event rate and the prediction at q uses the latest
available state of every level, stale by at most its computation delay. (Composition of causal maps; AWS §3 gives the
execution conditions for shared state.)

## 3. When is a deep level as capable as a shallow model?

Split competence into what the model class can represent and what it can learn from the data it sees.

**Proposition 2 (expressivity transfers).** The family's capacity results are stated for arbitrary marked point
processes: a race of exponential and defective clocks approximates any continuous inter-event density on compact gap
ranges, and hazard-weighted mark mixtures approximate any continuous time-dependent mark law (theory note 158, §§449-456).
A derived stream with discrete marks and positive gaps is such a process. So the same class at level k can represent any
law of the level-(k−1) stream that it can represent for raw input. This transfers representational scope for the derived stream under those assumptions; information discarded by emission, finite-data learnability and downstream usefulness do not transfer automatically.

**Identity is the condition that is easy to miss.** The mechanisms that make our shallow models strong are identity
addressed: per-mark slots and clocks, the predecessor message keyed by the previous mark, exact-match bonuses, sufficient
statistic counts. They require marks with identity. A deep level that receives only continuous vectors loses them. Give
every emission a discrete mark: the winning slot of a concept race over K_k learned keys (PART_WHOLE_COMPOSITION §5; the
concept memories already implement it), keeping the residual v_i as payload. With identities, the deep level runs the
identical keyed temporal-memory member on "concept tokens": embeddings of concept ids, keyed predecessor reads by concept
identity (induction over concepts: after concept A came B), per-concept clocks.

**Proposition 3 (learnability: data per level).** A level sees n_k = ρ_k n events when the token level sees n, with
emission rate ρ_k (measured: word 0.90, clause 0.11, sentence 0.069, line 0.023, document 0.0015 per token). Its estimation
error behaves as for a shallow model trained on n_k events of a stream with vocabulary K_k. Matched competence therefore
requires sizing each level to its data: K_k and the level's parameters must scale with n_k, not with n. Rule of thumb
from the token level (4M tokens, 50k vocabulary, competitive): events per mark of the same order, e.g. at 4M tokens
the sentence level (~280k events) supports K ~ 10^3-10^4 concepts; the document level (~6k events) supports only a few
dozen. Upper levels need more total data than the token level to be equally trained, which is why scaling data matters
more as depth grows.

**Proposition 4 (usefulness).** For Bayes-optimal unrestricted log-loss decoders in nats, the gain from adding S_k to S_<k equals I(Y; S_k | S_<k), the information in its
state about the target beyond the lower states (AWS §5). Since S_k is a causal function of the input history, this is
positive only through history the lower states no longer hold or cannot make accessible to the decoder: older content
(S_k's receptive field is N × span tokens) or structure that is cheaper to read from S_k. Recursive training (§4) teaches
S_k to predict the lower stream's future at its own horizon, which is exactly where lower states with short horizons
leave conditional information.

## 4. Training: the recursive likelihood

**Definition 2.** The level-k objective is the negative log-likelihood of the level-(k−1) derived stream under the
level-k model: for each lower emission, −log p_k(τ_i | S_k) − log p_k(c_i | τ_i, S_k), plus the survival term for observed
silence up to each query. At k = 1 with token marks this is the language-model loss; at k = 0 on raw events it is the
TPP loss our shallow models already train with. Every level receives a dense, exact learning signal of the kind the shallow
model gets, instead of only the weak top-down credit through the readout.

Three failure modes and their conditions:
1. **Collapse.** A learner that predicts its own codes can lower its loss by making them predictable (AWS warns of
   balanced but irrelevant codes). Condition: the level-k loss does not backpropagate into the level-(k−1) code
   assignment (stop-gradient at the mark). Codes are shaped only by level k−1's own objective and by the base task;
   level k must learn their true law.
2. **Drift.** The lower level keeps learning, so the stream's law moves. Two-timescale condition: code assignments
   change slowly relative to the upper level's adaptation (exponential-moving-average codebooks or commitment terms), so
   the upper level tracks a quasi-stationary target.
3. **Irrelevance.** Predicting the lower stream well is useful only if that stream carries task information. It does by
   construction when level k−1 is itself trained by the base task and its own recursive likelihood (codes summarise what
   predicts the future).

**Local and asynchronous learning.** With the recursive likelihood plus the one-hop readout credit, a level needs no
backpropagation through the levels below: it is trained like a shallow model on its input stream. Levels can learn on
separate cores at their own event rates, with memory per level rather than for the whole stack. This is the family's
local/asynchronous learning advantage applied to depth.

## 5. Construction (next integration)

On the concept hierarchy (`experiments/r1_concept_hierarchy_lm.py`):
- **marks:** the top-1 concept slot of each level's concept race at each emission (stop-gradient), residual = the
  level's whole vector;
- **recursive heads:** at every level-k event, predict the next level-(k−1) mark (softmax over K_{k−1} slots) and its
  gap in tokens (race of clocks), with the survival term at queries; weight beside the next-token loss;
- **concept-identity reads:** the level's keyed read keyed by concept identity and the predecessor concept (induction
  over concepts), alongside the content-keyed read;
- **codebook stability:** EMA updates of concept keys and a usage floor (dead slots re-seeded from recent wholes).

Contracts before admission: stop-gradient at marks (the level-k loss leaves the lower code assignment's gradient
exactly zero), causality of marks and gaps, survival terms equal to the integrated hazard, EMA codebook equals its
recurrence, init equal to the concept hierarchy without the heads.

## 6. Predictions

- **P9a (competence).** Trained with the recursive likelihood, each upper level predicts its lower stream's next mark
  better than count references fitted on the same derived stream (unigram/Kneser–Ney over concept ids): the R1 gate-2
  result repeated one level up.
- **P9b (usefulness).** Adding recursive heads raises the document-context value (word-initial, positions 256-1024)
  over the concept hierarchy with bag-of-future targets alone.
- **P9c (identity).** Concept-identity reads add to P9b, as the token-level identity read did for the keyed member.
- **P9d (local learning).** Level-local training (recursive likelihood + readout credit, no backpropagation across levels)
  reaches the end-to-end quality within 0.01 nats at lower peak memory.

## 7. Type closure: every layer receives and emits the same interface

What a layer hands to the next must have the same form as what that layer received, for the asynchronous data *and* for
the memories. Otherwise a deep layer faces a different kind of problem from the one our shallow models solve well.

**The interface.** I = (E, Λ, Σ):
- **E, the stream:** causal events e_i = (a_i, c_i, p_i): availability time, identity mark c_i ∈ V, payload p_i (possibly
  empty), with the gap to the previous event and observed silence.
- **Λ, the lexicon:** an addressable memory of the vocabulary, Λ: V → (κ_c, λ_c), a key used to *choose* marks and a value
  used to *read* them.
- **Σ, the store:** the events emitted so far, each entry (a, c, p, key, parts) readable by identity match, predecessor
  identity, content race and age, plus one live entry for the event in progress, whose parts pointer lists the entries
  of the store below that compose it.

The token layer is the base instance: E = tokens with unit times and empty payload, Λ = the vocabulary embedding table
(tied with the output embedding), Σ = the earlier tokens of the lane, read by the keyed member's identity-addressed slots
and predecessor message.

**A layer** is a causal map L: I_k → I_{k+1} that consumes E_k in order with input x_i = λ_{c_i} + P p_i + g(gap) (the
lexicon read replaces the embedding lookup), reads Σ_k with the same keyed race reads the token layer uses on earlier
tokens, keeps its own temporal state, and at its boundaries emits e' = (a', c', p') into E_{k+1}, writes it into
Σ_{k+1} with its parts pointer, and keeps the live entry of Σ_{k+1} equal to its provisional whole.

**Proposition 5 (type closure gives depth by induction).** If L maps instances of an interface family F to instances of
F, any stack of layers is well typed, every layer faces an in-family problem with the full mechanism set (lexicon,
identity reads, predecessor binding, temporal memories, races, recursive likelihood), and the same layer code serves
every depth with separate parameters. The token layer is L applied to (tokens, embedding table, earlier tokens).

**Choosing marks (emission).** In the token layer marks come from the data; above it, the layer chooses them. The rule:
c' = argmin_c ||W p' − κ_c|| (a race over the lexicon keys with zero noise), keys κ updated as exponential moving
averages of the wholes assigned to them, dead keys re-seeded from recent wholes. The choice carries no gradient
(stop-gradient at marks, §4), so the upper mark-prediction loss cannot directly collapse the lower vocabulary through its label path. Lower training, initialization or EMA dynamics can still collapse assignments and must be measured. The value table λ is
learned by gradient as the next layer's input embedding: keys choose, values are read, as the token layer's tied output
and input embeddings. The payload p' (the whole vector) carries gradient to the emitting layer, so marks give identity
and payloads give detail.

**Similar in form, measurably.** Type closure is structural; statistical similarity must be measured. The token stream
has the properties our mechanisms exploit: identities recur (51% of word-initial targets already occurred in their
1024-token window), usage is heavy-tailed, events are bursty in time. A derived stream is "similar in form" when its marks
recur within documents at comparable rates, its usage is heavy-tailed rather than uniform or collapsed (effective
vocabulary between ~K/50 and ~K/2), and its identity reads carry value (the upper keyed read's identity bonus learns a
positive weight). These statistics are recorded for every level of every trained stack.

**Contracts.** Type closure (each layer's output satisfies the interface: causal availability, marks in range, parts
pointers resolve into the store below, live entry equal to the provisional whole); the base instance reproduces the
token layer exactly; the EMA codebook equals its recurrence; marks are gradient-free; causality of every level's reads.

**Prediction P9e.** A stack of type-closed layers (layer code shared across levels, parameters separate) beats the
concept hierarchy of equal parameters at equal fresh tokens, and its derived streams meet the similarity statistics.

## 8. Additivity: what residuals need across clocks, levels and the mark interface

A residual connection x' = x + f(x) works because x and f(x) live in one vector space with one meaning, so the identity
path carries everything forward and each layer learns a correction. Across asynchronous, type-closed levels that space
must be built explicitly. Five conditions:

**(R1) Linear transport between times.** States issued at different times are added only after transport to a common
reference time by the family's complex-diagonal propagator Φ(Δ) (decay and rotation per mode). Φ is a linear semigroup,
Φ(s)Φ(t) = Φ(s+t), so sums of transported contributions compose consistently no matter when or in which order they
were issued (AWS asynchronous residual algebra: compare like state at like reference time).

**(R2) A linear part-to-whole identity path.** The whole of a span is
w = Σ_j Φ(n − j) W x_j + g(·): the ordered integration of its parts is linear (a superposition in which each part's
position is a rotation phase), and the nonlinear whole map g is the residual correction. Upper levels therefore start
from an additive summary of their parts rather than a re-encoding, the property that made pre-norm residual stacks
trainable. Hard selection (races, top-k, quantization) may act on the correction branch and on addressing, never on the
identity path: the race stack's stage output was only the winning receiver's value, a selection without an additive
path, which is the composition error measured in RESIDUAL_DEPTH_20261015.md.

**(R3) Mark plus residual payload.** At the interface a whole is decomposed additively into its mark's key and the
quantization residual: w = κ_c + r with r = w − sg(κ_c). The next layer's input is λ_c + P r. With λ initialised to κ and
P to the identity, the input equals w exactly at initialisation: the interface is an identity map plus learned
corrections, as residual quantization makes it. Marks carry the coarse identity (for identity reads and predecessor
binding), the payload carries the residual detail, and their sum is the whole.

**(R4) Additive top-down readout.** Every level contributes to the prediction through a zero-initialised linear map of
its transported state, y_t = h_t + Σ_k A_k Φ(t − a_k) u^k: contributions add in the readout space, each level one hop
from the loss, the stack at initialisation equal to its base.

**(R5) Scale control without breaking additivity.** Normalisation goes on the input of correction branches (pre-norm),
never on the identity path; decay keeps integrated sums bounded (geometric weights). Superposition has finite capacity:
an additive summary of n parts in d dimensions keeps them separable only while d grows with n log K (the capacity
bound of phase-coded superposition), so wholes with many parts need width at their level or exact recovery through the
store. The design uses both: the additive whole carries the gist, the store's parts pointers (tree read) recover detail
exactly.

**Contracts (additivity).** Transport composition Φ(s)Φ(t) = Φ(s+t) to rounding; with g, the corrections and A_k at zero,
each level's whole equals the linear integration of its parts and the interface input equals the whole exactly (R2, R3);
the stack at initialisation equals its token-level base.

Implementation qualification: [Native recursive race credit](RECURSIVE_RACE_CREDIT.md) supplies partition-invariant survival accounting, live published-memory integration and the semantic relevance tests. The current categorical length head is a bounded length prediction diagnostic; it is not the native competing-risk/silence implementation.
