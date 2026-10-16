# Part-whole composition: how higher-level concepts are represented, inferred and trained

15 October 2026, Curie. User direction: depth is the key to efficient learning and hierarchical representations; the
failure is our composition, not depth. Start from what the data is and how wholes are refined from low-information parts.

## 1. What the data is (`results/diagnostics/curie_part_whole_structure_20261015_v1.json`)

A GPT-2 BPE token is a low-information fragment (" Bar", "cel", "ona", ","). Text is a nested part-whole structure:
pieces → words → clauses → sentences → lines/paragraphs → documents. Measured on fineweb_val (2.1M tokens): 1.38 tokens
per word, 17.1 words per sentence, 1.9 sentences per line, 696 tokens per document. The tokenizer marks a word start by a
leading space; punctuation, newlines and the end-of-text token mark the larger boundaries, all causally observable.

Where the prediction problem is, on the flagship protocol (8192 targets, 1024-token windows):

| Target role | Share of targets | Keyed 4M member NLL | Share of its loss | Unigram NLL |
|---|---|---|---|---|
| Word-initial (choose the next word) | 77.8% | 6.08 | **84%** | 7.95 |
| Word continuation (complete the word) | 10.5% | 5.37 | 10% | 9.21 |
| Punctuation | 9.0% | 2.76 | 4% | 4.00 |
| Newline | 1.8% | 1.64 | 1% | 3.85 |

Language-model quality is almost entirely the choice of the next word. What constrains that choice lives at the slow
levels: the clause in progress (syntax, argument roles), the sentence and paragraph (what is being said), the document
(topic, entities, style). None of our members uses that: the strongest member's history beyond 64 tokens is worth
**0.023** nats on word-initial targets (beyond 16 tokens: 0.172, mostly from nearby repeats), and the race stack's deep
stages contribute ≤0.001 nats each to word-initial targets (stages 4-7 of D8 + reads); their remaining value is on
newline, punctuation and continuation tokens. The race stack even predicts word continuations worse than word starts
(7.44 vs 6.86): it does not assemble pieces into words.

## 2. What a level-k concept is

A latent variable that (i) holds over a span of level-(k−1) units; (ii) makes the parts in that span predictable (the
reason to represent it); (iii) is a function of its parts and their order/roles; (iv) can be estimated causally, a
provisional estimate improving as parts arrive and complete at the span end; (v) recurs (the same word, entity,
construction), so it should be stored and addressable at its own level.

Stacking token-clock stages provides none of (i), (iii), (iv) or (v): every stage sees the same token events, the same
16-token credit window and the same width, so stage d can only re-process what stages below extracted from the same
window (§5 of CREDIT_HORIZON_AND_ADDRESSED_MEMORY.md). The race stack additionally feeds the top stage's output into the
bottom of the next token, which makes each token-to-token transition D stages deep (credit paths ~k·D stages over k
tokens) and destabilised D16 with reads.

## 3. Construction: observed part-whole hierarchy

Level 0 (token): the strongest member's blocks (plan-B pre-norm residual temporal memory + per-layer keyed race read +
MLP), state u⁰_t.

Level k = 1 (word), 2 (clause), 3 (sentence/line), boundaries from observed tokens:
- **Clock.** A level-k unit starts at t when x_t starts a word (k=1: not a continuation of the preceding word piece),
  follows a clause mark `, ; :` or sentence end or newline (k=2), or follows a sentence end `. ! ?` or newline (k=3).
  End-of-text closes everything.
- **Ordered integration of parts (bottom-up inference).** Within the current span, completed parts are integrated by a
  decaying/rotating temporal memory reset at the span start, with lag counted in parts:
  I^k_t = Σ_{completed parts j in span(t)} c^(n_t − n_j) W u^{k−1}_(end of part j). The rotation phase encodes each part's
  position in the whole ("dog bites man" ≠ "man bites dog"); decay lets recent parts dominate an unfinished whole.
  For k = 1 the parts are the word's tokens themselves.
- **Provisional whole.** b^k_t = MLP(LN[I^k_t ; u^{k−1}_t]): the whole as estimated from the parts so far plus the part in
  progress. The prediction at every token sees every level's provisional whole: the next word is chosen before the
  current clause or sentence has closed.
- **Level memory over completed wholes (horizontal recurrence and addressing).** When a span closes, its whole
  v^k = b^k at the span end becomes a level-k event. Level k keeps a temporal memory over its own events (lag counted
  in level-k events) and a keyed race read over its past wholes (keys and values from the wholes): retrieval of
  earlier words, clauses and sentences, not only tokens. u^k_t = b^k_t + R^k_t is what level k passes up and to the
  prediction.
- **Prediction (top-down at use).** The readout mixes all levels nonlinearly: h_t = MLP(LN[u⁰_t ; u¹_t ; u²_t ; u³_t]),
  then the strongest member's tied logits and keyed token-slot bonus. Slow context conditions the choice of the next
  piece through this mixing; it never re-enters lower levels per token, so no token transition is deeper than one level
  step and every level is one hop from the loss.

Training (credit):
- Full backpropagation over each lane (no state-gradient truncation inside the lane); all level computations are exact
  parallel kernels over the lane (masked Toeplitz kernels with lags in parts/events, masked race reads).
- **Level-matched predictive targets.** Next-token loss is dominated by local structure and teaches slow variables
  weakly. Each level k ≥ 1 also predicts the content of its own future span from u^k_t: the tokens of the next N_k
  positions (N = 4, 16, 64 for word, clause, sentence), as an averaged cross-entropy over the tied vocabulary. These are
  exact causal targets that reward variables predictive over the level's horizon, the definition of a useful
  abstraction at that level.

Mechanisms: temporal decay/rotation memories at every level (time encodes order), sparse event clocks (level k is
active only at its boundaries: 89.7%, 11.2% and 6.9% of training tokens start word, clause and sentence/line units, mean spans
1.1, 8.9 and 14.4 tokens; `results/diagnostics/curie_part_whole_contracts_20261015_v1.json`), keyed
race reads at every level (separate keys/values, addressed retrieval of wholes), the strongest member's output keyed
bonus. Not included in this embodiment: hard routes with counterfactual credit, learned boundaries for unmarked units
(phrases, topics, entities); both are the next additions once observed-boundary levels are shown to carry value.

Relation to plan C (AWS, r1_async_hierarchy_lm.py): plan C learns emission by a straight-through race with a rate
target, integrates a segment by its mean, holds the last completed upper event, and trains levels to predict lower
hazards/concepts. This construction fixes boundaries to the data's observed wholes, integrates parts in order, exposes
provisional wholes to the prediction and trains each level on its own future horizon.

## 4. Predictions

- **P6a.** At equal fresh tokens, the hierarchy beats the plan-B L2 member it extends (one pass over 4M tokens): lower
  last-quarter prequential loss and lower flagship-protocol NLL.
- **P6b.** The gain concentrates on word-initial targets, and history beyond 64 tokens becomes worth more than the
  keyed member's 0.023 nats on word-initial targets.
- **P6c.** Every level carries unique value on word-initial targets (mean-replacing u^k in the readout raises their loss),
  in contrast to the race stack's deep stages (≤0.001).
- **P6d.** The level-matched targets raise P6b's long-history value relative to the same hierarchy trained on next-token
  loss alone.

## 5. Why higher-level concepts are hard to learn, and what the concept hierarchy changes

A document-level variable z (topic, entities in play, register) changes each token's prediction only slightly but
touches every token: its per-token information I(x_t; z | local context) is small while the document total is large. Its
learning signal is a sum of many small per-token gradients. It is learnable only if (i) evidence and credit span the
document, (ii) some target concentrates the slow signal, and (iii) a representation persists and is shared across
documents. Measured (`results/diagnostics/curie_document_concept_curve_20261015_v1.json`; 32 validation documents of at
least 1024 tokens, word-initial targets at document positions 256-1024, NLL with history cut to 64 tokens minus full):
keyed 1M **−0.032**, keyed 4M **+0.040**, plan-B L2 one pass 4M **+0.069**. Document-context use grows with data and with
per-layer addressed reads, and is still small.

**Receptive fields grow with cadence.** A level ticks once per completed whole; a memory mode with horizon N level
events reaches N × span tokens. Observed spans: word 1.1, clause 8.9, sentence 14.4, line 44, document 688 tokens
(val; 411 in the training sample). An addressed store of C wholes per level covers C × span tokens. Work: the token level
pays per token; level k pays per level-k event (11%, 7%, 2.3%, 0.15% of tokens above the word level); its memory is a
linear scan over its events and its reads score wholes, not tokens. Context therefore scales at near-linear cost, with
the token-level read the only quadratic term (to be made local for long contexts).

**Credit can scale the same way (multi-rate truncated credit).** By the credit-horizon theory a state is taught to
retain only what its gradient window reaches. Counting each level's truncation in its own events, a 16-event window
reaches ~230 tokens at the sentence level and ~700 at the line level. Stream long documents in chunks with state
carried, cut lower levels' gradients at each chunk and keep upper levels' graphs across many chunks (they hold few
activations): training context then scales with the hierarchy, not only inference context.

**Concept hierarchy** (`experiments/r1_concept_hierarchy_lm.py`): five observed levels (word, clause, sentence, line,
document; the document level's provisional whole is a running ordered summary of the current document), level-matched
future targets over 4/16/32/64/256 tokens, sparse concept memories on the sentence, line and document levels (race over
1024 learned slots, top 8, null route; documents about the same thing pool evidence into shared slots), trained on
1024-token lanes.

## 6. Race attention over the part-whole tree

Each completed whole knows its span, and its parts are the lower-level wholes inside it: the data's part-whole tree is
explicit. A read descends it coarse to fine. Nodes at each level: the completed wholes so far and the in-progress whole
(keyed by its provisional state), which parents the completed parts inside it; leaves are tokens. Weight of a node =
weight of its parent × softmax among its siblings (top level: softmax with a null route), so the weights form a
distribution over every earlier token reached through its wholes (contract: token mass + null mass = 1). Level reads and
the token read enter the prediction through zero-initialised maps.

What it gives: (a) long reach at a cost of about top-k × branching × levels per query once pruned (dense and exact on
1024-token lanes now); (b) verbatim detail retrieved through a gist: the whole is the key, its parts the values; (c) a
learning signal for concepts: read-time credit reaches each whole's key according to whether descending into it helped,
so a sentence or paragraph is trained to be an index of its contents, the signal next-token loss gives slow variables
only weakly. Coarse-to-fine retrieval has precedents (hierarchical attention, summary trees for retrieval, kNN memories);
the native embodiment here is the observed tree, races at every level with credit, in-progress wholes and per-level
clocks.

**Predictions (one pass over the same 4M fresh tokens, 1024-token lanes; `curie_concept_decision_20261015_v1`).**
- **P7a.** The concept hierarchy with the tree read beats its plan-B L2 base on the flagship protocol and the
  last-quarter prequential loss.
- **P7b.** Its document-context value (word-initial, positions 256-1024) exceeds the base's and the no-tree hierarchy's.
- **P7c.** The tree read is responsible: with tree < without tree on the flagship protocol.

## 7. Long context: test and training design

**Data.** FineWeb has the long documents a receptive-field test needs (validation shard: median document 402 tokens,
mean 695; 23,377 documents of ≥1,024 tokens hold 53% of tokens, 1,968 of ≥4,096 hold 17%, 594 of ≥8,192 hold 9%, 143 of
≥16,384 hold 4%; the training shard is the same within a few percent).

**Length test (no training).** The current models accept any lane length with bounded inference memory.
`analysis/document_concept_curve.py --length 8192 --cut-stride 16` scores documents of ≥8,192 tokens from their start and
reports, per position bucket up to 8,192, the word-initial NLL and the document-context value (history cut to 64 tokens
minus full, at sampled positions). Prediction **P8a**: models whose upper levels tick on wholes generalise beyond their
training lane (document-context value stays positive and the NLL keeps falling past 1,024 tokens), where token-clock
models trained on 256-token lanes do not.

**Training design (streaming with multi-rate credit).** Stream each document in chunks of 256 tokens with every state
carried: plan-B temporal memories (complex state), token-level keyed reads over a local window, each level's
in-progress span integrator (a linear recurrence with reset), level memory scan state, the stores of completed wholes
(keys, values, spans) and the tree nodes built from them. Exact gradients over a window of G chunks come from chunk-level
recomputation (store each chunk's input and incoming state, recompute it in backward): memory is one chunk's activations
plus the carried states, compute about twice the forward. Multi-rate credit then detaches each level's carry at its own
period: token-level carries every chunk, word and clause every 2, sentence 4, line 8, document 16 chunks, i.e. credit
horizons of 256 to 4,096 tokens counted in each level's own events. Contracts before use: carried streaming equals the
lane-parallel model on the same tokens (state carry exact), chunk recomputation equals full backpropagation over G
chunks, and each detach period cuts exactly the paths it names. Prediction **P8b**: at equal fresh tokens, multi-rate
training on 4,096-token windows raises document-context value at positions 1,024-4,096 above 1,024-lane training.

## 8. Result: P6 (`results/diagnostics/curie_part_whole_decision_20261015_v2.json`)

One pass over the same 4M fresh tokens, same seed and schedule, flagship protocol unless noted:

| | Plan-B L2 base | Part-whole hierarchy |
|---|---|---|
| Flagship NLL | 5.8648 | **5.8594** |
| Last-quarter prequential | 5.8362 | **5.8143** |
| 65,528-target slice | **5.7088** | 5.7112 |
| Word-initial NLL | 6.2988 | **6.2902** |
| History beyond 64 tokens, word-initial | 0.007 | **0.023** |

P6a, P6b and P6c pass under the preregistered rules; the quality margins are small (single seed), the 65,528 slice
slightly favours the base, and the mechanism evidence is the strong part: every level carries unique value on
word-initial targets when its readout input is mean-replaced (word 0.049, clause 0.013, sentence 0.010 nats; the race
stack's deep stages ≤0.001), and the next-word choice uses 3.3x more history beyond 64 tokens. The levels learn used,
slow structure; the next steps put them on the deep stack (plan-B L8) and give them long lanes, document levels, concept
memories and the tree read (P7). P6d (future targets) follows in chain v13.
