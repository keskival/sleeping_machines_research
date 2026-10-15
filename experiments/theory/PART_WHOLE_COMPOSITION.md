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
