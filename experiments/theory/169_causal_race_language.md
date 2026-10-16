# 169 — Language as a causal event race: the native language member, designed from the family's own principles

16 October 2026, Curie, founder direction: the language program has regressed toward the mean, "yet another Mamba variant
which doesn't even leverage the causal inductive bias which enabled us to win the causal sequences domains"; "we really don't
want to make our model a Transformer or an SSM piece by piece, death by a thousand cuts". This note records the failure, the
principle and the design that replaces the drift. Builds on the B4 pointer race (`experiments/tpp/race_tpp_pointer.py`), the
competing-risk geometry of note 168, the temporal credit of note 167 and the episodic language result of 15 October.

## 1. The failure

The current language flagship (`r1_deep_keyed_lm.py`, "plan-B") is, mathematically, a pre-norm residual stack of:
- a diagonal complex linear recurrence with input-independent decay at unit steps (an S4D/LRU-type SSM layer);
- softmax attention with a linear distance penalty (ALiBi-type);
- an MLP;
- a dense 50,257-way softmax with a log-additive slot bonus.

Each step that produced it was locally justified. The race core failed to compose with depth, so it was replaced by residual
blocks. Hard routes made gradients small, so full backprop was used. Dense FLOPs were the scoreboard. The sum is a generic
hybrid. What the family brings appears only as residue: the slot bonus, and decay/rotation memory at unit time steps where it
reduces to an SSM.

What won the causal-sequence domains is absent. B4 (Github and Wikipedia wins against the composite of published models) and
the B1 event battles explain every event by a **race among candidate causes**:
- specific past events, which repeat their mark or continue what followed them;
- a native generative route.

These are exactly normalised competing risks with age-aware clocks; the winner both emits the event and is its attributed
cause. Learning credits the winner and, counterfactually, the losers. The same bias, as an in-window copy/induction race,
already moved language: +0.345 nats over the 5.978 parent (+0.474 against the matched continuation), concentrated on rare
tokens (+0.6 to +1.2 nats by rarity rank). Then plan-B replaced the composition instead of building on it.

## 2. The principle

**A token is an event, and every event has a cause.** Text is a marked event stream with branching structure, as in a Hawkes
process:
- names, entities, numbers and phrases are re-emitted by the earlier events that introduced them;
- a phrase, once started, continues the span its parent began;
- topical vocabulary is emitted by a few active concepts;
- only a remainder needs a broad generative law.

The family's primitive, a race of clocks over persistent addressed memory, *is* this generative story:
- P(cause c wins) = λ_c / Σλ;
- the emitted token is the winner's proposal;
- the likelihood is the exactly normalised mixture over causes (competing risks, note 168: diagonal Fisher geometry per
  clock).

A Transformer must discover copying and induction through depth and data. Here they are the model's grammar from the first
update.

## 3. The native member: causal event race language model (CERL)

**Causes that race at each step t (issuing token t+1).**
1. *Episodic causes*: past events j from an addressed persistent memory. History is unbounded: entries are keyed and indexed,
   and retrieval admits a top-k candidate set. Each candidate j enters with two mechanisms:
   - *repeat*: proposes x_j;
   - *continue*: proposes x_{j+1}, i.e. induction, or continuation of j's span.

   The clock rate is λ_j = exp(q_t·k_j + mechanism bias − age penalty(t − j)) + span bonus. The span bonus applies when j
   caused t and j+1 is now proposed: a cause stays excited, so a copied phrase continues at almost no cost. This is
   self-excitation, the Hawkes kernel of the B-battles.
2. *Concept causes*: a large bank of dormant concept units (capacity beyond activity). Only units whose keys match the
   current state wake and race. Each proposes from its own sparse word set: it owns a few hundred words and normalises over
   them. Parameters grow with the bank; work per token grows with the few woken units.
3. *Background route*: a cluster race (word cluster, then a word within the cluster), exactly normalised. It gives every
   token support at about 2√V rows of work instead of V.

**Likelihood.**

    p(x_{t+1} | history) = Σ_c π_c(t) · p_c(x_{t+1})

with π the race-win probabilities. It is exactly normalised. Training uses this exact mixture over the admitted set.
Candidates outside the set receive the sampled counterfactual credit the family already uses for hard routes; with a race,
that credit has diagonal Fisher geometry (note 168). At inference the race is hard: the first arrival emits, and only the
winner computes its proposal.

**State: events update the memory of their cause, not every unit.**
- The emitted token is written as a new episodic entry, addressed by its own identity and carrying a pointer to its cause.
- The winning cause's persistent state receives a small message mixing the event's content with that state. This is the
  family's sparse addressed state update.
- Concept units decay with the elapsed time since their last win. Time computes here: a unit's readiness is its age, as in
  the B-battle clocks.
- No dense all-units recurrence runs at every token.

**Depth is depth of explanation.**
- Causes have causes: a continued span's parent is the event that began it; a concept's activation was caused by the events
  that woke it.
- Deeper levels race over higher-order causes (spans, phrases, concepts) with the same primitive.
- This is the part-whole composition already built for G-HIER, applied to text, not a stack of dense layers.

**Online learning is native.** Memory writes are learning. Causal weight adaptation runs on already-emitted text (contracts as
in `r1_native_stream_scorer.py`).

## 4. Why it should win, and where

- **Low-compute frontier.** Copy, induction and span continuation work from step 1. In-window repeats are 53% of FineWeb
  tokens on our development stream. Transformers spend data and depth acquiring induction heads. That gives a sample- and
  compute-efficiency advantage exactly where the CPU and early GPU ladders measure.
- **Active work.** Per token: retrieval of k candidates, a few woken concepts, one cluster race and one small state message.
  That is not O(d²·layers) plus a 50k softmax. Judged on loss against active FLOPs and chip work, it is a different
  denominator.
- **Scaling.** Capacity grows in the dormant bank and episodic memory without growing active work. That is the family's
  "capacity beyond activity", tested as a scaling exponent.
- **Hardware.** First-arrival races, local state, message-triggered updates and silence that costs nothing are the clockless
  target's native operations.

## 5. Guardrails against drift (founder direction, recorded in AGENTS.md)

- A language member's predictions must be delivered by the race of causes. Dense primitives (embeddings, normalisation,
  small local maps inside a cause) are permitted only inside a cause's computation. They never replace the race, the
  addressed memory or the sparse update.
- Every result reports its **native share**: the fraction of loss reduction over the background route delivered by episodic
  and concept causes. It also reports active work per token (candidates scored, units woken, messages sent, state bytes
  moved) next to loss.
- A failed mechanism is repaired at its cause: credit variance, retrieval misses or clock calibration. It is not replaced by a
  dense standard part. A proposed substitution requires this note's §1 analysis in writing before any code.
- The dense plan-B member is a labelled diagnostic control and is never the flagship again.

## 6. Development path

1. **Contracts:** exact normalisation of the mixture; causality (perturbing token p leaves every probability before p
   unchanged); hard-race inference that samples the same distribution; streaming equal to batch.
2. **CPU integrated fit** on PTB and FineWeb development slices against the dense diagnostic at matched active work and
   matched training FLOPs, with native share and work per token reported.
3. **Error analysis** by token role and history bucket (word-initial content words are where the dense member gains; the
   concept causes must carry those).
4. The GPU ladder runs the native member. Dense variants appear only as labelled controls.
