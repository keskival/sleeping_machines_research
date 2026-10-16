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

## 7. Development log

**16 Oct, first integrated checks (Curie, FineWeb, 4×256-token lanes, identical lanes, order and LR schedule as the dense
diagnostic).**
- *Sample efficiency.* Prequential loss after 300 updates is 6.58 against 7.46 (dense L2) and 7.42 (dense L8) at the same
  update. The race causes give explicit copying, induction and topic proposals from the first updates.
- *Native share* (posterior responsibility on unseen training lanes): concepts 67%, background 17%, continue 14%, repeat
  1.6%. The race causes reduce loss by 0.40 nats over the background route alone.
- *Long history is used.* The same validation tokens score 7.19 in 2048-token lanes against 7.30 in 256-token lanes.
- *Repair 1, train/eval consistency.* Exploration units first entered the training mixture but not the evaluation one. They
  now receive counterfactual credit only: "would I have raised p(y) against the race held fixed?". The predictive race is
  identical in training and evaluation.
- *Repair 2, bounded episodic hazard.* The power-law age kernel is initialised with slope 1.5, so it is integrable and the
  total pointer pressure stays bounded as history grows.
- *Execution.*
  - One word-law pass covers all causes: 1.51 s/update instead of 2.46.
  - Row-local uint8 match lengths: an 8,192-token lane peaks at 549 MB.
  - The product-key two-stage wake is exact; 65,536 units cost only 14% more time per update than 4,096.
- *Cause messages* (§3 state update) are implemented and pass contracts. At 300 updates there is no gain yet (6.60 vs 6.58)
  at 1.55× cost; the paired 4M run decides.
- Pending: the 4M one-pass fit against the dense diagnostic (5.8414 flagship); the capacity ladder (4k–262k units at
  constant active work); PTB round 1.


## 8. Compositional depth: a hierarchy of races over compressed cause streams (CERL-H)

16 October 2026, founder direction: "We need depth and compositional hierarchy there. This has been the main hurdle";
"Work towards compositional scaling. We need to win through that"; "don't make our models dense to do that".

**Why earlier depth failed, in one line each.**
- The serial race core passed only the winner's value up a chain, so credit vanished with depth.
- The dense residual stack composed, but it was not our model.
- Observed part-whole levels (word/clause/sentence boundaries) added 0.006 nats: they predicted tokens again from coarser
  summaries.
- The synthetic G-HIER binder recognises parts perfectly but binds wholes at chance.

**Construction.**
1. *Level 1* is the token race (§3).
2. *Attribution.* Its posterior names the cause of every observed token: a concept unit, the background route, a repeat or
   a continuation. This is an observation of structure, taken without gradient.
3. *Compression.* The attributed-cause stream is compressed in time: a level-2 event occurs only when the cause changes. A
   copied phrase or a persistent concept is one event.
4. *Level 2* is the same race over that stream:
   - its episodic pointers repeat and continue *patterns of causes*, which is induction over structure rather than over
     surface tokens (a paraphrase with the same concept sequence matches);
   - its dormant concepts propose which lower causes come next.
5. *Higher levels recurse* on their own attributions. Each level is sparser in time and sees further back at the same cost.
   Depth is the number of explanation levels; time compression makes deep context cheap.

**Top-down composition (sparse).** At token row t, level 2 holds the event that contains the cause of x_t and predicts the
next lower cause, as a stay/switch hazard over candidates:
- stay with the current cause: log sigmoid(g_t), with g_t = a + b log1p(age of the event), a duration clock;
- switch to cause c: log(1 − sigmoid(g_t)) + log p2(c).

The prior is evaluated only for candidate causes: level-1 key-woken units, a few units woken top-down by level 2's
strongest continuation pointers, and the three route types. λ × prior enters the level-1 race log-rates, so the token law
stays exactly normalised. Top-down-woken units pass a learned gate.

**Learning is local per level.** Level 1 learns from tokens (and through the one-hop top-down path). Level 2 learns to
predict the next attributed cause. No gradient travels a long chain through depth.

**What is not added.** No attention, no full-alphabet distribution, no dense layer between levels. Higher levels compute
only at their events.

**Contracts** (`experiments/r1_cerl_h.py --mode contract`):
- exact normalisation (1 + 3e-8);
- causality through attribution, compression and top-down (perturbing token p changes no earlier probability beyond
  rounding, 5e-7);
- row chunking;
- at λ = 0 the hierarchy equals level 1 (5e-7).

**What compositional scaling means here, and how it is tested.**
- (i) Loss improves with each added level at nearly constant active work, because higher levels run at compressed event
  rates.
- (ii) The gain from depth grows with data. The dense diagnostic already shows this (4→8 layers: 0.052 at 16M against a
  2→8 gain of 0.023 at 4M); the native levels must show it too.
- (iii) Structure-level induction appears: level-2 continuation responsibility rises, and the loss falls most on targets
  inside recurring cause patterns.

First test: CERL-H with 2 levels against level-1 CERL on identical data, schedule and seed at 4M; then 3 levels, and 16M.


**16 Oct, first full fit: the native race beats the dense hybrid (`curie_r1_cerl_4M_s0_20261016_v1`).** Same 4M fresh tokens, lanes,
order, one-cycle schedule and seed as the dense diagnostic.

| | CERL (2 state layers, 1 race level) | Dense hybrid L8 | Dense hybrid L2 |
| --- | --- | --- | --- |
| Flagship (8,192 targets, 1024-token windows) | **5.4665** | 5.8414 | 5.8648 |
| 65,528-target slice | **5.3094** | 5.6955 | 5.7088 |
| Selection (8 × 2048) | **5.6705** | 5.9958 | 6.0151 |
| Last-quarter prequential | **5.4582** | 5.8095 | 5.8362 |
| Training FLOPs per target | **7.7M** | 25.8M | 21.1M |
| CPU wall | **5,502 s** | 17,820 s | 6,558 s |

- The gain is 0.375 nats on the flagship, with 3.3 times less training compute per target. The earlier keyed member trained
  3 epochs on the same 4M tokens scored 5.601, so one pass of the race beats three passes of the dense keyed member.
- Native share on the flagship: race causes deliver 0.445 nats beyond the background route. Posterior responsibility:
  concepts 55%, background 26%, repeat 10%, continue 10%. On the 65,528 slice copying is larger (repeat 16%, continue 17%,
  native gain 0.554).
- Active scoring work per target at 255 tokens of history is about 0.64M multiply-adds: state 0.21M, episodic 0.07M,
  unit keys 0.07M, word laws 0.29M. A dense L8 forward pass is about 4.3M.
- Selection curve: 6.173 / 5.869 / 5.711 / 5.671 at the four quarters. Dense L8 was 6.578 / 6.243 / 6.047 / 5.996.
