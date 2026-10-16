# 171 — A trainable, composable race hierarchy with exact credit: additive hazards over owned memories

16 October 2026, Curie. Founder: "Can you analyze all this formally and properly, guided by intuition and make sure we have a
trainable, composable structure into which credit is assigned properly as well?" Builds on 168 (race Fisher geometry), 169
(causal event race; CERL-H), 170 (typed units; residual races and memories; measured cause stream). Derivations with proof
sketches; contracts in §8 turn each claim into a numerical check.

## 0. Intuition in four sentences

1. A race of causes is a *mixture of explanations*. Each cause claims probability (its hazard) and offers an explanation
   (its proposal law).
2. Credit for a mixture is exact and local: after the token is seen, every cause is credited by how much it explained,
   minus how much it claimed (posterior minus prior).
3. If every level's causes race *in the same token race* (additive hazards) and every level's knowledge lives in *owned,
   addressable memories* that others read directly, then every parameter is a few hops from the loss, at any depth and any
   distance in time.
4. Higher levels stay sparse because their memories change only when durable things (topics, entities, frames) change;
   they stay distinct from lower levels because they compete in the same race and win only where they explain better.

## 1. Objects

- **Token stream** x_1..x_T over vocabulary V. Prediction at step t is the law of x_{t+1} given x_{≤t}.
- **Levels** k = 1..K. Level k owns:
  - a **memory** M^k_t: a finite set of typed entries μ = (key κ_μ, value v_μ, type, birth time, last-write time,
    strength a_μ(t)), written only by level k;
  - a **private state** h^k_t: a function of level k's inputs, with no other level reading it;
  - a **candidate cause set** C^k_t, a finite subset of: entries of M^k_t, dormant units of level k, and level-1's
    background route. Each cause c has a log-rate s_c(t) and a proposal law q_c over V.
- **Inputs to level k at step t:** level k−1's events up to t, and read-only queries of memories M^j_{<t} for any j
  (lower, own or higher). A query is a race over entries: an attention-like read restricted to addressed entries.
- **Writes.** After x_{t+1} is observed, level k may write: create or refresh entries with strengths that depend on the
  posterior attribution r(t) (§2) and on x_{t+1}. A write at step t is visible from step t+1.
- **Events of level k:** its writes. Its cadence is its write rate ρ_k, a measured property, not assumed.

## 2. The model: one race over all levels' causes (additive hazards)

    p_θ(x_{t+1} = y | x_{≤t}) = Σ_{c ∈ C_t} π_c(t) q_c(y),   C_t = ∪_k C^k_t,   π_c(t) = exp s_c(t) / Σ_{c' ∈ C_t} exp s_c'(t).

This is the competing-risks law of exponential clocks with rates exp s_c: the first clock to fire names the cause, and the
cause emits from q_c. Superposing levels adds their hazards, the native residual (170 §7). The **posterior attribution**
after observing y is r_c(t) = π_c q_c(y) / p_θ(y | x_{≤t}).

## 3. Guarantees of the structure

**T1 (exact normalisation, causality).** Each q_c is normalised and π is a softmax, so p is normalised. C_t, s_c(t), q_c
and memories depend on x_{≤t} only, and writes made after observing x_{t+1} take effect at t+1, so p(· | x_{≤t}) is
causal. ∎

**T2 (nesting; a level never hurts at the optimum).** Give level k a gate log g_k added to all its causes' log-rates.
At g_k = 0 its causes have zero hazard and the model equals the stack without level k. So the best training likelihood
of K levels is at least that of K − 1 levels. ∎ (CERL-H uses top-down gain λ = 0 for the same purpose; contract equal to
5e-7.)

**T3 (exact, local credit at the race).** With L = Σ_t log p_θ(x_{t+1} | x_{≤t}):

    ∂L/∂s_c(t) = r_c(t) − π_c(t),        ∂L/∂ log q_c(x_{t+1}) = r_c(t).

*Proof:* log p = log Σ_c e^{s_c} q_c(y) − log Σ_c e^{s_c}. ∎

Consequences:
- Every candidate cause at every level is credited at every step it races. Winners that explained gain; losers that
  claimed mass without explaining are pushed down. Counterfactual credit to unchosen alternatives is *exact* for the
  candidate set; no sampling, no REINFORCE.
- The race Fisher information is diagonal in the clocks (note 168), so 1/E[wins] per cause is a natural-gradient scale.

**T4 (short credit paths at any depth and distance).** Count *nonlinear hops* from a parameter to L (linear decaying
memories count as one hop, as in 168/167). Assume:
- (a) each level's causes enter the token race directly (additive hazards);
- (b) memory reads address entries directly, with no relay through intermediate entries;
- (c) level k's state is a function of reads of memories and events at levels ≤ k, plus its own linear memory.

Then every parameter of level k reaches L in at most 2k + 1 nonlinear hops, independent of T and of how long ago an entry
was written.

*Proof sketch:* a level-k cause parameter reaches the race in one hop. A level-k write affects L either through a level-k
cause that reads it (write, read, race: three hops) or through a higher level that reads it. By (c), each extra level adds
at most two hops (read, then cause). Induct on k. ∎

Contrast:
- The serial race core passed the winner's value through every stage and every step: path length ∝ depth × time. Credit
  vanished (deep receiver gradients about 1e-5).
- Dense residual stacks have short paths, but only by being dense.

**T5 (credit to structure, the discrete choices).** The structure includes which candidates enter C_t (top-k by key or by
memory strength) and which entry index a write addresses (the attributed cause). Both are piecewise-constant in θ, so L is
piecewise smooth. T3 gives the exact gradient on each piece. Changes of structure receive credit through two exact local
signals:
- (i) **Responsibility-weighted writes.** A write's *address* is hard (the attributed cause's own entry), but its
  *strength* is r_c(t), which is differentiable. If strengthening an entry at t helps predict later tokens, the gradient
  raises r_c(t) through s_c(t) and q_c(x_{t+1}). The model learns attributions that are useful later, not only
  explanatory now. This is temporal credit for structure. (Detaching r, as CERL-H's attribution does, removes it.)
- (ii) **Counterfactual inclusion credit for dormant causes.** For a dormant cause e drawn uniformly from the N dormant
  causes, take

      g_e = ∂/∂s_e log p_{C_t ∪ {e}}(y) = π'_e (q_e(y) / p'(y) − 1),

  with π', p' the race with e added. g_e > 0 exactly when e's proposal explains y better than the current mixture. And
  E_e[g_e] = (1/N) Σ_{e ∉ C_t} g_e: an unbiased estimate of the average single-addition credit. It teaches keys to wake e
  when e would help. ∎ (CERL's paid exploration is this estimator with the race held fixed.)

What remains approximate, stated once: credit for the *boundaries* of discrete selections beyond these two signals; lane
truncation of gradient paths, even though memories persist across lanes; synchronous optimiser updates.

**T6 (bounded work under memory cadence).** Let level k write on a fraction ρ_k of steps and keep at most m_k candidates
in the token race. Work per token is

    W = Σ_k [ρ_k · (state update + write) + m_k · (score + proposal) + reads].

This is bounded by constants independent of K's history length. Candidate caps make the race O(Σ m_k). Sparse writes make
higher levels cheap exactly when their memories change slowly. ∎

## 4. Why higher levels become abstractions and not copies of level 1

- **Competition in one race.** A level-2 cause wins mass only on tokens it explains better than level-1 causes (T3
  credit). Like mixture-of-experts specialisation, but with the specialisation axis set by *what information each level
  holds*.
- **Different information.** Level 1 holds the token-level present: syntactic slot, recent tokens, copyable spans. 170
  §8 measured its causes as syntactic slots re-chosen every token. Level 2 holds slow memories: which topics and entities
  are active in this document. It therefore wins on tokens predictable from slow context but not from the local slot:
  topical content words, entity attributes, repeated constructions. That is the novel-in-window content where level 1 is
  only level with the dense model (7.87 vs 8.02).
- **Different proposal families.** A topic cause's proposal generalises to words not yet seen (a learned law shifted by
  the topic), unlike a copy pointer. That is the capability needed for novel tokens.
- **Durability is enforced by the memory, not hoped for.** A topic entry's strength decays slowly and is refreshed by
  responsibility-weighted writes, so its activation, and the set of woken topics, changes rarely. Level-2 events are sparse
  by construction; ρ_2 is measured, not assumed.

## 5. The construction (CERL-M v1)

**Level 1:** the CERL race:
- episodic repeat/continue causes over the token-event memory, which is append-only and owned by level 1;
- syntactic-slot concept units;
- the two-level background route;
- private temporal-memory state, labelled; AWS's receiver core is the native replacement.

**Level 2 (topics):** a bank of U_2 topic units, each with key κ_u, proposal vector o_u and decay rate β_u.
- *Memory:* strength a_u(t). After observing x_{t+1}, the token event writes to the top-m topics by key match of its
  embedding, with strength

      w_u(t) = r_{tok}(t) · softmax_m(κ_u · e(x_{t+1})),

  where r_{tok} is the posterior share of non-copy causes, so copied tokens do not drive topics. Strengths evolve as
  a_u(t+1) = e^{−β_u} a_u(t) + w_u(t): additive, sparse writes.
- *Causes:* the top-k_2 topics by strength enter the token race with log-rate

      s_u(t) = α · log(a_u(t) + ε) + q_2(h^1_t) · κ_u + b_u

  and proposal q_u(y) = two-level word law shifted by o_u. This generalises to unseen words.
- *Events:* a change in the top-k_2 set. ρ_2 is measured.
- *Credit:* exact by T3 at the race. Through the strengths (T5 i), credit reaches topic keys, the token embeddings that
  wrote them, and the responsibilities that weighted the writes. The decay rates are learnable.

**Level 3 (later):** discourse frames over topic events and level-1 slot sequences, same pattern.

**Nothing dense crosses levels:** level 2 reads token events (sparse writes) and level-1 posteriors (scalars). It
contributes candidate causes, never a vector into level 1's state.

## 6. What this changes relative to CERL-H

CERL-H's level 2 biases level-1 rates from a model of the attribution symbol stream. 170 §8 measured that stream as
syntactic slots at token cadence (ρ = 0.79), so CERL-H models tag sequences.

CERL-M's level 2:
- owns a slow memory built from token content weighted by posterior attribution;
- contributes its own causes (additive hazards), with direct exact credit (T3);
- has temporal credit for structure through responsibility-weighted writes (T5 i).

## 7. Experiments that decide it (registered before results)

1. **CERL-M vs CERL** at 4M, same data, schedule and seed. Pass: lower flagship and 65,528 loss, *with the gain
   concentrated on novel-in-window tokens* (the error-analysis split). ρ_2 ≤ 0.2, i.e. topic set changes on at most 20% of
   tokens.
2. **Credit health:**
   - level-2 responsibility share;
   - gradient norms of topic keys and decay rates, non-zero and stable;
   - exploration credit sign agreement (g_e > 0 when q_e(y) > p') on held-out tokens.
3. **Compositional scaling:** gain of level 2 at 4M vs 16M (does it grow with data, as dense depth did: 0.023 → 0.052?).
   Then level 3.

## 8. Contracts (each claim as a numerical check)

- T1: Σ_y p(y) = 1 at random positions; perturbing x_p leaves every earlier probability unchanged (including through
  memory writes).
- T2: gate 0 ⇒ equal to the model without the level (≤ 1e-5).
- T3: autograd ∂L/∂s_c equals r_c − π_c, and ∂L/∂log q_c equals r_c, for every candidate at every level (≤ 1e-5).
- T5 (ii): over many draws of e, the mean exploration gradient equals the enumerated average single-addition credit.
- T6: measured candidates per token and write rate per level.
