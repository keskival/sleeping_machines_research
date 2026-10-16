# 172 — Residual memory: layers refine one additive, persistent, addressed memory

16 October 2026, Curie. Founder: "By residual memory I mean additive memory. So the higher layer can only write memories
which refine these memories … Think about this properly and stop just going through where the fence is the lowest." "So the
first layer memory is A, the second reads A+B and writes B, and so on?" Supersedes the memory part of 171 §5. 171's race
readout and its exact race credit (T3) stand. Builds on 170 (typed units; measured cause stream) and the
temporal-memory/receiver algebra.

## 1. The idea, and why it is the right fence

A residual network composes because every layer reads and writes **the same space**: x_k = x_{k−1} + f_k(x_{k−1}).
Types match by construction. No layer can destroy what lower layers wrote; it can only add a refinement. The loss sees the
sum, so every layer's write receives the loss gradient directly.

Our earlier stacks broke one of these:
- They passed a *different* object up: the winner's vector, a pooled summary, or the attribution symbol stream (170 §2).
- Or they appended *new kinds* of entries, so the higher level's output was not the lower level's input type (170 §§3, 7).

Residual memory restores all three properties in *memory space* instead of activation space:

    level 1 writes A;   level 2 reads A + B and writes B;   level 3 reads A + B + C and writes C;   …
    the readout reads M = A + B + C + …

The memory is persistent and evolves linearly in time. So a level can write *whenever it fires*: cadence is decoupled
without any alignment machinery. A layer is no longer "a function from a length-T sequence to a length-T sequence". It is
**a sparse, asynchronous refiner of a shared persistent memory**.

## 2. Objects

- **Memory space 𝓜:** a set of addressed slots σ. Each slot has a key part and a value part, z_σ ∈ C^m. A slot is either:
  - an *event slot*, born when an event is observed (one per token event; holds what that event means);
  - a *standing slot*, persistent units such as concepts, entities or frames, dormant until written.

  All levels share 𝓜. Slots have no owner; **contributions** have owners.
- **Linear time evolution per slot:** z_σ(t + Δ) = D_σ(Δ) z_σ(t), with D_σ(Δ) = diag(e^{(−r + iω)Δ}). Event slots may
  use r ≈ 0 (stable meaning); standing slots decay and rotate (the family's temporal memory). D is linear, so it commutes
  with sums.
- **Level k's contribution:** M^k_σ(t) = Σ_{s<t} D_σ(t − s) W^k_σ(s), where W^k(s) are level k's writes at time s.
  These are sparse: few slots per firing, and only when level k fires.
- **Total memory:** M(t) = Σ_k M^k(t). This is the "A + B + C" of the founder's statement.
- **Level k at step t** (levels in order 1..K within a step):
  1. It fires or not; firing is itself a race clock (§5).
  2. If it fires, it **reads** the current total memory: everything from earlier steps plus this step's writes by levels
     < k. A read is a race over slots: candidate slots by key match, top-k, softmax-weighted values.
  3. It **writes** sparse additive refinements W^k_σ(t) into a few slots: the newest event slot (deepen what this event
     means in context) and the standing slots its read selected (refine what a concept or entity means in this document).
  4. Its private state is a small summary of its reads (no other level reads it).
- **Readout:** the CERL race (169) over the total memory. Episodic causes use refined event-slot keys and values;
  standing-slot causes use refined standing values; the background route uses the readout's own read. Exact race credit
  (171 T3).

## 3. Properties

**P1. Consistency of the sum over time.** M(t) = Σ_k M^k(t) for all t, and evolving the sum equals summing the evolutions,
because D is linear. ∎ This is what lets levels write at different cadences into the same memory without breaking the
residual algebra.

**P2. Type closure is by construction.** Every level's input is (𝓜 state, its events) and every output is a write into 𝓜.
Induction over levels is trivial at the type level. The substantive questions move to credit and work (P4, P5).

**P3. Nesting.** Initialise every level's write map at zero (as residual branches are zero- or 1/√(2K)-initialised). With
W^K ≡ 0 the model equals the (K − 1)-level model exactly. So the best attainable likelihood never decreases with depth. ∎

**P4. Exact credit and its residual identity.** Let G_σ(t) = ∂L/∂M_σ(t), the loss gradient on the total memory at the
times it is read (readout and higher levels). For any level k and write W^k_σ(s):

    ∂L/∂W^k_σ(s) = Σ_{t > s} D_σ(t − s)^* G_σ(t)   +   (terms through reads by levels j > k at steps ≥ s).

**The first term is the same for every level.** A refinement by level K is credited exactly like level 1's write to the
same slot at the same time. This is the residual-network gradient identity, transplanted to memory: there is no vanishing
with depth.

In time, credit travels from a read at t back to a write at s through the slot's own linear evolution: one hop, decaying
only as fast as the memory forgets (event slots with r ≈ 0 keep long credit). There is no serial chain through
intermediate steps or slots.

*Proof:* M is linear in every W; differentiate. ∎ Within a firing, the write map's parameters receive
∂L/∂W · ∂W/∂θ through ordinary backprop (a few nonlinear hops). The race readout supplies G exactly (171 T3: posterior
minus prior at the clocks, responsibilities at the proposals).

**P5. Work.** Let level k fire on a fraction ρ_k of steps, read top-k_r slots and write m_k slots. Work per step is
Σ_k ρ_k (k_r · key score + read + m_k · write) + readout. Memory size is capacity, not activity: untouched slots cost
storage only (plus lazy evolution, closed-form at read time).

**P6. Refinement cannot erase.** A level adds; it cannot overwrite lower contributions. To *suppress* a lower meaning it
must write the negative. That is learnable, costs a write and is visible to credit. The lowest-level evidence stays
available to every reader, which is the identity path that 170 §5 required.

## 4. What "deep" means here, and why it should compose

- **Depth is accumulated refinement per slot.** An event slot's meaning after K levels is the event's embedding plus K
  contextual refinements, each computed from the whole memory at the time that level fired.
- **Not limited to the moment of the event.** A higher level that fires later, for example when an entity is mentioned
  again, can write into the *original* mention's slot. Old memories improve in the light of new evidence. A Transformer's
  per-layer key/value cache is frozen once computed; this is a capability it lacks.
- **Standing slots become document-level concepts and entities** by accumulating refinements from every level that
  selects them. Their meaning in this document is the sum of what each level contributed. This is the durable structure
  that 170 §8 found missing from per-token causes.
- **Higher levels can afford to be expensive and rare**, firing on surprise or at boundaries, because their writes persist
  and are read by everyone at every later step.

## 5. Discrete choices and their credit (no low fences)

- **Which slots a level reads:** top-k_r by key. The softmax weights inside the selection are exact. Selection boundaries
  get inclusion credit: one random extra slot per read, credited by how much including it would have changed the read
  output's contribution to the loss. The race readout uses the 171 T5 (ii) estimator. Inside a level's read, the analogue
  is the gradient of L with respect to that slot's read weight, evaluated with the slot included.
- **Which slots a level writes:** the newest event slot (always, when firing) plus the slots its read selected, with write
  magnitudes from the read weights. This is a hard address with a soft magnitude, so credit to the magnitude is exact (P4).
- **When a level fires:** a two-clock race, fire vs wait, with fire probability φ_k(t). Training writes φ_k · W_k
  (expected refinement; differentiable). At inference the level fires when its clock wins. Work counts actual firings.
  Credit for φ is exact through the expected write. The train/inference difference is the variance of a Bernoulli
  write; the contract measures that gap and a hard-firing fine-tune closes it.
- **Standing-slot creation:** a pool of dormant standing slots (capacity beyond activity). A slot becomes active when it is
  first selected and written.

## 6. Relation to known constructions (attribution)

- Residual networks (He et al. 2016): the same algebra, in activation space.
- Neural Turing machines and differentiable neural computers (Graves et al. 2014/2016): addressed additive writes
  (erase + add).
- Fast weights and linear-attention memories (Schmidhuber 1992; Ba et al. 2016; Schlag et al. 2021): additive outer-product
  memories.
- State-space and linear-recurrent memories: linear time evolution.

The composition claimed here is all of the following together:
- every layer refines one shared addressed memory, asynchronously and sparsely;
- refinements may target *old* event slots;
- the readout is an exact race of causes with exact credit;
- each layer's cadence is its own race clock.

## 7. Construction for language (RMR-1)

- **𝓜:** event slots (one per token in the lane; value dimension m = d; r ≈ 0 learned per mode group) plus U standing
  slots (dormant pool; decay and rotation learned per slot).
- **Level 1 (fires every token):** writes the token embedding plus its local refinement into the new event slot. It writes
  into the standing slot attributed by the readout posterior, a hard address with responsibility magnitude.
- **Levels 2..K** (fire by their own clocks; at first every token, with φ learned, then sparse): read the total memory via a
  race over event and standing slots. Write refinements into the newest event slot and into the top-m read slots.
  Write maps are zero-initialised.
- **Readout:** the CERL race over the total memory:
  - repeat and continue causes over event slots, keys computed from refined values;
  - concept causes over standing slots, proposal laws shifted by refined values;
  - the background route.
- **Learning:** one objective, the exact token log-likelihood (171 T3 at the race; P4 through memory). Optional
  self-supervised terms only as diagnostics.
- **Labelled implementation choice:** the per-step level order within a token step is synchronous. The asynchronous
  version keeps the same algebra (P1).

## 8. Contracts and the decisive experiments

**Contracts:**
- P1: summing per-level contributions equals the total; evolving then summing equals summing then evolving (≤ 1e-6).
- P3: zero write maps give exactly the lower stack.
- P4: autograd gradient of a level-K write equals level-1's direct term plus measured higher-level terms. On a level with
  no higher readers the two are identical.
- Causality: perturbing token p leaves every earlier probability unchanged, including through writes into old slots.
- Exact normalisation.

**Experiments (registered):**
1. RMR-1 with K = 2 and K = 4 levels vs CERL at 4M, same data, schedule and seed. Pass: loss falls with K, with the gain
   on novel-in-window tokens.
2. Per-level credit health: write-gradient norms equal across levels at the same slot (P4 identity); firing rates ρ_k;
   standing-slot purity and lifetime.
3. Compositional scaling: the gain from K = 2 → 4 at 4M vs 16M.
