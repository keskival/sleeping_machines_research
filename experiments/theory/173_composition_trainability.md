# 173. When does composing good learners give a trainable model?

Curie, 17 October 2026. Founder's question: "if one component learns and learns well, y = f(x), and we compose them, what is
needed to make the composition trainable?" Builds on 169 (the race), 171 (exact race credit) and 172 (residual memory). The
code review against these conditions is in §4, and the registered dead ends are in `experiments/index/DEAD_ENDS.md`.

## 1. What "learns well" gives, and what composition additionally needs

Training a component alone uses a fixed input distribution p(x), a target y and a loss ℓ(f_θ(x), y). Learning well means
∂ℓ/∂θ is informative and the optimiser finds a good θ. Only the **parameter gradient** is ever used.

In a composition F = f_K ∘ … ∘ f_1 with one loss L(F(x), y), level k is trained by

  ∂L/∂θ_k = (∂L/∂z_K) · J_K · J_{K-1} ⋯ J_{k+1} · ∂f_k/∂θ_k,  where J_j = ∂f_j/∂z_{j-1} is the **input Jacobian**.

So composition needs things that isolated learning never tests:

- **C1, credit path.** Every level above k must pass an informative input Jacobian J_j (or an unbiased estimate of it)
  down to k. A component can learn perfectly through ∂/∂θ and still block composition: hard selection, `detach`, or a
  discrete or attributed output with no gradient. With no credit path, level k learns only from its own local loss (C6).
- **C2, sufficient interface.** z_k must carry what the levels above need. For a strict Markov chain with no skip path, additional memory or new observation, the data-processing inequality gives
  I(z_K; y) ≤ I(z_k; y). Information lost from every available path cannot be recovered by the upper computation;
  with memory and skip paths the interface includes those paths, not only one vector. A winner-only value, an argmax
  attribution or a single summary vector are lossy interfaces.
- **C3, identity at initialisation (the base case).** If F(θ_0) equals the best shallow model, the composed loss starts at
  the shallow loss and gradient descent can only improve it to first order. Residual form: z_k = z_{k-1} + α_k g_k(z_{k-1}),
  with α_k = 0. This is a safe initialization statement, not a guarantee that a finite optimizer step improves loss.
- **C4, descent direction for each new level (the inductive step).** With α_{k+1} = 0,
  ∂L/∂α_{k+1} = ⟨∇_{z_k} L, g_{k+1}(z_k)⟩. Level k+1 lowers the loss if and only if its correction can correlate with
  −∇_{z_k} L. That requires (a) ∇_{z_k} L to exist and be informative (C1 at the readout) and (b) g_{k+1} to compute, from
  what it reads, a function aligned with that gradient. If g_{k+1} reads only what level k already used, it can still
  transform the same available information into a function computationally inaccessible to a shallower restricted component.
  New memory access is one way depth pays, not a necessary condition. For example, two bilinear levels compute
  (x1*x2)*(x3*x4), while one bilinear level over the same four inputs has degree at most two and cannot represent that
  quartic function. No extra observations are required. Hierarchical representation transformation is itself a depth axis.
- **C5, conditioning over depth.** The product J_K ⋯ J_{k+1} must preserve an informative, usable signal on task-relevant directions.
  Requiring every norm or singular value to stay near 1 is stronger than necessary. The residual identity gives
  J = I + O(α); normalized reads help control scale but do not guarantee conditioning.
- **C6, alignment when credit is local.** A level trained on its own loss ℓ_k produces a sufficient statistic for ℓ_k's
  target, not for the target of the levels above. Local learning composes only when ℓ_k's optimum is useful upstream
  (the greedy layerwise condition, which generally fails). Otherwise producers need top-down credit: a delayed or
  replayed signal of what consumers needed.
- **C7, hard choices.** For a discrete selection (route, woken unit, write address, firing), the gradient is conditional on
  the choice made. The choice itself learns only by (a) exact marginalisation (a mixture likelihood: race credit
  r_c − π_c), (b) unbiased counterfactual estimates with variance control per level, or (c) a biased relaxation
  (straight-through). If the training forward is soft and the evaluation forward is hard, the model can learn to use the
  soft value as a continuous gain that disappears at evaluation (a relaxation gap).
- **C8, matching time.** For temporal components, the credit window must cover the dependency span at each level. Persistent
  state is separate from truncation. Training and evaluation must expose the same length statistics wherever state
  accumulates with length.
- **C9, stability while the input moves.** As lower levels learn, an upper level's input distribution shifts. The shift
  costs at most Lip(f_{k+1}) · ‖Δz_k‖. Small correction gates, normalised reads and bounded (tanh, decaying) writes keep it
  bounded.

**Induction (the founder's "proof by induction").** C3 is the base case: the composed model equals a component that learns
well. C4 is the proposed inductive step: an added level can improve the loss when its correction has useful correlation
with the negative task gradient. New information alone does not guarantee that correlation, and is not required.
This is a conditional trainability argument, not a proof of supremacy across arbitrary depths or tasks. C1, C5 and C9 keep credit and stability valid as K grows. The induction fails exactly where one
of these fails. Every registered depth dead end fails one of them (§3).

## 2. The race as a component

For the race readout, L = −log Σ_c π_c p_c(y) with π = softmax(s). The posterior responsibilities are
r_c = π_c p_c(y) / Σ_c' π_c' p_c'(y).

- Parameter credit (171): ∂L/∂s_c = π_c − r_c (exact, bounded in [−1, 1]).
- **Input credit:** ∂L/∂h = Σ_c (π_c − r_c) ∂s_c/∂h + Σ_c r_c ∂(−log p_c(y))/∂h. This is exact and bounded by the score and law
  Jacobians. So the race passes C1 for its *continuous* inputs, conditional on the selected candidate set (the woken units).
- The candidate set (top-k wake) learns through exploration units (counterfactual credit to random dormant units, C7b), and
  through the balance term since 17 October.
- **Choose a useful race output interface.** The current cause readout returns a likelihood and posterior over causes;
  the CERL-H argmax attribution stream was unstable and omitted lower-producer credit. That failed construction does
  not imply races cannot compose. Delivered values, posterior feature vectors, persistent memory and generated marked
  events can all be interfaces if they preserve useful information and have an adequate learning rule. RMR's tested
  composition uses **shared memory** edited by each level. Within its dense reads, a mean-field value replaces an
  explicitly sampled delivery; that execution choice and its work must remain labelled.
- **Race and softmax.** First arrival among independent exponential clocks with rates e^{s_c} is won by c with probability
  softmax(s)_c. A level's soft read Σ_s a_ts v_s is the expected value delivered by a race over slots. Formally this is a
  softmax attention read with a log-distance clock. What makes a level ours rather than an attention layer:
  - writes into old slots (addressed memory editing);
  - learned decay of those writes;
  - hard sparse reads (`--read-topk`);
  - standing slots and firing clocks (RMR-2);
  - concept capacity that sits unused and is woken by the race;
  - counterfactual credit.

  Any variant that drops these is a labelled diagnostic, as the dense stack `--top-m 0` already is.

## 3. The registered depth dead ends, by failed condition

| Dead end (DEAD_ENDS.md) | Failed condition |
| --- | --- |
| Serial winner chain | C2 (winner-only interface), C7 (route credit vanishes with depth), C5 (gradients about 1e-5) |
| Race stages stacked without memory | C2, C4 (the tested interface did not yield useful new corrections) |
| CERL-H over attribution events | C1 (level-1 states are detached and attributed under `no_grad`: zero credit from level 2 to level 1), C2 (argmax attribution), C4 (79% change rate: nothing slower to read) |
| Detached lower messages in the local learner | C1, then C6 (no producer credit) |
| Ungated old-slot corrections | C3, C9 |
| Undecayed old-slot corrections on long lanes | C8, C9 |
| Weight decay on clocks and gates | C8: decay pulls `log_rate` of the slowest temporal-memory modes toward 0. Over a 16M run, rate 0.01 → about 0.027, so the memory horizon shrinks from about 100 to about 37 tokens |

## 4. Code review against C1–C9 (17 October)

**RMR (`r1_rmr.py`), the flagship depth construction.**

| Condition | Status |
| --- | --- |
| C1 | Credit reaches every level through the total memory (contract: the corrections change the law; the brute-force algebra matches). Old-slot corrections receive credit from every later reader. |
| C2 | The interface is the whole memory (one slot per token plus all corrections). |
| C3 | ReZero gates at 0: the model equals CERL exactly (contract 0.0). |
| C4 | Each level reads memory refined by the levels below, including old slots under refined keys. |
| C5 | Residual identity; LayerNorm on query and read. |
| C7 | Write addresses are the top-m read slots: credit reaches the write magnitudes (w = read weight), not the address choice. A slot the level did not read never learns "a write here would help". No counterfactual write exploration: **open**. |
| C8 | Lanes of 256 in training vs 1,024 / 8,192 in scoring; learned decay and `--lane 1024` are under test (DEAD_ENDS). Persistent memory across lanes: **open**. |
| C9 | tanh on old writes, learned decay; per-target corrections since 17 Oct. |
| Precision | **Fixed today:** under bf16 autocast (GPU driver), the old-slot algebra summed thousands of writes per popular slot in bf16. It now runs in fp32. |

**RMR-2 (`r1_rmr2.py`), firing clocks and standing slots.**

| Item | Status |
| --- | --- |
| Crash | **Fixed today:** it still stored dense [B, T, T] old writes while the shared algebra takes sparse (idx, w), so training through this driver crashed. AWS's 4M results came from its own stable copy. |
| C7 relaxation gap | Training wrote soft expected corrections φ·δ, while hard-fire evaluation fires only when φ > 0.5. **Fixed today:** `--fire-st 1` trains with straight-through hard firing (contract: training forward = evaluation forward, 5e-7). |

**CERL (`r1_cerl.py`), the readout and level 1.**

| Item | Status |
| --- | --- |
| Level 1 | `states()` is two complex-diagonal temporal-memory layers (a linear recurrence) plus MLP blocks. It is labelled as not completely native; the native replacement (AWS receiver core / event-local states) is open. |
| C1 | Exact race credit to the readout and its input. |
| C7, concept wake | Exploration units supply counterfactual credit. **Fixed:** zero-initialised concept vectors made every unit the same law, and there was no balance (`--unit-init`, `--balance`). The balance term needs the full unit softmax, so it is off for product-key banks: **open** for capacity runs past 1,024 units (needs a two-stage balance over the half-keys). |
| C8 | **Fixed:** weight decay on clocks (`--wd-groups 1`). |

**AWS local learner.** C1/C6 failure (detached lower messages); AWS's producer-credit repair is running.

## 5. Second round, 17 October: the open conditions worked

| Open item (§4) | Implemented | Contract |
| --- | --- | --- |
| C7 write addresses | Separate write-address race (`--write-race 1`, own keys over refined slots), and counterfactual write probes (`--write-explore E`). A probe has magnitude p − sg(p): zero in the forward pass, so training and evaluation predict the same, and its gradient is ⟨∂L/∂slot, probe correction⟩, the first-order value of writing there. Credit reaches only the address race. | Probes leave the forward pass unchanged (0.0) and add credit to the write keys |
| C8 persistent memory | `--carry W`: the level-1 temporal state continues across lanes; the previous W slots with every level's corrections stay readable, refinable and pointer-addressable; credit stays within the lane. Training rows walk contiguous streams; final scoring reports reset and carried lanes. | Carried lane 2 equals rows L..2L of one long lane (9.5e-7); row chunks with a prefix (4.8e-7) |
| C7 capacity | Load balance per half-bank for product keys (2√U scores) | NLL unchanged; term ≥ balance; credit reaches the half-keys |
| Writers saw unrefined slots | Address keys and per-target corrections now read refined slot values (old corrections included), as the read does. The old-correction input is stop-gradient: its forward value is used but its first-order producer-input VJP is omitted (saves autograd memory). | Contracts v7 |

Second-round review findings, all fixed:
- `ParallelTemporalMemory` built its kernel on the CPU, so the GPU driver would have failed on CUDA.
- RMR-2 had no prefix interface; it now refuses carry explicitly.
- Resume identity broke on new flags, which would have failed the PTB resume and AWS's running RMR resumes after a sync. Flags at behaviour-preserving defaults are now accepted.
- The FLOP count ignored the carried prefix.
- Write probes on read weights would bend reads toward write targets (separate keys and values); probes now require the write race.

Disclosed protocol notes:
- Carried scoring on the selection and 65,528 slices has a one-token gap at lane boundaries: the last token of a lane is only a target. Flagship windows are contiguous.
- `--carry` changes the training data order.

## 6. Next tests
1. The 16M arms on AWS: fixes; plus 1,024-token lanes; plus per-target corrections; the integrated member (everything above, `aws_r1_rmr_integrated_K4_16M_s0_20261017_v1`, after its RSS gate).
2. Carry for RMR-2 (firing clocks and standing slots), then the integrated member with RMR-2 at 16M.
3. Native level 1: event-local states in place of the linear recurrence, against CERL at 16M.
4. The capacity ladder (65k–262k product keys with balance) at 16M.

## 8. Third round, 17 October: native level 1

- **Native level 1** (`native_slot_memory.py`, `--level1 slots`) replaces the dense complex-diagonal recurrence. It has persistent addressed slots; each event writes its value into the top-m slots chosen by a write race; each slot decays at its own learned rate (1 to 1,000 tokens); reads are a race over slots with separate read keys and an abstain option, returning each slot's evidence-weighted mean; counterfactual write probes are included.
  - Execution is exact and chunked: the decay factorises within a chunk, and the slot state is carried between chunks and lanes.
  - Contracts: chunked equals the explicit sum (1.5e-8); causal; carry equals one long lane (7e-9); probes leave the forward pass unchanged and credit the write keys; read bounded at 2,000 tokens; the whole RMR member with slots is exact and normalised, and carry equals one long lane (9.5e-7).
  - It also subsumes RMR-2's standing slots (pooling into persistent decaying slots) at level 1.
  - Cost (fwd+bwd, 4 × 255 tokens, d 128): recurrence 0.76 s per layer; 1,024 slots 0.11 s; 4,096 slots 0.51 s. 4 of S slots are updated per token.
- **Fixes:** carried lanes keep a contiguous input stream (overlapping training lanes; scoring feeds the skipped token as an unscored input); measured inference work of the whole member is reported.
- **Dead end:** a raw-sum slot read (unbounded with length, C8).
- **Still open:**
  - GPU driver: no carry and no slot level 1 in its data path yet (CPU drivers have both).
  - RMR-2 firing clocks: no carry.
  - Level-1 slot execution is dense; event-driven execution would touch only m slots per write.

## 7. Earlier list (superseded by §5–6, §8)


1. C7 writes: counterfactual write exploration. A few random older slots per step receive a detached trial correction,
   credited by the change in the readout likelihood (the same estimator as exploration units).
2. C8: carried memory across lanes in training and scoring (persistent state, bounded credit window).
3. C7 capacity: a product-key balance term (balance each half-bank) for the 65k–262k capacity ladder at 16M.
4. Level 1 native: event-local states in place of the linear recurrence, against CERL at 16M.
5. The 16M arms already queued: review fixes; plus 1,024-token lanes; plus per-target corrections (DEAD_ENDS §1–2).

## Review qualification: writer input credit

`RMR.slot_values` currently evaluates the old-correction contribution under `torch.no_grad()`.
Its value informs per-target writing, but the derivative through that contribution is omitted.
This is a stopped **first-order producer-input path**, not generally a second-order derivative:
for a writer f_theta(v_phi), the missing term is J_f times d(v_phi)/d(phi). Other read paths still
train old corrections. The memory-saving approximation must be stated as such and tested before
calling all refined writer-input credit complete. Retaining this graph can be expensive; a bounded
version-bound VJP is an alternative to silently equating stopped credit with the full derivative.

The independent [17 October architecture review](ARCHITECTURE_REVIEW_20261017.md) gives three
passed numerical failure witnesses and the completed negative reciprocal-policy result.

## 9. Fourth round, 17 October (Curie, responding to the AWS review)

- **Writer input credit, fixed.** `RMR.slot_values` now returns refined target values with full first-order credit. Writes before the chunk accumulate per slot at the chunk start (the decay factorises), and writes inside the chunk use an explicit address match, so memory is O(T d), not the one-hot O(R m T m′). Contract: equals the one-hot `corr_values` reference in values (4.8e-7) and in all gradients (3.8e-6). The stop-gradient approximation is removed.
- **Firing clocks with carry** (`--fire 1`, `--act-cost`): RMR-2's asynchronous mechanism inside RMR-1.
  - Level k fires at t from its events in (current slot, size of the correction below, the lower level's firing).
  - It is hard in training and evaluation; straight-through credit is the first-order value of firing, because the correction is computed anyway in the reference execution.
  - Contracts: identity at init, normalisation, causality, training forward = evaluation forward (3e-6), carry exact.
  - Standing slots are subsumed by the native slot level 1.
- **Evidence-weighted old corrections** (`--old-norm 1`): a slot's old corrections read as their decayed weighted mean, which stays bounded as β → 0 (AWS review P2).
  - It is exact in every path: reads, write race, readout pointers and writer slot values (brute-force algebra 4.8e-7; whole-member normalisation, causality and carry contracts).
  - Held until the K4 decay run's long-lane result shows whether accumulation remains a defect.
- **GPU path:**
  - carried memory: contiguous overlapping lanes from the loader (unit-tested), carry in the checkpoint, carried final scoring;
  - slot level 1 and firing flags;
  - slot memory forced to fp32 under autocast (chunk factors span e^±63; its mass is a cumsum); matches fp32 exactly.
- **AWS's formal qualifications are accepted:**
  - new information is neither necessary nor sufficient for C4 (the bilinear counterexample);
  - C3 is an initialisation statement, not convergence;
  - failed argmax composition does not mean races cannot compose through values, posterior features or generated events.

Still open:
- a sparse reads arm (`--read-topk`) at 16M;
- event-driven sparse execution (wall-time savings from firing and slot sparsity);
- AWS's own P1 items in the native local learner: direct producer-input edges, nonlinear alternative returns, and the critic's moving feature coordinates.

## 10. Fifth round, 17 October: sparse reads and sublinear discovery

- **Sparse level reads with read probes** (`--read-topk k --read-explore E`). A level's read is a hard race among the top-k slots. In training, E unread slots s ≤ t get a zero-forward weight p − sg(p), with p = exp(score − logZ_topk), and the read becomes r − (Σ probes)·sg(r). The gradient with respect to p is v_s − r: the first-order value of adding s to the mixture. Without probes a top-k read never credits unread slots (C7). Contracts: eval law exact and causal; probes leave the training forward unchanged and credit the level's keys.
- **Slot level 1: sparse reads with probes** (`--slot-read-topk k`) and **product-key addressing** (`--slot-pk 1`). The write and read races find their exact top-k over n² slots from 2n half-key scores, so 4,096 slots cost 256 key scores per token instead of 8,192. Contract: exact top-k (1.8e-7).
- **Event-driven execution** (`SlotMemory.event_forward`, inference). Per token it writes only the m chosen slots, with lazy exact decay from each slot's last write, and delivers only the read slots. It equals the chunked execution (3e-8, also with product keys). Measured at 4,096 slots, 4 written, 8 read: 12 state rows touched per token, against 4,100 with a dense read. Training still uses the dense chunked reference (CPU wall time 0.53 s versus 0.67 s per layer step).
- **Review findings this round:**
  - Dense slot-key scoring contradicted capacity beyond activity (fixed by product keys).
  - Top-k reads without probes gave no credit to unread slots (fixed).
  - Neither e2e smoke exercised the sparse paths, and the GPU driver had not run end to end. The AWS e2e smoke v3 and a GPU-driver CPU smoke cover both.

Still open:
- event-driven execution of the depth levels (their reads over all T slots are dense scoring; product keys over slots would need learned slot keys);
- sparse training execution;
- AWS's P1 items in its local learner.
