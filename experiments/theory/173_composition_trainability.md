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

The earlier `RMR.slot_values` evaluated the old-correction contribution under `torch.no_grad()`; the current driver repairs it with exact accumulated-slot credit.
The stopped version informed per-target writing but omitted its derivative; the current slot accumulator retains that first-order VJP.
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

## 11. Sixth round, 17 October: where the remaining dense work is

**Active work per target** is now reported in every RMR and GPU-driver result (`member_work`). It is modelled event-driven work at history H, with firing rates measured on a flagship window, and broken down into key scores, values delivered, state rows written and read, and MACs.

Modelled for the queued ladder at d 128, K = 4, history 1,023, plus 256 carried slots where carry is used (one model instance each, untrained, fire rate 1):

| Member | MMACs per target | State rows moved | Level-1 key scores / rows read | Level read keys / values | Readout episodic candidates |
| --- | ---: | ---: | --- | --- | ---: |
| K4 residdecay (reference) | 3.40 | 3,087 | dense recurrence | 1,024 / 1,024 | 2,046 |
| Integrated | 4.51 | 3,855 | dense recurrence | 1,280 / 1,280 | 2,558 |
| Native L1, 4,096 slots, dense read | 6.63 | 12,055 | 8,192 / 4,096 | 1,280 / 1,280 | 2,558 |
| Sparse native | 4.06 | 87 | 256 / 8 | 1,280 / 16 | 2,558 |

**Finding.** Sparse native cuts state traffic 44× against the integrated member, but MACs barely move. What remains is **discovery over history**: every level read scores every past slot (H keys, plus the old-correction algebra), and the readout's episodic race scores 2H pointer candidates. That O(H)-per-token scan is the attention-like cost left in the member, and it grows with context.

**Design for the next step: indexed history discovery** (not implemented; it needs evidence that the sparse arm holds quality first).
- Each memory slot is filed, when written, into one of C regions by a race of its key against C region keys (product keys: 2√C scores). The filing is hard; region keys move with training, VQ-style.
- A read races over regions first (2√C scores), then over the slots in its top-b regions (about H·b/C candidates), using exact slot scores including corrections. The two-stage score is S_ts + log π_q(region(s)), so region keys get credit from the query side.
- Probe regions (zero-forward) give credit to regions not read.
- The same index can serve the readout's episodic race, with the normaliser taken over the causally chosen candidate set, so the law stays normalised. The risk is losing copy sources outside the candidate set; the contract must measure copy-hit coverage.
- Prior art for clustered attention (e.g. routing attention, Roy et al. 2021) is cited. What is ours: memory filed by races at write time, with refinement, decay and probes.
- Contracts needed: normalisation over the candidate set; causality (regions assigned at write time from data at or before t); exactness when b = C; coverage of copy hits; work per token O(√C + H·b/C).

**Review findings this round:**
1. Results lacked the active-work breakdown AGENTS.md requires; fixed.
2. The AWS queue holds 8 16M arms plus 2 smokes, about two days of slot time, and no arm has started. The per-target and 1,024-lane arms are attributable inside the ladder (fixes → integrated) and are deprioritised, so the decisive ladder reads out sooner.
3. No new mechanism is admitted until the ladder reports. Mechanisms added without results would now be busywork.

## 12. Seventh round, 17 October: indexed history discovery implemented (held)

- **Level reads and write races** (`--index-regions C --index-read b`). Each slot is filed, from the level's input memory (causal), into the region its key wins (hard, VQ-style). The read race and the write-address race score C regions first and then only the slots in their top-b regions, with score S + log π(region). The current slot is always readable. Read probes reach unread regions through log π, which credits region keys. Stats: candidate fraction, largest region share, filing entropy.
- **Readout pointers** (`--readout-index C --readout-read b --readout-tok K`). The same two-stage race goes through the CERL hook (S_rep_add / S_cont_add carry log π and −∞ outside the candidates), so the law stays exactly normalised over the causal candidate set. An exact token-identity lookup always makes continuation pointers from the K most recent earlier occurrences of the current token candidates, so induction copies stay reachable. Stat: readout copy coverage, the share of targets with an earlier copy source that keep one among the candidates.
- **Contracts** (v14): normalised; causal; carry equals one long lane; reading all regions recovers full coverage (1.0); region keys credited. At initialisation with 2 of 4 regions, coverage is 0.55 on random 8-token data.
- **Modelled work** (sparse native, d 128, K 4, C 64, b 4): 4.06 → 1.74 MMACs per target at history 1,023, and 18.7 → 2.65 at 8,191.
- **Review findings this round:**
  - indexed reads left the write race dense (fixed);
  - copy coverage is the quality risk (token-identity lookup plus a coverage stat; the arm's pass criterion is coverage ≥ 0.95);
  - a fixed C only flattens the slope; sublinear work needs C ∝ √H;
  - filing can collapse onto few regions (stats reported; a filing balance is added only if observed);
  - AWS's separate writer-input variant (`r1_rmr_writer_input_credit.py`, checkpoint recomputation, 2.75 GB at its gate) duplicates the exact O(T d) fix in `r1_rmr.slot_values` and can retire.
- **Arm** `aws_r1_rmr_nativeindexed_K4_16M_s0_20261017_v1`: held until the sparse arm holds quality.

## 13. Eighth round, 17 October: fix all

- **Region index scaling:** product-key regions (`--index-pk 1`). C = n² regions are scored from 2n half-keys. π is separable, so log π(a, b) = log softmax(s1)[a] + log softmax(s2)[b] exactly (contract: sums to 1 within 1.2e-7), and the top-b regions come from the exact product-key top-k. C can grow as the square of the scoring cost, which is the √H rule.
- **Filing collapse:** balanced filing (`--index-balance γ`). Per-region filing biases are nudged toward uniform load after each training forward (auxiliary-loss-free balancing, as in DeepSeek-V3). They are buffers (checkpointed), fixed in evaluation, and filing stays causal. Contract: biases move only in training; law exact, causal, carry exact.
- **Resume identity:** `--product-keys` (added to the RMR command line in round 3) was missing from the behaviour-preserving defaults, so the running K4 (Curie) and K8 (AWS) runs could not have resumed under current source. Fixed and verified against the live checkpoints (K4 decay and PTB).
- **No results yet:** Curie runs the sparse native member itself after the K4 decay run (chain v33: e2e smoke → RSS/time gate → run → PTB resume). AWS skips that rung.

## 14. Depth and data: the two halves of the slope (17 October, founder)

Founder: the slope gap is about depth, and depth is two things: (a) effective learning over compositional depth, and
(b) adding depth as data grows, because useful depth tracks roughly the logarithm of the data: each level represents
one more layer of hierarchical decomposition, and a level is only learnable once the data contain enough instances of
its parts. The random hierarchy model (Cagnetta, Petrini, Wyart et al.) gives this form: resolving L levels of a
hierarchy with branching m needs a sample count that grows like m^L, so the resolvable depth with D samples grows like
log D / log m, and loss falls stepwise as each further level becomes resolvable.

What our runs already say (one seed, FineWeb development, one pass):

| data | dense depth gain | native depth gain |
| --- | --- | --- |
| 4M | L2 → L8: 0.024 (5.865 → 5.841) | K = 2/4/8 vs CERL: within 0.01 |
| 16M | L4 → L8 → L16: 0.052, 0.015 (5.270, 5.218, 5.203) | K = 4 old-slot vs CERL: 0.073 |

Both families gain more from depth as data grow, and both saturate early at a fixed data size, which is (b). Depth
alone does not explain the whole slope gap at 4–16M: the shallow dense model gains 0.595 per 4× data (L2 at 4M 5.865
→ L4 at 16M 5.270) against CERL's 0.403, and the best-depth envelopes are 0.638 (dense) vs about 0.47 (native). The
remaining candidate is usable capacity: CERL has more parameters (7.0M vs 3.8M) but used 124 of 1,024 concept units
(zero init, no balance, weight decay on dormant tables, fixed since). A model that cannot put new statistics into new
capacity flattens with data whatever its depth.

Consequences:
1. Compare scaling by envelopes: for each data size, the best depth of each family. A fixed-depth curve against
   another fixed-depth curve mixes slope with depth mismatch.
2. Test (a) apart from data on a synthetic hierarchy with known depth L: does a K-level member reach the m^L sample
   scale, and does removing level k remove one hierarchy level? This separates credit failure (C1–C9) from data
   sufficiency, which language at 4–16M cannot do.
3. Grow depth with data. A new level enters at its ReZero gate 0, so insertion is exactly function-preserving (C3):
   one run can add a level each time the data multiply by a fixed factor, and is compared with fixed-K runs at the
   endpoints.
4. Make depth cheap through timescale. If level k fires at the rate of its hierarchical events, about m^-k per token,
   active work for K levels is a geometric sum bounded independently of K, while a dense stack pays every layer on every
   token. This needs durable upper-level events (the CERL-H dead end: causes changed on 79% of tokens) and measured
   firing rates per level (`--fire`, untested in training).
5. The capacity lever is tested by the fixed 16M capacity arm (`--wd-groups 1 --unit-init 0.02 --balance 0.01`):
   the slope from 4M to 16M must move toward the dense slope if usable capacity is the cause.

Predictions to falsify: K = 8 does not beat K = 4 at 16M by more than about 0.02 but does at 64M; the fixed-capacity
arm steepens the 4M → 16M slope; a grown-depth run matches fixed K = 4 at 16M with less training work.

Implemented 17 October (founder: be guided by the results and fix the weaknesses observed):
- Usable capacity is measured in every CERL/RMR milestone (`capacity`: concept units with at least a tenth of the uniform
  responsibility share, unit perplexity, share of the busiest 1%). The responsibilities add up to the concept cause's share
  (contract, 1e-8).
- Depth growth `--grow f2,...,fK` in `r1_rmr.py`: level k enters at data fraction f_k at ReZero gate 0. Insertion is exact
  (contract 0.0, including a level entering after a carried prefix); levels not yet in use receive no credit and cost
  nothing; training FLOPs are counted at the depth in use at each step.
- `experiments/rhm_depth.py`: the synthetic-hierarchy test of consequence 2, with the exact block-truncated Bayes ladder
  B_0..B_L (equal to brute-force enumeration to 2e-16) and the resolved composition depth of each model and of each
  level knockout. Hierarchy v 16, m 4, s 2, L 4: B = 2.60, 2.03, 1.71, 1.55, 1.47 nats per leaf (each level adds about half
  the information of the one below).
- Queued: the hierarchy grid (K = 1, 2, 4 at 1k-64k trees) and the fixed CERL pair at 4M/16M on Curie; the grown K4 16M arm
  and the fixed CERL 64M slope point on AWS.

## 15. Inductive bias and scaling (17 October, founder)

Founder: model classes that scale well tend to lose to models that scale badly on small datasets (n-grams vs neural LMs, LSTMs
vs Transformers, CNNs vs ViTs below JFT scale). The exception is a bias that only speeds learning and fades with data (ConvNets
match ViTs at JFT-4B scale, Smith et al. 2023): a soft bias costs nothing at scale, a hard constraint flattens the curve.

Our own runs show the pattern inside the model (one seed, FineWeb DEV, selection slice):

| | CERL 4M | CERL 16M | K4 4M | K4 old-slot 16M |
| --- | --- | --- | --- | --- |
| native gain (race causes over background only) | 0.437 | 0.405 | 0.433 | 0.386 |
| background responsibility | 0.24 | 0.32 | 0.24 | 0.36 |
| background-only NLL | 6.107 | 5.688 | 6.108 | 5.577 |

The generic background route carried most of the 4M → 16M improvement (0.42 of 0.39 nats for CERL); the race causes are a
near-constant 0.4-nat offset whose share does not grow. Rules from here:
1. Small-data wins are not scaling evidence. Scaling claims need fitted curves over at least three data sizes on the compute
   axis; matched-compute wins are reported for their budget only.
2. Every result is read for per-cause share against data: a mechanism whose share shrinks with data is a small-data prior; a
   mechanism meant to scale must take a growing share.
3. The concept bank is the parametric memory that should grow with data (capacity beyond activity). Pass condition for the
   fixed CERL pair (curie_r1_cerl_fix_{4M,16M}): units_used and the concept share both rise from 4M to 16M. If not, the bank
   is a hard constraint and is redesigned before any 64M run.
4. Size must scale with data in frontier comparisons: in this family through concept units or slots at near-constant active
   work (product keys), and depth through growth (section 14). Fixed-size 4M → 16M comparisons understate any family whose
   capacity is held small.

## 16. Both credit paths in residual memory (17 October, founder question)

As in a residual Transformer (x_{l+1} = x_l + f_l(x_l), so dL/dx_l = dL/dx_{l+1}(I + df_l/dx_l)), a level's corrections get credit
by two paths: directly, because the readout reads the total memory, and through every upper level that reads them as
conditioning data (its query is the current slot under the running total; its read race scores and reads slots that include
lower new-slot corrections; lower old-slot corrections change the keys and values it reads). Nothing on these paths is detached
(only zero-forward probes, straight-through firing, statistics and the carried prefix). The conditioning path is closed at
init (upper ReZero gates 0) and opens with the gates, as in ReZero Transformers; within one level, own corrections are visible
from earlier chunks only (training parallelism). Contract (curie_rmr_contracts_20261017_v17): the derivative of level 3's
corrections with respect to level 2's new- and old-slot corrections is nonzero, and gating level 3 off changes the gradient on
level 2's corrections (share 12% at the contract's random parameters). CERL-H and the detached local learner (DEAD_ENDS) lacked this path.

## 17. The event spine composes with depth (17 October, founder)

Founder: levels must exchange both memories and events; the async spine must also scale with depth and composition, or it
degrades until it is removed. Memory composes by addition. Events are pointers in time, so they compose by following
pointers: level k at t points to slot s (its read-race winner when it fires); level k's event at s points to ptr_k(s). Level
k+1 races over its own candidates plus ptr_k(t) and the composed pointer ptr_k(ptr_k(t)), so its own pointer can realise a
two-hop chain; K levels reach chains of length about 2^(K-1), and the depth needed grows with the logarithm of the chain.
Implementation (`--events 1`, r1_rmr.py): learned bonuses on the two pointed slots (always candidates under sparse reads),
their ages and the event arrival (the lower attribution moved) in the query, event arrival as a firing input (a silent level
holds its last pointer: event-driven state), the top level's pointer and composed pointer in the readout's repeat/continue
race. Credit: straight-through weights (forward 1) carry the upper use of an event back into the lower read race. Zero at init
equals the member without events (C3). Contracts v18: identity 0.0, normalisation 1 +- 1e-7, causal, train = eval, carry exact,
nonzero event credit to the lower read keys, composed reach >= pointer reach. Reported per level: event rate, pointer reach,
composed reach. Durability target (CERL-H dead end): event rate falling with level.

## 18. Synergy and attribution (17 October, founder)

Founder: some features are synergistic with others, so they may not work in isolation. Designed pairs: sparse reads need probes
(unread slots get no credit); firing needs events (nothing to trigger on) and events need firing (token-rate events carry nothing,
the CERL-H failure); slot level 1 needs carry (slots reset every lane cannot pay); carry needs cross-lane credit (declared gap); a
large concept bank needs symmetry breaking and balance (the 4M product-key banks collapsed); event pointers make sparse reads reach
far slots cheaply. A mechanism added alone to the dense-read stack can show nothing even when the construction needs it.

Method from here: (1) the headline comparison is the full native member against the diagnostic dense-read stack (K4 old-slot
4.9901, K4 decay); (2) attribution removes one whole bundle from the full native member (sparse addressing, async spine,
persistence, capacity, writes, local learning; `# Role: ablation <bundle>`, checked by scripts/check_native_coverage.py so that
exactly that bundle is off); (3) a seed replicate of the full member sets which differences are real; (4) no mechanism is declared
dead from an isolated test on the diagnostic stack, or from a 16M leave-one-out alone (synergy can depend on data scale, section 15).
The native member now also carries the large concept bank (65,536 units, product-key wake), which it lacked.
