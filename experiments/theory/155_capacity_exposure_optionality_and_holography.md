# Capacity, exposure, optionality and holography (§§422–425)

5 October 2026. These sections extend §§152, 159–163 (optionality as a continuation value over an attainable set that
includes learner updates), note 87 (the conditional exposure model) and §§419–421 (recruitment of unused slots and
routes). They are derivations under stated simplifying models and an experiment program. No benchmark result is
claimed.

**User question.** Reason about optionality learning rules for memories and for routing, advance the formal theory, and
design experiments that verify optimal use of representational capacity at every level. Dense networks are
"holographic": every example shapes every weight. A sparse substrate cannot be, since a route sees only the examples
routed to it. How should it approximate that behaviour?

---

## 422. Exposure economics: what sparse capacity costs and what it buys

**Model (parametric, local).**
- Unit u has d_u private parameters. After n_u training examples routed to it, its excess loss on its own traffic is
  approximately κ·d_u/n_u. This is exact in expectation for least squares, where it is σ²d/(n − d − 1), and the
  standard asymptotic rate for regular models.
- Shared parameters, of dimension d_0, see all N examples.
- Traffic shares are p_u, so n_u = p_u·N.

**Proposition 422.1 (traffic-weighted cancellation).** The average estimation excess loss over examples is

    E_est = κ·d_0/N + Σ_u p_u · κ·d_u/(p_u N) = κ·(d_0 + Σ_{u: p_u>0} d_u)/N.

- It depends on *which units are used*, not on how traffic is divided among them.
- Each recruited unit adds κ·d_u/N, however small its share; a dormant unit adds nothing.

**Corollary 422.2 (when unused capacity is optimal).**
- Let A(S) be the approximation loss with the set S of used units.
- Recruiting unit u is worth it if and only if ΔA_u = A(S) − A(S ∪ {u}) > κ·d_u/N.
- With little data, dormancy can be optimal. With much data, it is a credit failure.
- **Numbers.** A p32 unit has d_u ≈ 4·32² + 3·32 ≈ 4.2K private parameters (input, output, gate and key_read maps;
  control, key). On 10M events, κ·d_u/N ≈ 4·10⁻⁴·κ.
- So at our data scales, almost any approximation gain justifies recruitment. **The observed under-use is a learning
  failure (§419 Propositions 1–3), not an optimal allocation.**

**Corollary 422.3 (what sparsity buys).**
- At equal *total* parameters, the estimation cost equals that of a dense network: κ·P/N with P = d_0 + Σ d_u.
- The sparse network pays only the active d_u per example in compute.
- At equal *active* compute, a dense network has fewer parameters (lower estimation cost, higher approximation loss).
- Capacity beyond activity is therefore a claim about approximation per unit of compute. It pays only if the
  recruited units carry distinct, useful functions: the approximation term must fall.

**Corollary 422.4 (tying and low-rank private maps).**
- Tying (§398) moves most parameters into d_0, which every example trains, and leaves keys and state private.
- A low-rank private map θ_u = B·c_u (shared B; private c_u of rank r) gives d_u ≈ r·(rows + cols).
- Both lower the price of each recruited unit, making Corollary 422.2's recruitment condition easier to satisfy. They
  also remove §419 Proposition 3 (untrained losers), because B is trained by everyone.

## 423. A learning-value rule for routes (parametric optionality)

**Setting.**
- Unit u has n_u past exposures, and m_u expected future exposures over the remaining training.
- Routing the current example to u lowers the excess loss of all its future traffic. Summed over the remaining
  horizon,

      L_future(u) ≈ Σ_{k=0}^{m_u} κ·d_u/(n_u + k) ≈ κ·d_u·ln((n_u + m_u)/n_u).

- The marginal learning value of one more exposure now is

      B_u = −∂L_future/∂n_u ≈ κ·d_u·m_u / (n_u·(n_u + m_u)).

**Rule 423.1 (learning-aware choice credit).**
- Replace the choice credit F_i − R (or its linearization g·(v_i − v̄), §413) by (F_i − B_i) − Σ_j π_j (F_j − B_j).
  This is the §159 continuation value with the learner's own update included, evaluated under the exposure model.
- **Properties:**
  1. B_u ≈ κd_u/n_u while m_u ≫ n_u: an inverse-exposure bonus, the value-of-information analogue of a UCB variance
     term in squared-loss units.
  2. B_u → 0 as training ends (m_u → 0), so the rule anneals itself.
  3. **Recruitment schedule.** A unit whose immediate loss gap is Δ = F_u − F_best keeps being tried while
     Δ < κd_u/n_u, i.e. for about n* ≈ κd_u/Δ exposures. It is given enough exposure to prove itself, then abandoned
     if it doesn't. This is the formal version of the user's "recruit the high-optionality, low-confidence route".
  4. For tied units, B_u is small (d_u is small): they are cheap to recruit and need little exposure.
- **Implementation.**
  - Per-unit expected-exposure counters n_u ← n_u + Σ π_u, summed over lanes and events; no extra forward work.
  - m_u = (remaining events) × the target share 1/U, an aspiration that avoids a policy-dependent fixed point.
  - The detached bonus B_u is added to the race score in training only.
  - κ is a scale knob, set from the observed loss scale.
  - This is a parameter-level optionality index, complementing the slot-level ones of §419.

**Caveats.**
- The κd/n model is local and asymptotic. Early exposures of an untrained unit follow a different curve.
- B_u values *estimation*, not the approximation gain ΔA (which unit specializes usefully); the measured immediate
  loss supplies that through F_u.
- Interaction terms between units are ignored. Two recruits can split a region usefully or compete for it.

## 424. Slot-level optionality: the overwrite cost and its estimators

- Within an episode, writing slot i costs the future value of the content it destroys:
  C_i = E[future loss | slot i overwritten] − E[future loss | slot i kept].
- **First order (truncated to the credit window):** C_i ≈ −G_i·(m_new_i − m_i)_content, where G_i is the
  backpropagated future gradient on slot i's memory. This is `linear_write_credit` with D restricted to replaced
  content. Counting lazy decay as change caused the divergence of linear_rw (findings, 3 October).
- **Proxies, in decreasing fidelity:**
  1. A *learned slot-value head* trained to predict |G_i·m_i| (or the truncated future reduction in loss attributable
     to slot i) from the slot's state, age and usage; the bonus is −b·V̂_i. This is the critic analogue, an optionality
     index with a learned scale.
  2. Staleness 1 − retained fraction (§419 R7).
  3. Never-written (§419 R3).
- C_i is never negative for a free slot, and is zero only if nothing useful is lost. All three proxies respect that
  ordering.

## 425. Holography and its sparse substitutes

**What dense networks get.**
- Every example's gradient touches every parameter, so information about each example is spread over all weights
  (superposition).
- More importantly, features are *shared*: what is learned from one example transfers to every other through the same
  weights.
- Under Proposition 422.1 the estimation cost of a dense network with P parameters equals that of a sparse one with P
  total parameters. Holography's advantage is sharing of structure (approximation through common features), not
  estimation as such.

**Why a sparse substrate cannot copy it directly.** A route sees only its traffic. Unrealized alternatives see the
example only through credit; under the §413 linear credit, score credit only.

**Five substitutes, each with a resource charge:**
1. **Shared basis, sparse coordinates (holographic in the basis).** Tied maps or low-rank private maps (§422.4). Every
   example trains B; specialization lives in keys, state and small coordinates. *Charge:* none at inference; fewer
   private degrees of freedom.
2. **Soft-exposure training phases.** The exact expected-reception member (§418) delivers Σ π_u v_u, so every unit's
   content receives gradient in proportion to π_u.
   - Training with expected reception, or a mixture with race training, then deploying winner-only, gives
     holographic credit in training and sparse inference: soft-to-hard routing.
   - *Charge:* all proposals computed in those training phases (charged as fitting work). Train/inference mismatch
     must be measured.
3. **Virtual exposure through counterfactual content credit.** Note 87's estimator
   h = π_W·dF_W + (π_I/q)·dF_I teaches a sampled alternative I's content from the current example.
   - Exposure multiplier: 1 + k for k sampled alternatives.
   - *Charge:* k extra branch forward and backward computations per race.
4. **Consolidation ("sleep").** Periodic offline passes replay stored examples through forced or low-temperature
   routing, so dormant routes are trained on data they never saw. The forms are distillation from the active routes
   or direct supervised replay.
   - This is the substrate's analogue of systems consolidation, and the project's name.
   - *Charge:* replay compute; the size and protocol of the replay buffer must be declared.
5. **Learning-value routing (§423).** Exposure is directed to where its marginal value is highest, rather than spread
   uniformly.

**Optimal use across levels.** Capacity exists at six levels:
1. slot state within an episode;
2. routes and their private parameters;
3. heads;
4. layers (depth);
5. payload channels inside a unit (which are dense, hence locally holographic);
6. time: delays, clocks and memory horizons.

At an optimum under a total resource budget, the marginal loss reduction per unit of resource should be equal across
levels (a KKT condition). A level with a markedly higher marginal value is under-provisioned or under-used.

**Measurement per level:**

| Level | Utilization statistic | Marginal value probe |
|---|---|---|
| Slots | slots written per run, dead-slot fraction, occupancy entropy | pool sweep at fixed active work |
| Routes | expected-exposure distribution (Gini of Σπ), dead-route fraction | pool / tie / rank sweeps |
| Heads | inter-head output correlation, per-head ablation Δloss | heads sweep |
| Layers | per-layer ablation Δloss, skip-gate openness | depth sweep (with skip init) |
| Payload | effective rank of values and memories | payload sweep |
| Time | spread of learned delays, half-life quantiles (§419) | horizon init (`--tau-max`) |

---

## Experiment program (pre-declared; cheapest decisive first)

Every arm reports the four FAS rules (or test bpc with the 2–4-character history-curve bin for language), occupancy,
exposure Gini and half-lives, and charges fitting and inference work.

| # | Experiment | Arms | Task / host | Decides |
|---|---|---|---|---|
| X0 | Level audit instrumentation | on saved weights: payload effective rank, head correlation, layer ablation, exposure Gini | curie, evaluation only | which level is under-used |
| X1 | **Capacity-utilization curve** (decisive for "capacity beyond activity") | pool {2, 8, 32} × {untied, tied, tied + best §419 knob} at fixed active work | FAS v1 (R0–R2 cover part) | whether loss falls monotonically with available capacity |
| X2 | Learning-value bonus (§423) | κ ∈ {0, .1, 1} with tied pool 8 | FAS, then language p32 pool 8 | route-level optionality |
| X3 | Soft-exposure training (§425 item 2) | expected-reception training → winner-only deployment, versus race-only at equal charged work; with mixture schedules | FAS, language p64 | holographic credit with sparse inference |
| X4 | Shared-basis rank | private rank r ∈ {0 (tied), 4, 16, full} at pool 8 and 32 | FAS | the price of private capacity (§422.4) |
| X5 | Virtual exposure | sampled alternative content credit, k ∈ {0, 1, 2} | language p32 pool 8 | exposure multiplier versus cost |
| X6 | Stabilized write credit (§424) | D = replaced content, credit clipped; versus the staleness proxy | FAS pool 8 | slot-level overwrite cost |
| X7 | Consolidation (§425 item 4) | periodic replay through forced routes, buffer sizes {1K, 10K} episodes | FAS | dormant-route training without live traffic |

**Gates:**
- X1 is the headline. The best variant must improve monotonically from pool 2 to 8 to 32 at fixed active work, by
  more than the seed spread (two seeds at the final pool).
- A knob from X2–X7 is promoted if it moves X1's curve, by lowering loss at large pool or steepening the slope, without
  worse validation loss at pool 2.
- Losses and null results are recorded beside the derivations.

---

## 426. Holography, defined: the transfer kernel

**User question.** We need a sparse holography. How is it defined formally? What do we lose when items are stored on
non-intersecting routes? How does the degree of holography affect grokking? How can holography be approximated across
sparse routes without making routing dense: a sub-holography, a mixture of holograms?

**Definition 426.1 (transfer kernel).**
- One gradient step on example x changes the prediction on x' by δf(x') = −η·K(x, x')·∂L_x/∂f, where
  K(x, x') = ⟨∇_θ f(x), ∇_θ f(x')⟩ is the neural tangent kernel, restricted to the parameters the two examples touch.
- *All* cross-example transfer, positive (generalization) and negative (interference), passes through K.

**Definition 426.2 (holographic degree).**

    H = E_{x≠x'} |K(x,x')| / E_x K(x,x).

- A dense network has H of order one (every example moves every weight).
- A model whose examples use non-intersecting parameter sets has K(x, x') = 0 across sets, so H ≈ Pr[same route].

**What is lost with non-intersecting routes (Proposition 426.3).**
- If supp ∇f(x) ∩ supp ∇f(x') = ∅ whenever r(x) ≠ r(x'), training on x never changes the prediction on x'.
- Each route is an independent learner on its own traffic n_r = p_r·N. It gets *zero interference and zero transfer*.
- **Losses:**
  1. *Data efficiency.* Shared structure must be relearned on every route. If the structure has dimension d_s, the
     estimation cost rises from κd_s/N to κ·U_used·d_s/N (§422.1).
  2. *Generalization to new combinations.* A test item routed to r is predicted only from r's training items.
  3. *Graceful degradation.* There is no redundancy in superposition.
- **Gained:** no interference, hence no catastrophic forgetting between routes, and isolated memorization capacity.

**The ideal is neither extreme.** Let a(x, x') = cos(∇L_x, ∇L_x') be the target alignment: whether the two examples
want the same update. The ideal kernel transfers where a > 0 and isolates where a < 0. Holography should be
*selective*, high between aligned examples and low between conflicting ones. A dense network shares everything and
relies on cancellation; disjoint routes share nothing.

## 427. Holographic degree and grokking

**Setting.** Grokking (Power et al., 2022) is the late transition from a memorizing to a generalizing solution, driven
mainly by weight decay. The generalizing circuit has a lower norm than the sum of per-example memorizers.

**Proposition 427.1 (no cross-route grokking without overlap).**
- With disjoint parameter blocks, the minimum-norm interpolant decomposes into independent per-block solutions. No
  shared, lower-norm rule exists to grok toward across blocks.
- Weight decay only shrinks each block's memorizer.
- Grokking can occur *within* a route, on its own traffic n_r, so the data each route needs to grok grows with U.

**Proposition 427.2 (overlap enables grokking).**
- When routes share a basis (θ_u = θ_0 + B·c_u), a rule expressed in θ_0 serves all examples at one norm cost.
- A memorization spread over the c_u costs norm per item.
- Weight decay therefore favours migrating structure into the shared part, the sparse analogue of the grokking
  transition. **Prediction:** with more holographic overlap, grokking becomes possible and faster. With pure isolation,
  the model memorizes without grokking.

**Asymmetric decay as a sparse grokking driver (Proposition 427.3).**
- Weight decay λ_p on private parameters greater than λ_s on shared ones makes private storage costlier than shared.
- Information that *can* be expressed in the shared basis migrates there over training. Item-specific residue that
  cannot stays private.
- This is the compression half of the user's §420 picture: memorize in private routes, then compress into the shared
  hologram.
- §425 consolidation replay accelerates the migration by distilling private into shared.

## 428. Sparse holography: a mixture of holograms

**Construction 428.1 (our routes are already combinatorial codes).**
- An example's route is the tuple of winners over all (layer, head) positions, so there are U^{H·D} route codes.
- Two examples share a unit's parameters at each position where they pick the same unit. Their transfer kernel is

      K(x,x') = K_shared(x,x') + Σ_{(l,h)} 1[r_lh(x) = r_lh(x')] · K_{u}(x,x').

- For unrelated examples with balanced routing, the expected overlap is Σ_{positions} Σ_u p_u² ≈ H·D/U positions. For
  similar inputs (similar keys) it is higher.
- This is Kanerva's sparse distributed memory and locality-sensitive coding, realized by learned races (Kanerva 1988:
  each item stored in several of many hard locations, retrieved by summing them).

**The knobs, and what each does to the kernel:**

| Knob | Effect on K | Compute |
|---|---|---|
| Pool U ↑ | per-position collision falls ~1/U: more isolation | flat (winner-only) |
| Heads H ↑ (at fixed H·P) | more positions: finer-graded overlap, more "mixture components" | flat at fixed H·P |
| k-winner races (first k arrivals deliver, values summed) | each item in k locations: explicit SDM | ×k delivered |
| Tied or low-rank private maps | adds a dense K_shared; private blocks smaller | flat |
| Soft exposure in training (§418 expected reception) | dense K in training, sparse at inference | training only |
| Key smoothness (temperature, key norm) | overlap tracks input similarity (LSH quality) | flat |
| Asymmetric decay λ_p > λ_s | moves information from private to shared over training | flat |

- **"Mixture of holograms":** each (layer, head) position is a small hologram over its pool, and an example is a
  superposition across positions.
- **"Sub-holography":** a shared dense basis (the hologram) plus sparse private coordinates.
- Both are realized by the same substrate; they differ in where the parameters live.

**Optimal holographic degree.**
- Transfer should follow target alignment a(x, x').
- The routing learns this if co-routing is credited by the realized benefit of shared updates. Counterfactual credit
  supplies it at the route level: routes whose shared parameters reduce other examples' loss gain score.
- A cheap diagnostic is the correlation, over example pairs, between route-code overlap and gradient alignment. It
  should be positive.

---

## Grokking and holography testbed (pre-declared; small, fast)

**Task:** an algorithmic event task with a known shared rule. Modular addition mod p = 97, as event sequences
(a, b, =) → c, with a training fraction of 30–50%. This is the standard grokking setting.
- It measures memorization (training accuracy) and generalization (held-out accuracy) over long training with weight
  decay.
- The native integrated core is used. Dense MLP and Transformer controls are diagnostic, run on AWS per AGENTS.md.

| # | Arms (equal active compute) | Prediction |
|---|---|---|
| G1 | pool U ∈ {2, 8, 32}, untied | larger U: memorizes faster, groks later or never (§427.1) |
| G2 | same with tied maps | groks at every U (§427.2) |
| G3 | heads H ∈ {2, 4, 8} at fixed H·P | more positions: earlier grokking (graded overlap) |
| G4 | asymmetric decay λ_p/λ_s ∈ {1, 10} with low-rank private maps | λ_p > λ_s: earlier grokking, information moves to shared |
| G5 | k-winner races k ∈ {1, 2} | k = 2: higher H, earlier grokking at ×2 delivery |

**Measurements:**
- Training and held-out accuracy curves, and the step at which held-out accuracy crosses 50% and 90%.
- Holographic degree: sampled NTK overlap on 256 example pairs.
- Overlap–alignment correlation.
- Private versus shared norm over training.

**Gates:** the predictions above, each with two seeds. A failure of G2 (tied maps still do not grok) would falsify
§427.2 for this substrate, and must be recorded as such.

## Testbed results, first pass (5 October; train fraction .40; `experiments/results/grok/`)

| Arm | Held-out ≥ .99 at step | Final private / shared norm | Holographic degree; overlap–alignment corr. (last) |
|---|---:|---|---|
| G1 untied, pool 2 | 2,000 | 26.5 / 19.1 | .173; .08 |
| G2 tied, pool 2 | 2,500 | 1.6 / 27.7 | .182; .02 |
| G2 tied, pool 8 | 3,000 | 1.9 / 28.2 | .179; .10 |
| G1 untied, pool 8 (operator stop after generalization) | 3,000 | 28.8 / 26.7 | .160; .29 |
| G4 tied pool 8, λ_p/λ_s = 10 | 3,000 | 0.4 / 26.9 | .212; .05 |

**Reading against the predictions:**
- **§427.1:** pool 2 generalizes earlier than pool 8, in both the untied and tied variants. *Consistent.*
- **§427.2 (overlap enables grokking):** every arm generalizes, and the tied arms are not earlier. *Not discriminated
  at this training fraction.*
- **§427.3 (asymmetric decay earlier):** G4 shows no change in timing, although its private norm collapses to .4.
  *Not supported at this setting.*
- **§428 (route overlap tracks update alignment after learning):** the strongest signal is G1 pool 8 (−.01 before,
  +.29 after); the other arms are small. *Mixed.*

**Status:**
- The task is too easy at .40: the memorization lag is only 250–500 steps, so arms do not separate.
- A harder pass (train fraction .25; G1/G2 pools 2 and 8, plus G4) is queued after the curie chain.
- The remaining .40 arms (G3 head count; pool 32 tied and untied) are running.
