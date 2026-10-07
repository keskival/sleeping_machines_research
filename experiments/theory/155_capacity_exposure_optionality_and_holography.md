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

---

## 429. Credit fidelity: measure the learning rule against exact counterfactuals, then repair the largest error

**User direction (5 October).** Leaderboard wins should not be expected before the systematic problems in the learning
rules are fixed so that the model uses its full capacity. Advance understanding in a way that leads to the fix.

**Principle.** Every hypothesized defect (§§419, 424, 425 and the FAS/language results) is a claim that the implemented
credit differs from the true credit in a specific way. Our substrate allows that difference to be measured exactly:
forcing one race to a chosen alternative while sharing all other noise draws gives the true counterfactual loss of
that choice. So defects can be *measured* on trained models before repairs are trained.

**Decomposition.** For a race r at event k with probabilities π and full-suffix losses F_u (force alternative u, roll
the episode forward), the exact choice credit is c_u = π_u (F_u − R), with R = Σ_j π_j F_j. Split each suffix loss by
horizon:

    F_u − R = Δ_u^next + Δ_u^window + Δ_u^beyond.

- Δ^next is the next prediction's loss: mostly the delivered value.
- Δ^window is the remaining loss inside the training segment: delivery effects that persist plus the write
  consequences that truncated BPTT can still see.
- Δ^beyond is the loss after the training segment ends: credit that truncated training never sees.

The implemented score gradient (timing credit plus the linear value credit, §413) is the estimator ĉ.

**Fidelity metrics, per trained model and task:**
1. corr(ĉ, c) and sign agreement over (race, alternative) pairs: does the rule point the right way?
2. The share of |F_u − R| carried by Δ^next, Δ^window and Δ^beyond:
   - a large Δ^window share means myopic delivery credit misses write consequences (§419 Proposition 2);
   - a large Δ^beyond share means truncation drops credit (FAS items span ~720 s, while segments cover ~85 s).
3. Free versus occupied alternatives: the mean exact c_u for writing a free or stale slot against an occupied one.
   This tests whether optionality is real on the data.

**Repair rule.** Each candidate repair is evaluated first by fidelity: whether it raises corr(ĉ, c) on the same races,
or captures the missing horizon share. Only then is it given a long training run. This orders the repairs by measured
error, not intuition:

| Measured largest error | Repair |
|---|---|
| Δ^window large, ĉ blind to it | write-consequence credit (§424; stabilized written-content form) |
| Δ^beyond large | longer credit windows, carried eligibility, or segment overlap |
| Exact credit favours unexplored alternatives with untrained content | content credit for sampled alternatives (note 87) or soft-exposure phases (§425) |
| ĉ accurate but quality poor | not a credit problem: capacity, binding architecture or write bandwidth |

**Tool:** `experiments/credit_fidelity_audit.py`. It uses saved selected weights and forced shadow lanes in
`batched_logits`, and runs as evaluation only.

## 430. First-order counterfactual credit with writes: the coordinate theorem behind the write-credit divergences

**User direction (5 October):** less trial and error, more formal theory. §429 supplies exact ground truth by forced
shadow lanes. This section derives what that ground truth should equal to first order, why both earlier write-credit
estimators diverged, and the estimator that is correct by construction.

**Setting.** At race r, at time t, in head h of layer d:
- slots j = 1..U hold stored memories m_j with stamps τ_j (τ_j < t, or "never written");
- the race selects W ~ π, which delivers v_W and writes slot W;
- every slot's candidate is m'_j = A_j(t − τ_j) m_j + w_j, where:
  - A_j(Δ) = Rot_j(Δ)·exp(−Δ·rate_j·f_j) is the transport (rotation and decay, with forget gate f_j);
  - w_j is the written content.
- Memory is *lazy*: an unwritten slot keeps (m_j, τ_j) and is transported only when read.

A later read of slot j at time s > t sees

    written at t:   A_j(s − t) [A_j(t − τ_j) m_j + w_j]
    not written:    A_j(s − τ_j) m_j.

**Lemma 430.1 (semigroup).** If the forget gate is constant on [τ_j, s], A_j(s − t)·A_j(t − τ_j) = A_j(s − τ_j). The
difference between the two cases at any later read is then exactly A_j(s − t)·w_j: *the written content, transported
from t*.
- Proof: rotations about fixed axes commute and compose additively; exponential decay composes multiplicatively.
- With an input-dependent forget gate the identity holds up to the gate difference, a second-order effect when gates
  vary slowly.

**Proposition 430.2 (first-order choice credit).** Let g be the loss gradient at the delivered value. Let
Γ_j(t) = Σ_{reads at s > t} A_j(s − t)^T ∂L/∂(value read at s) be the *transported state gradient*: the derivative of
the future loss with respect to a perturbation of slot j *expressed at time t*. Then, to first order,

    F_u − F_W = g·(v_u − v_W) + Γ_u(t)·w_u − Γ_W(t)·w_W + O(‖δ‖²),

and the exact first-order choice credit is

    c_u ≈ π_u [ g·(v_u − v̄) + (Γ_u·w_u − Σ_j π_j Γ_j·w_j) ].

- The first bracket term is the implemented linear value credit (§413).
- The second is the *write credit*: the value of placing content w_u in slot u rather than where the race would
  otherwise place it.
- Its sign carries optionality. For a free slot, nothing is overwritten. For an occupied slot, the content's future
  value is lost, and that loss enters through Γ_u: the term −Γ_u·(1 − A)m_u appears when the forget gate differs
  across the overwrite.

**Theorem 430.3 (why linear_rw and linear_rwn diverged).** Both surrogates add (π − π̄)·D_j to the *stored* memory
of every slot. For a slot that does not win, the stored memory sits in its *old* coordinates (m_j, τ_j). Backpropagation
therefore returns the stored-coordinate gradient G_j = ∂L/∂m_j|_{τ_j}. By the semigroup lemma,

    G_j = A_j(t − τ_j)^T Γ_j(t).

- **linear_rw** used D_j = m'_j − m_j = (A_j(t − τ_j) − I) m_j + w_j. The term (A − I)m_j is pure lazy decay: not a
  change of future reads, yet counted as one. Its magnitude grows with the age t − τ_j and the norm of m_j, and its
  sign is systematic (decay shrinks memories). The credit is then biased toward choices that "change" old slots, with no
  bound on the bias.
- **linear_rwn** used D_j = w_j, but paired it with the stored-coordinate G_j:

      G_j·w_j = Γ_j(t)·A_j(t − τ_j) w_j ≠ Γ_j(t)·w_j.

  The written content is effectively transported *backwards* to the old stamp: under-weighted by the decay
  exp(−age·rate) and rotated by the wrong phase. For old slots the phase error makes the sign of the credit for long
  ages essentially random. The bias is therefore systematic in rotation and grows with slot age. Adding pool slots adds
  old, rarely written slots, which matches rwn diverging at pool 4 but not at pool 2.

**Corollary 430.4 (the correct estimator).** Compute Γ_j(t) for every slot, i.e. the gradient with respect to a
zero-valued perturbation ε_j added to slot j *with stamp t*. Equivalently, carry a per-slot shadow perturbation channel
that is transported from t like a real write. Inverting A_j would amplify by exp(+age·rate) and is unstable; the shadow
channel avoids that.
- Implementation, zero-valued and exact to first order: at each race, let every slot j's next-read value include
  (π_j − π̄_j)·A_j(s − t) w_j, detached in w and transported from t. Backpropagation then delivers
  π_j Γ_j·w_j − Σ π Γ·w to the scores.
- Cost: one extra transported vector per slot touched, which is the same order as a read. For winner-only inference
  the cost is zero (the forward is unchanged).

**Predictions (falsifiable with the §429 audit, no training needed):**
1. On trained FAS and language models, the first-order credit of Proposition 430.2 tracks the forced-lane exact
   credit c better than the value-only credit does: higher corr(ĉ, c) and higher sign agreement.
2. The gain concentrates where the audit's window share is large.
3. Pairing D = w with stored-coordinate G (rwn) loses fidelity as the age of the slots raced over grows; the
   transported form does not.
4. Training with the transported write credit is stable at pool ≥ 4 (unlike rwn) and lengthens learned half-lives
   (forgetting stops substituting for routing, §419).

**Order of work implied by the theory:**
1. Audit the current credit (running).
2. Compute the first-order prediction and compare it with exact credit on the same races (estimator-level test).
3. Only then train with the transported write credit.

**Scope.** First order in the choice (hard switches can change later topology: route flips downstream). The audit
measures the residual. Forget-gate variation enters at second order.

**§429 first measurements (5 October):**
- R0: corr −.08, sign agreement .60; horizon shares next / window / beyond .47 / .79 / .46.
- R8: corr .29, sign agreement .64; shares .16 / .70 / .79.
- Both are consistent with §430: the omitted write term and the truncated horizon carry most of the true credit, and
  more so with longer memories.

**§430 refinement (5 Oct 14:30 UTC): two read paths, two coordinates.** The layer reads slot memory in two places:
- **Proposals** read the *transported* memory m′_j = A_j(t − τ_j) m_j + w_j. For this path, Lemma 430.1 applies: the
  difference "written versus not" at later reads is A(s − t) w_j.
- **Race scores** read the *stored* memory through `key_read · m_j`, with no transport. For this path the stored
  coordinate is what matters: a write changes it by m′_j − m_j = (A_j − I) m_j + w_j. Lazy decay *is* a real change
  for score reads (a refresh re-stamps the key; note 142).

**Corrected Proposition 430.2.** With G^score_j = ∂L/∂(stored m_j through later score reads) and Γ^value_j(t) the
transported value-path gradient:

    F_u − F_W ≈ g·(v_u − v_W) + [Γ^value_u·w_u + G^score_u·(m′_u − m_u)] − [same for W].

Restated, Theorem 430.3:
- **linear_rw** applied the stored-coordinate difference to *both* paths. That is correct for scores, wrong for values:
  decay was counted as a value change.
- **linear_rwn** applied the written content in stored coordinates to both paths. That is wrong for both: values were
  transported from the wrong stamp, and scores missed the re-stamping.

**Corrected Corollary 430.4.** Use two zero-valued shadow channels per slot:
- σ^v, transported continuously like memory, accumulating (π − π̄)·w at each race;
- σ^s, in stored coordinates, accumulating (π − π̄)·(m′ − m), and transported by A when the slot is really written.

Proposals read m′ + transport(σ^v), and scores read m + σ^s. The forward is unchanged (both channels are exactly zero).
Backpropagation delivers the full first-order choice credit to the scores.

## 431. Route chaos: long memory makes hard routing sensitive, and first-order credit fails exactly there

**Measurement** (§429 audit with route-flip counts; 18 races × 8 alternatives per model; FAS v1; 6 October). Forcing
one alternative at one race changes the winners of many *later* races:
- **R0** (untied pool 8, learned half-lives ~7 s): median **5** later winners change, 0.7% of later races.
- **R8** (tied pool 8, ~24 s): median **125**, about 16% of later races (p90 433).

Fidelity of the credit estimators against exact forced-lane credit, split at the median flip count:

| Model / half | Value-only (corr, sign) | Transported write credit (§430) |
|---|---|---|
| R0, few flips | .00, .76 | **.49, .77** |
| R0, many flips | −.19, .44 | −.27, .43 |
| R8, few flips | .49, .63 | −.08, .53 |
| R8, many flips | .23, .66 | .32, .54 |

**Interpretation.**
1. In the low-sensitivity regime (R0, few flips), the transported write credit does what §430 predicts: it adds the
   write consequence that value-only credit misses, raising the correlation with exact credit from .00 to .49.
   **§430 is supported where its first-order assumption holds.**
2. Long memories turn hard routing *chaotic*. A changed choice persists in slot memory, shifts later scores across
   decision boundaries, and flips a large fraction of later routes. No first-order (gradient) credit can represent the
   true effect, because it is dominated by discrete cascades. On R8, even the "few flips" half is chaotic (median 125).
3. **The resulting tension:** binding needs long memory (FAS item gaps of ~29 s), and long memory with hard routing
   makes credit cascade-dominated. This explains the whole FAS pattern: long horizons trained (R8) but nothing could
   teach binding.

**Formal quantity.** The route sensitivity S = E[#later winner changes | one forced change] is a discrete analogue of a
Lyapunov exponent for the routing dynamics. First-order credit is valid only when S is small. S rises with the
memory horizon, and with small score margins at later races.

**Repair directions implied (prioritized):**
1. **Control sensitivity, not just credit.** Keep long memory but enlarge routing margins, so a remembered change
   rarely crosses a later decision boundary:
   - a margin regularizer on the top-two score gap at races;
   - a lower training temperature on a well-trained model;
   - or soft or expected reception (§418) in long-memory regions, where choices act smoothly.
   - Prediction: S falls while horizons stay long, and transported credit fidelity rises on long-memory models.
2. **Rollout credit where S is large.** Use exact paired forced-lane differences over a bounded horizon, the audit's
   own estimator, as the training signal for a few sampled races per window. This is unbiased for the sampled races by
   construction. *Cost:* (alternatives × horizon) extra lane-steps per sampled race, charged as fitting work.
3. **Transported write credit where S is small** (short-memory layers), now with measured support.

**Falsifiable next measurements:**
- S and estimator fidelity on the three-seed FAS models and on a language model.
- S as a function of horizon initialization (R0 against R8 is one point each).
- A margin-regularized R8 variant: does S fall without shortening horizons, and does transported fidelity rise?

## 432. Interleaving symmetry: why anonymous merged event logs are race-native (user direction, 6 October)

**Setting.** K processes (assembly items, sessions, machines) each emit a timed event sequence
s_i = ((e_{i,1}, t_{i,1}), (e_{i,2}, t_{i,2}), …). The log is their merge: the events of all s_i sorted by time.
It has no process ids and no correlation keys. Processes may interact through shared resources such as a station,
a queue or a lock.

**Definition (interleaving symmetry).** Let L = merge(s_1, …, s_K). Shuffling which hidden process each event came from
is not a symmetry, because each process keeps its own order and timing. The ambiguity is the shuffle product: the
same L arises from every assignment of its events to K order-preserving subsequences that the generator could have
produced. A detector should depend on L only through the likelihood marginalised over those consistent assignments:
  p(L) = Σ_{assignments a consistent with L} Π_i p(s_i(a)) · p(interactions | a).
- Interleaving is a special case of permutation symmetry, but it is not the same thing:
  - a permutation-symmetric model (a set function, attention without order) discards the order within each stream,
    which carries the signal;
  - an order-sensitive sequence model ties its prediction to one particular merge and must learn the binding from
    scratch.
- No standard classical primitive is invariant over shuffles while preserving within-stream order.
  - Attention is permutation-equivariant.
  - Recurrent nets, Transformers with position codes and SSMs read a single order.

**Proposition 432.1 (superposition is a race).** Suppose each process i has, at time t, a pending next event with hazard
λ_i(· | its own history, shared state). The merged log is the superposition of the K point processes. Its next event is
the earliest pending event:
  argmin_i τ_i, with τ_i ~ λ_i,
and the event that fires is the winner's. This is exactly a temporal race among K competitors whose clocks are
conditioned on persistent per-competitor state. Interactions couple the clocks through shared state.

Sketch: superposition of independent point processes is a standard result. The first event time is the minimum of
the competitors' next-event times, and the identity of the minimiser is the race winner. Dependence through shared
state makes the hazards conditional, and the race is still the generative form.

**Consequence.**
- A model with addressed persistent state (slots) and a race between them can represent the generative process
  directly:
  - a slot stores a process's own history (binding);
  - the race decides which slot's continuation explains the next event;
  - the delay law carries timing.
- Learning to bind is learning which slot wins. The route credit questions of §§429–431 are therefore the central
  learning problem on this data, not a side issue.
- Depth adds the interactions: deeper races condition on other slots' states.

**Fair references.**
- Knowing the processes (the route, its repeated and optional steps, tie behaviour) is privileged knowledge.
- A de-interleaving tracker whose structure was designed by inspecting true identities is a structure-assisted
  diagnostic, even when its parameters are fitted on anonymous logs.
  - It is not an information-matched reference.
  - experiments/fas/beam_deinterleave.py is such a diagnostic. Its initialisation and link rule were designed after
    inspecting identity routes on 6 October.
- Generic sequence and point-process models (LSTM/RMTPP, THP-style Transformer, LRU, S5, Mamba) and generic classical
  detectors are the fair references. They see the same anonymous log.

**Predictions to test** (FAS v2, Stages 3–5):
- Native advantage over generic references grows with K (more concurrent streams) and with event dropout, which
  breaks rigid structure.
- Native performance degrades under the within-sample timestamp shuffle control: timing carries the binding.
- The native model's per-slot write sources concentrate on single processes. This is measured by the binding
  diagnostic with oracle identities, used for evaluation only.

## 433. Race readout and posterior-routed writes: the superposition likelihood as the model's head (6 October)

**Failure addressed** (measured on FAS v1):
- The native model reaches validation/test AUROC ≈ .59–.60 at N=256, against the identity oracle's .755.
- The credit audit (§429) shows the implemented route credit nearly blind (corr −.08/.29).
- Long-memory routing is chaotic (§431: a median of 125 later winner flips per forced choice).
- The oracle's power comes from item-own durations. The current head predicts the next *merged* event from one pooled
  vector, so the model must learn binding indirectly, through hard write routes whose credit is the measured blind
  spot.

**Theory** (§432.1). The merged log is a superposition of per-process renewal processes, so the exact next-event law is
a race over the processes' pending events.

The readout makes each top-layer slot s a competitor:
- a type law p_s;
- a log-normal own-duration law f_s measured from t_s, the slot's last write;
- a pending probability q_s.

The model's likelihood is then the exact superposition density:
  log p(e, t') = Σ_v [log S_v(t'−t_v) − log S_v(t−t_v)] + logsumexp_s [log q_s f_s(t'−t_s) p_s(e) − log S_s(t'−t_s)].
- It integrates to the probability that any event occurs (contract: tests/test_race_readout.py).
- Survival of the non-firing slots is silence-aware supervision. A slot that predicted an event that did not come
  pays for it.
- If slots hold processes, the likelihood is literally the oracle's model with learned per-process laws.

**Posterior-routed writes.** The responsibility r_s ∝ q_s f_s p_s(e) / S_s is the posterior probability that slot s's
race produced the event. Writing the event to a slot drawn by an exponential race over log r (per head) is posterior
sampling: the filtering rule "write where predicted".
- It is a hard sparse write, chosen by a race.
- Its scores come from the model's own predictive likelihood, not from a separate query/key race that needs learned
  credit. Binding therefore needs no learned write credit: it is self-consistent, like an E-step.
- The likelihood gradient is exact for the readout's soft mixture over slots, with no route chaos in the likelihood path.

**Mechanisms.**
- Retained:
  - the deep race network below the top layer, with its learned query/key races, delays, transport and carried state;
  - sparse addressed writes, one slot per head;
  - small messages mixing content with memory;
  - key/value separation in the lower layers;
  - timing as computation.
- Added:
  - per-slot race readout (a temporal race at the output);
  - silence-aware survival supervision.
- Replaced (R2 only): the top layer's query/key write scores, by posterior responsibilities.
- Removed: nothing else.

**Work.**
- Inference adds per event H·U·(P·h + h·(V+3)) + H·P·h multiply-adds (p32, h64, H2, U8: ~114K), plus survival terms
  (special functions) for H·U slots. This is dense over the H·U top-layer slots: the readout scores every slot, as a
  race scores every competitor's clock.
- Learning adds the same, times the usual backward factor. No counterfactual lanes are needed.
- Sparse variants come next: score only seen slots plus one free slot.

**Required comparison** (FAS v1 validation, disclosed development, seeds 6):
- E1 (race control, standard head, pool 8, τmax 1000) against R1 (race readout, learned writes) and R2 (race readout,
  posterior writes);
- same data, same segment, epoch and learning rate;
- validation AUROC at N=128/256/512 (rule total), validation NLL in log-gap units, and top-layer slot use.
- Adopt R2 for the v2 development budget if it beats E1 by more than the seed spread (.009 at N=256 on v1) at
  N ≤ 512.

## 434. Binding as inference: the race readout is a particle filter over interleavings (6 October)

**Setting.** As in §432, the anonymous log L = (x_1, …, x_T), x_k = (type, time), is a merge of hidden processes.
The binding is a latent assignment path a_{1:T}: a_k is the slot (process) that emitted x_k.

The exact interleaving-marginal likelihood is
  p(L) = Σ_{a_{1:T}} Π_k p(x_k, a_k | x_{<k}, a_{<k}).
It sums over up to U^T paths. Exact marginalisation is a data-association problem, exponential in general.

The race readout (§433) supplies the per-step joint: log p(x_k, a_k = s | past) = base + fire_s. Posterior routing
(R2) draws a_k ∝ p(x_k, a_k | past).

**Theorem 434.1 (posterior-routed writes are single-particle SIS with the optimal proposal).** Let
Ẑ = Π_k p(x_k | x_{<k}, a_{<k}), with a_k drawn from p(a_k | x_{≤k}, a_{<k}) along one sampled path. Then
E[Ẑ] = p(L).

*Proof.* Proposal q_k(a_k) = p(a_k | x_k, past). The incremental importance weight is
p(x_k, a_k | past) / q_k(a_k) = p(x_k | past). This is independent of a_k. The product of the weights is the
standard SIS estimate of the normalising constant, which is unbiased for any proposal with adequate support
(Doucet & Johansen 2009; one particle, no resampling). ∎

Stochastic routes in the lower layers are further latent variables drawn from their prior. Their weights are 1, so
unbiasedness is unaffected.

*Numerical check* (toy: 3 slots, 4 types, 7 events, log-normal defective laws; exact sum over all 3^7 paths;
implementation-level `event_terms`; experiments/theory/smc_check_434.py):
- E[Ẑ]/p = 1.007 ± 0.005 over 20,000 paths;
- E log Ẑ is 0.144 nats below log p (the Jensen gap);
- the argmax path's log-likelihood is 1.07 nats below log p.

**Corollaries.**
1. R2 trains a **filtering variational objective**: E log Ẑ ≤ log p(L) (Jensen). This is FIVO/IWAE with one particle
   (Maddison et al. 2017; Burda et al. 2016). The detached routing matches FIVO's omission of the resampling
   score-function terms. The binding rule therefore optimises a bound on the likelihood marginalised over all
   interleavings, not a heuristic.
2. **More particles tighten the bound** and sharpen the anomaly score. Run L lanes per sample with independent route
   noise; resample by the incremental weights p(x_k | past) when the effective sample size falls; score
   −log((1/L) Σ_l Ẑ_l). The forced-lane machinery already runs lanes in parallel. Compute ↔ binding-accuracy is then an
   explicit dial: L × per-event work.
3. **Evaluate by sampling or particles, not argmax.** The argmax path is not an estimator: it is 1.07 nats below in the
   toy, against 0.14 for a sampled path. native_race_readout.py samples with a fixed seed.
4. **Exactness condition.** The theorem treats the emitting slot as one variable over the readout's H·U slots. R2
   draws one write per head from that head's conditional posterior, so with H > 1 each head is its own redundant
   tracker and the weight identity holds only approximately. An exact variant uses one emission head (H = 1 at the
   top layer), or a per-head product-of-experts emission. Record which variant runs.

**Proposition 434.2 (binding accuracy and detection power).** Take a detector that commits to an assignment and scores
the mean robust z of item-own durations, as the oracle and FIFO/timed trackers do. Let a fraction α of pairs be
correctly bound. A correct pair carries the fault shift δ (unit variance). A mis-bound pair is a duration from the
wrong item, with zero mean shift and variance ρ ≫ 1 (heavy tails, clipped at 50). Over n pairs:
  d'_eff = δ √n · α / sqrt(α + (1 − α) ρ) = d'_oracle · α / sqrt(α + (1 − α) ρ),   AUROC = Φ(d'/√2).
- *Calibration from the v2 probe* (K=2, p=.02, N=512): the oracle is .673 (d' = .634); the timing-aware tracker
  binds α = .86 of pairs and scores .568 (d' = .242), which implies ρ ≈ 30.
- *Consequence:* 14% mis-binding removes 62% of d'. Retaining 87% of d' needs α ≈ .99.
- *Steepness:* committed-assignment detectors are steep functions of binding accuracy, so de-interleave-then-detect
  pipelines collapse under modest dropout or speed offsets, as measured.
- *Race readout:* it scores each event under the mixture over slots (log Σ_s), not under a committed slot. An event
  whose committed slot is wrong can still be explained by the right slot in the score. Errors act only through
  corrupted slot states (wrong t_s) at later events.
- *Hypothesis to measure:* the readout's effective ρ is far below 30. Test: compare its validation AUROC with the
  oracle as top-layer binding purity (the binding diagnostic, oracle identities used for evaluation only) varies.

**Proposition 434.3 (slot capacity by Little's law).** In steady state the mean number of items in progress is
L_items = λ W (Little's law: arrival rate λ, mean time in system W).
- For K lines, a binding layer needs U ≥ K λ W slots per head, plus a fluctuation margin.
- FAS: items start every ~60 s per line (λ ≈ 1/60 s⁻¹) and take W ≈ 820 s. So λW ≈ 13.7 per line; the measured
  maximum is 18 per line and 35 at K=2.
- *Prescription:* U ≈ 20 K per head (pool 40 at K=2, 60 at K=3).
- With U < KλW, items must share slots. Sharing is mis-binding by construction, and the loss follows 434.2.
- The current v1 development pool (8 per head, 16 slots) is below λW = 13.7 per head for a single line.
- *Prediction:* R1/R2 at pool 8 under-use the readout. A pool-16/24 arm on v1 should gain, saturating near
  U ≈ 18.

**Proposition 434.4 (prefix scaling isolates interleaving).**
- *Per line:* evidence about the faulty line grows with the number of that line's events. FAS faults are
  progressive, so d' grows faster than √n. The v1 oracle's d' rises ×1.49 from N=128 to 256 and ×1.97 from 256 to 512.
- *Merged prefix:* at merged prefix N, each line contributes ~N/K events. Detection at fixed merged N then falls with
  K for every detector, even with perfect binding.
- *Per-line prefix:* N_line · K holds the faulty-line evidence fixed, so differences between detectors measure
  binding and interleaving alone. This is the formal basis of the Stage 1 amendment.

**Intuition: why race models carry the right inductive bias.**
- *Race readout:* with U ≥ KλW slots, the R2 model class contains the generative family (K marked renewal processes,
  log-normal own durations, Markov types) as a special case. Its objective is then the one-particle bound of
  Theorem 434.1 on the true likelihood.
- *Sequence model:* it would have to learn, in one dense state, a filter over exponentially many assignments, with no
  structural prior for "one process per slot, own clocks, earliest fires".
- *Clocks bind:* the responsibilities are sharp because each slot measures its own elapsed time. A line-speed offset
  is absorbed into that slot's own law. Pooled detectors cannot absorb it (the .554 pooled oracle at δ = ±5%).

**Tests implied** (ordered by cost):
1. R2 against R1 against E1 (queued).
2. Particle evaluation of the R2 checkpoint, L ∈ {1, 4, 16} lanes with resampling. Evaluation only. Predicted:
   validation NLL falls and AUROC rises with L.
3. A pool sweep on v1, U ∈ {8, 16, 24}. Predicted: saturation near U ≈ 18.
4. A binding-purity diagnostic for R1/R2 against AUROC, testing the readout's effective ρ.
5. An H=1 top-layer R2 variant for exactness.

## 435. What anonymity costs, what the readout must factor, and calibrated alarms for free (6 October)

**Theorem 435.1 (interleaving information-loss identity).** Let A be the hidden binding and L the anonymous log, with
clean and faulty laws P_c and P_f. The chain rule of KL divergence gives exactly
  D(P_f^L ‖ P_c^L) = D(P_f^{L,A} ‖ P_c^{L,A}) − E_{P_f} [ D( P_f(A | L) ‖ P_c(A | L) ) ].
- The information in the anonymous log equals the identity oracle's information minus how much the fault changes the
  posterior over bindings.
- By Stein's lemma, these divergences are the error exponents at a fixed false-alarm rate. No anonymous detector can
  exceed the left side.
- The Neyman–Pearson-optimal anonymous detector is the marginal likelihood ratio p_f(L)/p_c(L), marginalised over
  bindings.
- The one-class score −log p_c(L) needs the same marginalisation. R2 with particles (§434) estimates exactly that.
- Consequence for the Stage 1 gate: oracle − best classical mixes two parts. Only D_anon − D_classical is reachable
  by any anonymous model; the binding-posterior term is irreducible. Estimate it with the SMC model when sizing claims.

**Proposition 435.2 (type–duration coupling is created by silence).** A slot's next event law must be p_s(e) f_s(τ | e).
The factorised readout of §433, p_s(e) f_s(τ), loses exactly E[I(E; T | state)] nats per event.

Measured on true item tracks (150 runs, K=1, 24 log-duration bins; shuffle floor ≤ .002; experiments/theory/mi_check_435.py):

| state given | p = 0 | p = .02 | p = .05 |
|---|---|---|---|
| current type | .405 | .443 | .486 |
| previous + current type (route position) | .019 | .105 | .191 |

The duration entropy given the route position is 0.67–0.81 nats.
- Once the slot knows its route position, coupling is near zero without dropout. It grows linearly with the dropout
  rate: a silently dropped step makes the next type skip ahead *and* the duration double.
- Prescription for v2 (dropout is part of every setting): **type-conditional own durations** f_s(τ | e), with
  survival S_s(τ) = 1 − q_s Σ_e p_s(e) F_s(τ | e). This is silence-aware modelling of missing events inside each
  slot's race.
- The cost is V log-normal CDFs per slot per event. The predicted gain is ≈ .1–.2 nats per event at p = .02–.05,
  concentrated in the duration term, which carries the fault signal.

**Proposition 435.3 (time rescaling gives calibrated alarms).** The readout is a marked point process with total
cumulative hazard Λ. Between events, ΔΛ_k = −base_k = Σ_v [log S_v(t − t_v) − log S_v(t' − t_v)], in closed form
(§433). By the time-rescaling theorem (Brown et al. 2002), under the clean model the ΔΛ_k are i.i.d. Exp(1). The
mark residuals (randomised probability integral transform of the type) are i.i.d. U(0, 1).
- Anomaly statistics therefore have a known null with no validation calibration. Faults add delay, so Page's
  one-sided CUSUM on ΔΛ_k − 1 gives alarms whose false-alarm rate is set analytically.
- Model check, development only: the Kolmogorov–Smirnov statistic of 1 − exp(−ΔΛ) on clean validation measures
  misfit.
- Declaring this as a v2 rule requires a protocol amendment before Stage 3 (rules are fixed per protocol).

**Tests implied:**
- R4: R2 with type-conditional durations, on v2 data (dropout present). Predicted ≈ .1–.2 nats/event lower NLL than
  R2 at p = .02–.05, and higher AUROC.
- KS uniformity of rescaled intervals on R2's clean validation, as a fit diagnostic.
- An SMC estimate of the binding-posterior term of 435.1 on v2, to bound the reachable native gain.

## 436. Binding memory: capacity separated from computation (6 October)

**Failure addressed.** Binding needs U ≥ KλW slots (§434.3): ~36 at the selected v2 setting (K = 2). In the deep race
layers each slot computes a proposal, O(P²) per slot per event, so raising every layer's pool to 40 multiplies the
whole network's work by ~5. The top layer also has H = 2 heads, so Theorem 434.1 holds only approximately (§434.1.4).

**Change.**
- A BindingMemory of U_b slots sits above the deep race network, which keeps pool 8.
- Each event is written to one slot s*, drawn by the exponential race over the readout's log responsibilities
  (posterior sampling; argmax when deterministic): m_s* ← exp(−(t − t_s*) r) ⊙ m_s* + W x. x is the deep network's
  output; the rates r are per dimension, initialised to time constants 1–1000 s.
- The race readout reads the U_b binding slots with one emitting head.

**Retained:**
- deep temporal races with learned query/key routes and linear credit;
- transport, delays and carried state;
- sparse addressed writes (one binding slot per event);
- the superposition race readout with silence-aware survival.

**Replaced:** the top layer's role as binder. The deep layers now compute event representations, and the binding
memory holds processes.

**Work per event.**
- Deep network: unchanged.
- One binding write: P × H·P multiply-adds.
- Readout: U_b · (P·h + h·(3V + 1)) multiply-adds plus U_b · V log-normal CDFs (type-conditional durations).
- Binding capacity is linear in U_b, with no per-slot proposal computation: capacity beyond activity.

**Exactness.** One emitting head, so R2-style training with binding memory is exactly the single-particle filtering
objective of Theorem 434.1. race_smc_eval.py gives the L-particle estimate.

**Comparison.** On v2 validation (K=2, p=.02), the native development configurations (budget 8) start from
R5 = binding memory U_b = 40 + type-conditional durations (§435.2), against R2-style top-layer binding at pool 8.
Contracts: tests/test_race_readout.py (12 pass: chaining, posterior writes, gradients into the binding path).

**436.1 Step-class mixture (cost repair of §435.2).**
- Type-conditional durations need V duration laws per slot: 40 slots × 46 log-normal CDFs per event, 491 events/s at
  16 lanes.
- The coupling §435.2 measures comes from skipped steps (one step against two), so a mixture of M step classes
  carries it: p_s(e, τ) = Σ_c π_c p_c(e) f_c(τ), with survival 1 − q Σ_c π_c F_c(τ).
- At M = 3 the cost is 120 CDFs per event. With a compiled binding write, throughput is 818 events/s at 16 lanes
  (U_b = 40, p32 d4 pool 8); R2 at pool 8 does 1,242.
- Contracts: density normalisation, binding episode, compiled = eager (14 pass).

**436.2–436.3 (small integrated fit, `experiments/fas/binding_toy.py`; 3 interleaved 8-step processes; Bayes-greedy
binding ceiling .911).**

| readout variant (600–1,200 steps, gated writes) | argmax α | validation NLL |
|---|---|---|
| hidden 32, additive writes (300 steps) | .71 | 1.97 |
| hidden 32, gated writes | .80–.84 | 1.77 |
| hidden 32, no merged context | .908 | 1.83 |
| hidden 64, with context | **.934** | **1.73** |

- *Gated writes* (§436.2) let a slot hold its process's latest state.
- *The context shortcut* (§436.3): with merged-stream context, slots can predict the merged next-event law without
  binding. Removing the context forces binding at fixed readout capacity; more capacity reaches binding with the
  context kept.
- Learned binding can exceed the greedy-oracle ceiling: the readout uses information the greedy assignment ignores.

**436.4 Binding needs spare capacity (optionality, §§422–424, measured on binding).** Toy with 6 concurrent processes
(Bayes-greedy ceiling α .886, 600 steps):

| slots | argmax α | validation NLL |
|---|---|---|
| 3 | .31 | 2.19 |
| 6 | .48 | 2.11 |
| 9 | **.90** | **1.82** |

- *Mechanism:* posterior writes made early in training, before the laws are informative, are partly wrong. With free
  slots, a wrongly placed process can move to an unused slot and specialise there. With U equal to the concurrency,
  every error forces sharing, and shared slots learn mixed laws that keep their responsibilities flat: a near-symmetric
  trap like the beam tracker's EM collapse.
- *Free capacity is optionality:* the exposure value of unused routes (§423) appears here as the escape route from
  mis-binding.
- *Sizing rule:* U ≈ 1.5 × peak concurrency, where peak concurrency is estimated anonymously (`anonymous_concurrency.py`).
  The cost of spare slots is linear: readout scoring only, with no deep-layer work.

**434.1.2 measured (binding toy, 6 processes, 9 slots).** Validation NLL with L-particle SMC evaluation of one trained
model:
- L = 1: 1.885;
- L = 4: 1.834;
- L = 16: 1.815;
- L = 64: 1.810;
- argmax path: 1.821.

The bound tightens monotonically with diminishing returns. The argmax path beats a single sampled particle on this
trained model, but not 16 or more.

## 437. Races everywhere: from one operation to the whole model and its training (user question, 6 October)

**Diagnosis.** Against Transformers we replaced one operation, attention, by a race. A race is simultaneously:
- a selection;
- a posterior sampler (the exponential race draws from the softmax);
- a clock (the winner's time carries computation);
- a likelihood with silence (the losers' survival terms).

Every other place where a conventional stack makes a choice or spends compute is still dense. This applies at
inference (MLP experts, output layer, depth, which inputs to process) and in training (credit through hard routes,
dense backward over all alternatives).

**Programme.**

*Inference:*
- I1 race-selected expert banks in place of MLP blocks (capacity beyond activity);
- I2 competing-risks race readouts, done for FAS (§433); tournament races for large output sets;
- I3 halting races for adaptive depth (delay encodes confidence; anytime answers);
- I4 surprise-gated processing (predictable events take a cached state update; full computation is spent on
  surprising events, so cost scales with information);
- I5 posterior-routed writes, done (§§434–436).

*Training:*
- T1 internal routes as latent events learned by filtering inference rather than backpropagation through hard choices.
  Theorem 434.1 generalises to any route whose consequence is an observed event: posterior routing plus the filtering
  bound. This is the direct remedy for the measured credit blindness (§429) and route chaos (§431);
- T2 training work proportional to activity (posterior or k-sampled counterfactual credit instead of all-alternative
  linear credit);
- T3 particles in training (measured in evaluation: NLL 1.885 → 1.810 from L = 1 to 64);
- T4 elastic capacity (recruit slots when the free pool runs low; §436.4: binding needs spare capacity);
- T5 event-proportional training cost on bursty data (closed-form survival); to be benchmarked explicitly.

**Order.**
1. T1 on a toy whose hidden route is observable only through outcomes: a two-layer race model trained with
   posterior-routed internal races against learned races with linear credit. Measure route recovery and NLL.
2. I4 for FAS inference after C1.
3. I1 for language (R1 owner).

Each change states the failure it addresses and runs its contracts before any long run (AGENTS.md).

**437.1 A taxonomy of internal routes, by when their consequence is observed.**

| class | example | learning rule without learned route credit |
|---|---|---|
| (a) consequence observed now | which process emitted this event (binding); which expert explains this output | posterior routing plus the filtering bound (Theorem 434.1); EM for mixtures of experts is the classical instance |
| (b) consequence later, but a content key exists at both write and read time | associative memory: write (key, value), later query by key | content addressing: the slot is a function of the key (hashing, product keys, race over key similarity), so read and write meet with no credit |
| (c) consequence later, with no shared key | which state to carry for a future, unspecified use | needs backward information: fixed-lag smoothing (posterior over the route given the next Δ events, computable by L-particle look-ahead) or sampled counterfactual credit (§429's forced lanes, k alternatives) |

*Implication.* Credit blindness (§429) and route chaos (§431) are class-(c) problems. Classes (a) and (b) can be
removed from the credit problem by construction. A model design should push routes into (a) and (b):
- predictive readouts make consequences immediate;
- keys make them addressable.

Only the remainder pays for counterfactual credit. The T1 toy therefore tests a class-(a) route inside the network
(a race-selected expert whose output is the next observation), and a class-(c) route with fixed-lag particle smoothing.

**437.2 T1 first result (binding toy, 3 processes).**
- *Posterior-routed writes:* α .93, NLL 1.73.
- *Learned query/key write race with linear write credit* (our deep layers' rule): α .10–.22, below chance, NLL 2.25
  and then diverging.

The write route's consequences appear only in later predictions. The first-order credit gives the race scores almost
no usable signal, while choosing the route by the readout's posterior makes it an inference problem with an exact
filtering objective.

Next T1 test: replace the learned races of the deep layers by posterior-style routing where a predictive consequence
exists (class (a) of §437.1). For the deep layers, that means a per-slot predictive head scoring the slot's proposal
against the next observed event.

## 438. Predictive routing: every layer as a superposition race model of its input stream (proposal, 6 October)

**Failure addressed.**
- The deep layers' routes are learned by first-order credit. That credit is nearly blind (§429: corr −0.08/0.29),
  chaotic under long memory (§431), and fails outright on binding (§437.2: α ≤ chance where posterior routing
  reaches .93).

**Theory.** Layer d receives a stream of event representations x^d_k with times t_k. Treat it like the observed log of
§432:
- each slot u of the layer keeps a predictive law g_u(x, τ) for the next input it expects (type-like features, own
  duration since its last write);
- an arriving input is routed by the posterior race over r_u ∝ q_u g_u(x^d_k, τ_u) / S_u(τ_u);
- the layer's local objective is the superposition log-likelihood of its input stream (Theorem 434.1: an exact
  filtering bound for that layer's own latent routing).

The route is class (a) of §437.1: its consequence, how well the chosen slot predicted the input, is observed when the
route is taken. No learned route credit is needed.

The layer's output is unchanged: the winning slot's proposal (memory mixed with the input), delivered with the race
delay.

**What it means.**
- Each layer partitions its input events into predictable streams at its own level of abstraction: processes at the
  binding layer, sub-processes or event roles below.
- Training gets layer-local objectives (one predictive likelihood per layer) beside the end-to-end task loss. This
  is predictive-coding-style local learning, with a proof that each local objective bounds that layer's marginal
  likelihood.

**Mechanisms.**
- *Retained:* temporal races (now posterior-scored), delays as computation, sparse addressed writes (one slot per head
  per event), persistent state, small messages mixing input with memory, and the race readout.
- *Replaced:* the query/key race scores of routed layers, by predictive responsibilities. Keys survive as part of each
  slot's predictive law (its static identity).
- *Removed:* the linear route credit for those layers. Their routes no longer need it.

**Work.**
- *Inference:* per layer, U predictive-law evaluations per event (with cached per-slot laws as in §436.3: U duration
  CDFs plus a feature-likelihood term), replacing U query/key scores of similar size. Winner-only proposals as in
  sparse inference.
- *Training:* local likelihood terms add little. The linear-credit backward over all U proposals disappears, so
  training work moves toward activity-proportional (T2).

**Required comparison before any long run.**
- *Contracts:* the per-layer density integrates to the event probability; segment chaining is exact; posterior writes
  go to the most responsible slot.
- *Integrated fits:*
  - a toy where two levels of grouping exist (processes made of sub-processes);
  - then FAS v1 validation, against C2-style learned routing at equal size.
- *Measures:* binding purity per layer, validation NLL and AUROC, and training events/s.

**Risk.** The feature-likelihood g_u(x) on continuous representations can collapse (every slot predicts the mean). Two
guards:
- predict only low-dimensional projections of x;
- keep the slot keys as fixed random anchors for diversity.

The capacity margin of §436.4 applies per layer.

**438.1 First test: negative (binding toy, 6 processes, 9 binding slots).** A predictive-routed deep layer (9 slots, own
race readout, local likelihood) in place of the learned deep layer:

| deep layer | α (sampled / argmax) | validation NLL |
|---|---|---|
| predictive layer | .67 / .59 | 2.18 |
| learned deep layer | .82 / .90 | 1.82 |
| no deep layer | .77 / .74 | 2.25 |

Revised reading:
- At this level the learned deep layer supplies *cross-process history context*, not binding. A layer whose slots
  are partitioned by predictability duplicates the binding layer and provides no shared context.
- Predictive routing therefore belongs where a layer's job is partitioning (binding, experts), not where its job is
  summarising across partitions.
- The next form to test keeps a shared context path beside the predictive slots, with local and global objectives
  weighted. Until then, class-(a) routing is adopted only at the binding layer.

**437.3 I4 first test: negative at inference-only gating** (toy, 6 processes). Skipping the deep layers for
well-predicted events costs NLL roughly in proportion to the skipped share (+0.06 at 9%, +0.20 at 20%). The deep
layers' cross-process context (§438.1) needs every event. Remaining forms: training with the gate (the model learns
to tolerate gaps), or gating that still applies a cheap state update (a decay-only step) instead of skipping.

## 439. The resolution principle for race readouts: score recorded cells, not points (7 October)

**Failure.** FAS records time in milliseconds. On v2 validation:
- 36% of merged gaps are exactly 0 ms;
- 29% of item-own durations are exactly 0 ms (same-item steps sharing a timestamp).

A continuous own-duration density with log(τ + ε) and ε = 1 ms gives a tied event a density up to ~1/(σ ε). That is
+10 nats per tie, which the likelihood rewards without any information about faults. Round 1 shows the symptom:
- C1 has a far lower validation NLL than C2 (0.338 against 2.655);
- yet C1 has worse detection (.630 against .663);
- and its timing rule is near chance (.550).

The B1 battle found the same pathology ("zero-gap spike"; grid audit).

**Principle.** Data recorded at resolution δ have the likelihood of the recorded cell. Slot s firing in
[τ, τ + δ), with no other slot before it, has probability ≈ [S_s(τ) − S_s(τ + δ)] Π_{v≠s} S_v(τ_v)/S_v(e_v):
- ties get a bounded probability;
- per-second units are kept by subtracting log δ;
- as δ → 0 the density is recovered (contract);
- the discrete probabilities sum to one up to same-cell coincidences of two slots (contract, 5e-3);
- log(Φ(z₂) − Φ(z₁)) is computed in the stable tail (`log_diff_ndtr`).

The likelihood then spends no capacity on resolving below the recording cell. The duration laws must explain
non-zero durations, which is where faults act.

**Test.** C6 = C1 + δ = 1 ms (`--cell-ms 1`). Prediction: validation NLL rises (no tie spike), the timing (gap)
AUROC rises above C1's .550, and the total AUROC rises toward or above the classical .685.

## 440. Silence-aware score tests: the locally most powerful one-class detector for slowdowns (7 October)

**Problem.** Training is one-class: there is a clean model p₀ and no fault model. The B3 primary score, mean NLL over
the prefix, tests typicality in every direction at once.

On v2 the measurements show its cost:
- the type rule beats the total rule, so the timing NLL adds noise;
- the oracle's strongest statistic is *signed* and *per step* (max over transitions of mean robust z).

The faults have a known *shape* without known parameters: a component slows down, so durations of the affected steps
lengthen.

**Alternative family.** For a process (slot) s and an affected step class c, the slowed model scales own durations by
a = e^ε: f_{s,c}^ε(τ) = f_{s,c}(τ e^{−ε}) e^{−ε}. The score test against ε > 0 is locally most powerful among
level-α tests for small ε (Neyman–Pearson in the local limit; Rao). Its statistic is
T = ∂_ε log p_ε(L) |_{ε=0}.

**Proposition 440.1 (score of the race readout).** For log-normal own-duration laws in u = log(τ + ε₀) with location
μ and scale σ:
- the firing term contributes ∂_ε log f^ε = (u − μ)/σ² per firing event;
- the survival of any slot contributes ∂_ε log S^ε(τ) = h(τ) τ̃ with h = f/S and τ̃ = τ + ε₀, i.e.
  q Σ_c π_c φ(z_c)/σ_c / S.

Summing over the merged likelihood of §433 with posterior responsibilities r_s (and class posteriors w_{s,c}) gives,
per event k:

T_k = Σ_s r_s Σ_c w_{s,c} (u_s − μ_{s,c})/σ_{s,c}²   (firing lateness)
    + Σ_v [ℓ_v(τ'_v) − ℓ_v(τ_now,v)],   with ℓ_v(τ) = ∂_ε log S_v^ε(τ)   (silence lateness).

- The first term is the *signed* standardised lateness of the event that arrived, attributed to its slot.
- The second grows while expected events fail to arrive. A stalled step is detected through silence, before any late
  event appears.

**Corollary 440.2 (the oracle's statistic is a special case).**
- With true identities (r = indicator), Gaussian log-durations and the silence term dropped, Σ_k T_k restricted to one
  transition type is the oracle's mean signed z for that type, up to the 1/σ weighting.
- The oracle's max over transitions is the union-intersection test over the unknown affected step.
- The race readout supplies r, the per-step laws and the silence term from anonymous data. It is the model-based
  analogue of the oracle's detector.

**Detector family (declared before use).**
- **late**: mean_k T_k over the prefix.
- **late_max**: max over event types e of the mean of firing lateness on events of type e (minimum 5 events), plus
  the silence term.

Two-sided NLL remains the primary B3 rule unless an amendment admits a native rule selected on validation; classical
detectors already select from nine statistics.

**Predictions** (toy with synthetic slowdowns, then FAS):
1. late and late_max exceed mean NLL in AUROC for duration-scaling faults at equal prefix.
2. The silence term adds power for stalls (faults that withhold events).
3. On FAS, the late family moves native timing above the type rule and toward the oracle's per-fault profile
   (largest gain on retry delay, §439 analysis).

**440.3 GLR over the slowdown magnitude, per step (toy results).**
- The score test is locally optimal but loses to mean NLL on gross faults: at 1.5×, late_max .818 vs NLL .871 with 3
  processes, and .710 vs .850 with 6.
- The generalised likelihood ratio maximises the log-likelihood gain over the slowdown magnitude. *Per step*, it is the
  union-intersection test over the unknown affected step:
  G = max_e max_s Σ_{k∈e} [ℓ_k(s) − ℓ_k(0)] / n_e.

AUROC, 600-step trained models, synthetic single-step slowdowns:

| setting | mean NLL | late_max | global GLR | **per-step GLR** |
|---|---|---|---|---|
| 3 processes, 1.2× | .636 | .691 | .575 | **.731** |
| 3 processes, 1.5× | .871 | .818 | .663 | **.911** |
| 6 processes, 1.2× | .644 | .656 | .576 | **.723** |
| 6 processes, 1.5× | **.850** | .710 | .693 | .832 |

- Locality matters: a global slowdown dilutes a one-step fault.
- The per-step GLR keeps the score test's gain on subtle faults (+.08 to +.10 over NLL) and stays at or near NLL on
  gross ones.
- It needs no identities and no fault labels: only the readout's per-slot laws, the posterior responsibilities and the
  silence terms.
- Declared as the B3 native primary rule before any FAS evaluation (protocol amendment, 7 October).
