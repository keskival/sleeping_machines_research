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
