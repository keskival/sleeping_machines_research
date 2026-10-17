# 175 — Credit, counterfactuals and critics in sparse asynchronous compositions

17 October 2026, Curie. Founder: "We do need true advance and progress in the analytical formal understanding of how our
components compose in a trainable way, including learning sparse routes, assigning credit and counterfactuals and critics";
"Our training steps should form a mathematical iteration which at its limit approaches full route tree, full depth true values
which they estimate. Have we proven that at least?"

Builds on 160 (forward/backward duality), 164 (sparse forward-trained delayed credit), REVEALED_FUTURE_CREDIT, 173 (C1–C9) and
174 (asynchronous modules). Witnesses: `experiments/credit/sparse_async_credit_theory.py` (T1–T4, float64).

**Short answer to the founder's question: it had not been proven. §8 now proves it for our construction: (A) with complete races, soft decisions and an unbounded credit horizon, our training step is exact gradient descent on the full route-tree, full-depth model; (B) the model we actually train deviates from it, in loss and gradient, by at most a constant times the total excluded race mass plus a geometric horizon term, and every term is measured during training as a route-tree certificate.**
§4 states what the current iteration provably converges to: first-order route-optimal stationary points of the hard-routed,
credit-truncated objective. It also gives the conditions under which an iteration converges to stationary points of the full
route-tree, full-horizon objective. Global optimality over the route tree is out of reach for any tractable method: best-subset
selection is NP-hard (Natarajan 1995).

## 0. Objects

A route D collects every hard decision: concept units woken, slots written and read, firing, event pointers, clock choices.
Parameters are θ = (φ, σ): φ continuous (maps, values, rates), σ the scores that rank candidates.
- The **hard-routed objective** L(φ, D(σ)) is what we evaluate: argmax routes, identical in training and evaluation.
- The **full route-tree objective** J(θ) = E_{D ~ π_θ}[L(φ, D)] averages over every route under the race distribution π_θ.
  With all candidates in every race and an unbounded credit horizon, J is "full route tree, full depth".

Credit for a decision comes from one of three sources, which trade cost against bias and variance:
- **exact:** a pathwise derivative or an exact counterfactual field;
- **sampled:** probes or score-function samples, unbiased with variance;
- **predicted:** a critic, biased, with low variance and no lag.

## 1. T1 — Counterfactual credit for an addressed memory is a duality

**Theorem.** Let a decaying addressed memory hold M_s(t) = Σ_{τ≤t} e^{−b_s(t−τ)} w_{τs} δ_τ and be read as r_t = Σ_s a_{ts} M_s(t),
and let g_t = ∂L/∂r_t for any differentiable loss. Then for every slot s, written or not, and every read weight, read or not:

  ∂L/∂w_{τs} = ⟨δ_τ, Λ_s(τ)⟩,  where Λ_s(τ) = a_{τs} g_τ + e^{−b_s} Λ_s(τ+1);
  ∂L/∂a_{ts} = ⟨g_t, M_s(t)⟩.

*Proof.* M is linear in w, with ∂M_s(t)/∂w_{τs} = e^{−b_s(t−τ)} δ_τ for t ≥ τ. So
∂L/∂w_{τs} = Σ_{t≥τ} a_{ts} e^{−b_s(t−τ)} ⟨g_t, δ_τ⟩, and the sum obeys the stated reverse recursion. The read derivative is
immediate. ∎

Witness: both agree with autograd on dense weights to 2e-15 and 4e-15.

Consequences:
- **Reads need the forward state; writes need the adjoint state.** The adjoint Λ is the memory run backward in time with the
  same decays, the reciprocal adjoint of theory 160 §1 applied to addressed slots. Forward and backward are the same machine
  in opposite directions.
- **Exact credit for every alternative address costs only the reads.** Λ_s(τ) is zero unless slot s is read at some t ≥ τ
  (witness: exactly 0 outside future reads). Complete counterfactual write credit therefore costs O(reads × d), with no
  sampling.
- **The backward pass is itself a race, in the dual space.** The best alternative write address for step τ is
  argmax_s ⟨δ_τ, Λ_s(τ)⟩: a race over adjoint states with query δ_τ. The best alternative read is argmax_s ⟨g_t, M_s(t)⟩: a race
  over forward states with query g_t. The structures that make forward races sublinear (product keys, region indices)
  apply to credit discovery unchanged.
- **Most write credit lives on addresses not written.** In the witness, 92% of the total write-credit mass sits on unwritten
  addresses.
  - Random probes over E of the n_out excluded addresses, weighted n_out/E (Horvitz–Thompson), are unbiased for that mass
    (Monte Carlo bias 2.5%). Their relative standard deviation is 2.1 per entry at E = 4.
  - Our unweighted probes carry E/n_out of it (0.18 here, far smaller at our lane lengths).
  - The dual race replaces this sampling with the exact field restricted to slots that are read later.

## 2. T2 — Collapse of sparse banks, and the cost of recruiting dormant capacity

**Proposition.** In a bank with hard top-k selection, a unit's value parameters enter the forward pass only through its
inclusion weight. A zero-forward probe sets that weight to p − sg(p), whose value is 0, so the probe's gradient with respect
to the unit's values is exactly 0. Probe credit trains the unit's score (would including it, as it is now, help?), never
its value. A unit that is never selected and only probed keeps its initial value forever. If dormant units start
identical, their probe credits are identical too, so nothing breaks the symmetry. The dormant set is absorbing.

*Witness.* Probed-only units: value norm exactly 0 after 400 steps. With paid exploration, where excluded units join the
forward race, the same units train (value norm 0.11), and all 64 units acquire values against 61.

Consequences:
- **Recruitment needs paid exploration.** Usable capacity grows only through exploration that participates in the forward
  pass (CERL's exploration units) or through forced usage (balance, homeostatic prices). This is the formal reason for the
  124-of-1,024 collapse.
- **Operating limit of capacity beyond activity.** A unit is explored on about E·Q/U of steps (E explorations per query row,
  Q rows per step, U units). If it needs n* value updates to become competitive, recruiting it takes about n*·U/(E·Q) steps.
  - Native member: U = 65,536, E = 2, Q = 1,024 gives 32·n* steps per unit, affordable in a 16,384-step run for n* up to
    about 500.
  - At U = 2^20 it becomes 512·n* steps, and dormant capacity stays untrained unless E·Q grows with U.

  Dormant capacity is free in inference work but costs recruitment time linear in U.

## 3. T3 — Future credit is a temporal-difference problem on the memory adjoint

**Theorem.** Let λ_n be the cotangent on the memory carried into lane n (length L), and ℓ_n the credit of lane n's own reads
to that state. Because the transport between lane boundaries is exactly diagonal for decaying slots:

  λ_n = ℓ_n + Γ λ_{n+1},  where Γ = diag(e^{−b_s L}).

*Witness:* agrees with backprop through a five-lane chain to 1e-16. Per-slot discounts span 0.04–0.48 per 8-token lane.

Consequences:
- **Forward trains backward, and vice versa, concretely.** A critic C_ψ(M) ≈ λ is trained by n-step temporal differences:
  targets are the exact cotangents of the next H lanes from CarryCredit (exact backward), bootstrapped by Γ C_ψ beyond them.
  Its predictions then credit writes immediately and indefinitely (forward). This is synthetic-gradient training (decoupled
  interfaces, Jaderberg et al. 2017) with an exact, known transport.
- **Each slot's discount is its own clock.** The credit horizon of a slot equals its memory horizon. Linear TD with a
  discount below 1 converges under on-policy sampling (Tsitsiklis & Van Roy 1997), and the target variance grows like
  1/(1 − γ_s²).
- **Operating limit:** slots and clocks with very long horizons (γ_s near 1) are exactly the ones whose credit is slow and
  noisy. Learnable lifetimes (theory 174 P8) are paid for in credit variance.
- **Truncation bias.** Without a critic, truncating credit at H lanes leaves a bias bounded by
  Σ_{n>H} ‖Γ^n ℓ‖ ≤ C max_s γ_s^{H+1}/(1 − max_s γ_s).

## 4. T4 — What the training iteration converges to

**Proposition (route fixed points, corrected 17 Oct after the witness).** Take an iteration with:
- exact pathwise credit inside the route;
- probes with full support outside, using the direction of a renormalised race (v_j − r̄, as the RMR and slot-memory read
  probes do: `r − probe·r.detach()`);
- straight-through firing;
- horizon H.

At optimal mixing weights every selected candidate has the same marginal credit (the KKT conditions on the simplex), so an
excluded candidate j moves only through its inclusion credit c_j = ⟨∂L/∂r, v_j − r̄⟩. Hence:
- **(i)** a route is a fixed point of a fixed-k race iff c_j ≥ 0 for every excluded j, that is, iff the route is first-order
  optimal over all supports: the best unrestricted solution near this route uses no more than k candidates;
- **(ii)** otherwise some excluded candidate's score keeps rising until it displaces a selected one, and the route churns with
  no fixed point;
- **(iii)** the continuous parameters are stationary within any settled route;
- **(iv)** sign conditions, and so the fixed points themselves, do not depend on positive rescaling of the probes. The
  rescaling changes speed and churn.

*Witness (T4, `curie_sparse_async_credit_witnesses_20261017_v1`):* top-3 routes over 10 candidates, 20 problems.
- 9 of 20 converged routes are first-order optimal over all supports, and they settle: 1.0–1.3 distinct routes in the last 300
  updates.
- The other 11 churn: 7.0 distinct routes unweighted, 10.8 with Horvitz–Thompson weighting.
- 14 of 20 (unweighted) end on the globally best size-3 subset. The globally best fixed-size subset need not be a fixed point,
  because a larger support would do better.
- An earlier version of the witness used probes without the renormalisation term; its routes were not first-order optimal
  (7 of 20). Probe direction matters.

**Consequence: threshold races.** A race that admits every candidate whose score exceeds a learned price λ, rather than a
fixed top-k, optimises L + λ·|S|. Its fixed points exist and are exactly the first-order (KKT) conditions of that penalised
problem. The price is the dual variable of an activity budget: theory principle P5, "thresholds are prices", here as the
condition for convergent routing. Fixed-k races are a constrained special case that churns whenever the constraint binds;
churn adds exploration and noise but no convergence. Route certificates (non-winning mass) and route-change counts measure
it in our runs.

What the iteration does not reach: the global optimum over routes (combinatorial), exact flip optimality where the loss is
nonlinear in a decision (straight-through linearises), and credit beyond H lanes.

**Theorem (consistency for the full route-tree objective).** Suppose:
- (a) decisions are sampled from π_θ mixed with an exploration floor ε (full support);
- (b) decision credit is unbiased: score-function credit with a control-variate critic (REVEALED_FUTURE_CREDIT), or
  HT-weighted exact counterfactuals (T1);
- (c) the horizon bias is at most β_H, from truncation (T3 bound) or a TD critic with error ε_c;
- (d) staleness is bounded by τ;
- (e) step sizes satisfy η ≤ 1/(L_J(τ + 1)) with the Robbins–Monro conditions.

Then the standard analysis of biased stochastic gradient descent on a nonconvex L_J-smooth objective gives

  min_{k≤K} E‖∇J_ε(θ_k)‖² ≤ O(1/√K) + O(β_H²),

and as ε → 0 and β_H → 0 the limit points are stationary points of the full route-tree, full-horizon objective J.

**Where the iteration we run stands against (a)–(e).** Each gap below is a design choice, not an accident:

| condition | current flagship | gap |
| --- | --- | --- |
| (a) sampled routes | argmax races; CERL exploration units are sampled | the objective is the hard-routed L, not the route expectation J |
| (b) unbiased decision credit | straight-through firing, unweighted probes | biased; same route fixed points as (b) up to linearisation |
| (c) horizon | H = 0 lanes in the running flagship (carry detached); H = 1 in the AWS cross-lane arm | bias up to C max_s γ_s/(1 − max_s γ_s) |
| (d) staleness | 0 (H with CarryCredit) | none |
| (e) steps | OneCycle with AdamW | standard |

Races that renormalise over their woken candidates (the concept wake) define their own exactly normalised model, consistent
with itself but not with the full bank. The truncated partition function can be estimated unbiasedly from the woken set plus
HT-weighted samples of the rest; its logarithm is then consistent as the sample count grows.

## 5. Operating limits

| component | credit | cost | bias / variance | limit |
| --- | --- | --- | --- | --- |
| race readout (repeat/continue/concepts/background) | exact mixture credit r − π over its candidates | candidates scored | exact for the woken set | concept recruitment n*·U/(E·Q) (T2) |
| level reads (top-k) | exact inside; read probes outside | k + E per row | probes biased low by E/n_out (sign-correct) | discovery of far slots: dual race ⟨g, M_s⟩ (T1) |
| writes (top-m) | exact magnitudes; write probes outside | m + E per step | 92% of credit outside the written set (witness) | dual race ⟨δ, Λ_s⟩ gives the exact field at read cost (T1) |
| firing | straight-through | one clock per step | linearised flip value | silence absorbing at zero gates (174 P2) |
| event pointers | straight-through product weights | 2 candidates per level | depth-independent (174 P5) | alternatives only through read probes |
| persistent memory | exact for H lanes (CarryCredit) | (H + 1) lane graphs | truncation bias ∝ γ^H | long clocks need a TD critic (T3) |
| clocks | exact rate gradient; straight-through choice | K clocks | — | long horizons: credit variance 1/(1 − γ²) |

## 6. Consequences for the build, in order

1. **Dual-race write credit:** replace random write probes with the exact counterfactual field restricted to slots read later.
   Credit then flows to the best alternative addresses by ⟨δ_τ, Λ_s(τ)⟩, found with the same sublinear indices as forward
   reads. This removes the largest measured blind spot (92% of write credit).
2. **TD critic over memory:** n-step targets from CarryCredit, bootstrapped by the exact per-slot transport, so credit reaches
   past the H-lane window at its natural per-slot horizon.
3. **Exploration budget scaled with bank size,** E·Q ∝ U·n*/T_budget (T2), and the concept-usage statistics already in every
   milestone to watch recruitment.
4. **Threshold (price) races** in place of fixed top-k for reads, writes and wakes, so routing has fixed points (§4); the
   price is set by the activity budget (homeostasis, P5).
5. **A sampled-route arm** (Gumbel-top-k with full support, score-function credit with a critic baseline), the variant that
   satisfies (a) and (b), paired with the argmax flagship. It tests whether route-local fixed points limit quality.
6. **Probe weighting:** fixed points are invariant to it, so it is a speed/variance choice. Capped HT weights with bank-local
   clipping are the candidate.

## 7. Status

| | status |
| --- | --- |
| T1, T3 | exact identities with float64 witnesses |
| T2 | proposition with proof and witness |
| T4 proposition | proof by first-order (KKT) conditions, corrected; witness confirms settle versus churn |
| consistency theorem | assembled from standard biased-SGD analysis; a proof sketch, not a new convergence proof |
| dual-race credit, TD critic, sampled-route arm | designs, not yet implemented |

## 8. The route-tree limit (theorem, witness, certificate)

Founder: "Wait, you haven't even proven that we approximate the full route tree at a limit? I think that's like the minimum
requirement to having faith that we have something workable as our architecture and learning rules."

**Full route-tree model.** Replace every race by its complete soft form:
- reads r = Σ_s a_s v_s over all slots;
- writes Σ_s ω_s δ over all addresses, with ω = strength × softmax over addresses;
- the concept wake over all units;
- firing by its probability ph (expected correction ph·δ);
- event pointers by their race distribution;
- clock choices by their probabilities (the clock age weight is linear in the choice weights, so the soft mixture is exact).

The readout race already is the exact marginal over its cause tree: competing risks, p(y) = Σ_c π_c q_c(y).

**Theorem A (exact limit).** If every race is complete, every decision soft and the credit horizon unbounded (H = ∞ in
CarryCredit equals full backpropagation through the stream, 173 §19), our training step is exact gradient descent on the full
route-tree, full-depth model. Every credit path is the exact pathwise derivative: residual memory with both credit paths
(173 §16), the dual-state identities of T1, and the exact transport of T3. Standard stochastic-approximation results then give
convergence to its stationary points: L-smooth objective, bounded gradient variance, Robbins–Monro steps, with bounded
staleness allowed (174 P6).

**Theorem B (quantitative approximation).** Let the hard model replace each race d by its truncated form. For a top-k race
with selected set S, renormalised over S, the output becomes Σ_{c∈S} p_c v_c / P_S; for a winner-take-all decision, the
winner. Then

  ‖Σ_all p v − Σ_S p v/P_S‖ = (1 − P_S) ‖v̄_out − v̄_S‖ ≤ 2 m_d V_max,  m_d = 1 − P_S (the excluded mass).

Change the races one at a time in topological order (a hybrid argument). Each change moves the loss by at most Λ_d · 2 m_d
V_max, where Λ_d is the Lipschitz constant of the loss in race output d. Summing:

  |L_hard − L_full| ≤ 2 V_max Σ_d Λ_d m_d + β_H,

with the same form for gradients under Lipschitz-continuous gradients (constants Λ'_d). Here β_H is the horizon bias of T3,
at most C max_s γ_s^{H+1}/(1 − max_s γ_s). The limit points of our iteration are therefore O(Σ_d Λ'_d m_d + β_H)-stationary
for the full route-tree model, and the guarantee tightens exactly as races sharpen (m_d → 0) or k grows. Residual memory keeps
the Λ_d bounded with depth: each level adds a gated, bounded correction (tanh-bounded old-slot corrections, ReZero gates), so
no level multiplies the sensitivity of the ones below.

*Witness* (`experiments/credit/route_tree_limit.py`, `curie_route_tree_limit_witness_20261017_v1`; memory with write and read
races, 30 steps, 16 slots):
- at complete races, loss and gradient gaps are exactly 0;
- across k = 1 … 16 and temperatures 1 … 0.05, the loss gap never exceeds 0.32 × the total excluded mass;
- at k = 4, sharpening the races drives excluded mass and gap to 0 together (temperature 0.1: excluded mass 6e-4, loss gap
  9e-5).

**What the theorem changed in the model.** Our write race had sigmoid magnitudes, independent per address. That is not a
normalised race and has no full route-tree limit: the complete version would write strongly to every address, and its
excluded magnitude does not vanish. Fix: `--write-norm 1`, magnitude = strength × softmax over addresses (contracts v26).
Excluded write mass is now a proper m_d that goes to 0 as the race sharpens. It is in every native flagship file and required
by the coverage gate.

**Route-tree certificate.** Every level reports per step and in final results:
- `cert_read_excluded_mass` (m_d of the top-k read race);
- `cert_write_excluded_mass` (of the normalised write race);
- `cert_fire_uncertain_mass` (mean min(ph, 1 − ph): where hard firing and its expectation differ);
- `cert_clock_nonmax_mass`, `cert_read_nonmax_mass`.

With the horizon term, these give a computable bound on how far each run is from the full route-tree model at any time. A run
whose certificates stay large is optimising a model far from the full tree. That is a measured diagnosis, not a guess.
