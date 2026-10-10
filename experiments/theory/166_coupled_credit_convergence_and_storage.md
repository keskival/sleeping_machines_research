# 166 — Coupled forward–credit learning: a contraction condition, and how much pending credit must be stored

10 October 2026, AWS session (founder direction: advance the formal theory analytically in support of progress).
Answers two items in curie's required-proof list ([RECIPROCAL_EVENT_LEARNING.md](RECIPROCAL_EVENT_LEARNING.md) §8):
"coupled-learning stability under stated assumptions" and "complete storage … as functions of … credit horizon and
pending-record count; fixed packet size alone does not bound outstanding packets under unlimited delay". Builds on notes
160 §11 (mutually trained feedback), 164 (delays, TD credit) and 165 (water-filling sparsity). The contraction core is
machine-checked: [`CreditTheory/Coupled.lean`](../lean/credit_theory/CreditTheory/Coupled.lean).

## 1. Coupled learning converges when the credit model out-runs its moving target

**Setting.** Forward parameters θ minimize L, locally μ-strongly convex and L-smooth with minimizer θ*. The forward
update uses credit produced by a separately parameterized credit system ψ (feedback matrices B, credit predictor
weights, cause posterior):

    θ_{t+1} = θ_t − η (∇L(θ_t) + b_t),       ‖b_t‖ ≤ M ‖e_t‖,

where e_t = ψ_t − ψ*(θ_t) is the credit system's error relative to the exact credit for the *current* forward
parameters (b_t = 0 when ψ = ψ*). The credit system contracts toward its current target at rate κ per unit step β,
and that target moves with θ:

    ‖ψ_{t+1} − ψ*(θ_t)‖ ≤ (1 − βκ) ‖e_t‖,      ‖ψ*(θ') − ψ*(θ)‖ ≤ G ‖θ' − θ‖.

**Theorem 1 (local linear convergence of the coupled learner).** With x_t = ‖θ_t − θ*‖, ε_t = ‖e_t‖ and η ≤ 1/L,

    x_{t+1} ≤ (1 − ημ) x_t + ηM ε_t,
    ε_{t+1} ≤ ηGL x_t + (1 − βκ + ηGM) ε_t,

and the pair converges to zero geometrically whenever 1 − βκ + ηGM < 1 and

    βκ  >  η M G (1 + L/μ).                                              (★)

*Proof.* The first line is the strongly convex gradient step with an additive error ηb_t. For the second,
e_{t+1} = (ψ_{t+1} − ψ*(θ_t)) + (ψ*(θ_t) − ψ*(θ_{t+1})), and ‖θ_{t+1} − θ_t‖ ≤ η(Lx_t + Mε_t). The comparison system
has a nonnegative 2×2 matrix A = [[a, b], [c, d]] with a = 1 − ημ, b = ηM, c = ηGL, d = 1 − βκ + ηGM. If a, d < 1 and
(1 − a)(1 − d) > bc, there is a weight w > 0 and q < 1 with Φ_t = x_t + w ε_t satisfying Φ_{t+1} ≤ q Φ_t (machine-checked
as `coupled_contraction`). (1 − a)(1 − d) > bc reads ημ(βκ − ηGM) > η²MGL, i.e. (★). ∎

**Reading (★).** The credit system must contract faster than η·M·G·(1 + κ_L), the product of the forward step, how
strongly credit errors bias the forward step (M), how fast the exact credit moves when θ moves (G), and the condition
number. This turns curie's caution ("two-time-scale schedules alone are not a convergence proof") into a checkable
inequality with measurable constants. Instances already in the programme:
- **Mutually trained feedback** (note 160 §11). B receives exactly W's update and both decay at rate λ, so
  B − W ← (1 − ηλ)(B − W) exactly: the credit system receives the same move as its target, so the comparison system has
  c = 0 and d = 1 − ηλ, and (1 − a)(1 − d) > bc = 0 holds for every λ > 0: the measured cosine 0.99998 and −0.0021 nats against exact transposes is the
  predicted behaviour, and fixed random feedback (κ = 0) is predicted to fail, as it did (−0.30 nats).
- **TD-learned credit tail** (note 164 Result 2). With linear features, κ ≈ (1 − ρ_m)·λ_min(feature covariance); the
  slower the mode (ρ_m → 1), the smaller κ, and (★) demands a smaller forward step or a faster credit learner. This
  is the dynamical counterpart of note 165 Theorem 8 (learned credit is least suitable for the slowest modes).
- **Cause posterior q** (note 160 §§6, 9). G is the sensitivity of the exact posterior to θ; near a sharp posterior
  (low collision entropy, note 164 Theorem F) G is small and (★) is easy.

Scope: local (strong convexity near θ*), deterministic bounds on stochastic updates' means; noise adds the usual
O(η) neighbourhood. It is a sufficient condition, not a global convergence theorem for nonconvex learning.

## 2. Pending credit storage is bounded by the sent-credit rate times the optimal delay

Credit messages are created at events, held for a delay, then applied (note 164 Result 3). By **Little's law**
(L = λW, valid for any stationary queue with finite means), the mean number of outstanding credit records is

    E[N_pending] = Σ_m λ_m^send · E[τ_m],

where λ_m^send is the rate at which component m actually *sends* (not the event rate) and τ_m its delay.

**Theorem 2 (storage law).** With the optimal per-mode delay of note 164 Theorem B and water-filling silence (note 165
Theorem E):
1. τ*_m ≤ σ²/(2Kη²Δ) for every mode (the plateau of Theorem B; finite whenever staleness has a price K > 0), so delay
   is never unlimited under the optimal schedule;
2. λ_m^send = 0 below the water level, so only the k(ε) active components contribute; for a summable residual spectrum
   (α > 1) k(ε) is independent of model size (note 165 Theorem 2), and for route credit at most 1/θ routes send
   (note 165 Theorem 3);
3. hence E[N_pending] ≤ λ_event · k(ε) · σ²/(2Kη²Δ): bounded independently of the number of parameters, routes and
   dormant modes, growing only with the event rate, the fidelity target and the inverse staleness price.

Each record holds a cotangent of the sending component (O(1) numbers) plus its version stamp, so pending storage in
numbers is O(k(ε) · τ* · λ_event). Compare BPTT: stored history O(n · T) for sequence length T, and exact forward
traces O(Σ_m P_m) per stream (note 160 §12). The three regimes trade: traces cost parameters, BPTT costs sequence
length, delayed sparse credit costs (active components × delay).

**Caveat.** Little's law gives the mean; tail bounds need the delay and send distributions (e.g. deterministic delays
give N_pending ≤ (max send burst)·τ*). Records whose parameters change while pending carry staleness already priced in
Theorem B; deleting a record is a declared truncation (curie §3).

## 3. Consequence

Two of curie's required items now have explicit, checkable conditions:
- coupled stability: measure M (credit-bias gain), G (target drift), κ (credit contraction) and verify (★) on the
  online learners before scaling; the existing §11 result is an instance;
- storage: report λ_send, mean delay and N_pending; Theorem 2 predicts N_pending flat in pool size K and width at fixed
  ε, a direct test on the depth/capacity axes of DEEP_LEARNING_SCALING.md.

Attribution: linear convergence of perturbed gradient descent and two-time-scale stochastic approximation (Borkar
1997; Konda & Tsitsiklis 2004); Lyapunov weighting of coupled comparison systems (standard nonnegative-matrix argument);
Little (1961). Kolen–Pollack alignment (1994; Akrout et al. 2019). New here: the explicit coupling constants for the
credit systems of this programme and the storage law combining Little's law with the derived optimal delays and
water-filling silence. Post-boundary, unpublished (IDF-08).
