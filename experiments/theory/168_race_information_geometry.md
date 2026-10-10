# 168 — The information geometry of races: diagonal Fisher information, local natural gradients, and data-limited capacity

10 October 2026, AWS session, founder request: advance the formal understanding of these models' learning dynamics and scaling
"maybe through new points of view". The point of view here is **counting-process martingales** (Andersen, Borgan, Gill &
Keiding 1993): each clock of a race is a counting process; its score is a martingale; orthogonality of martingales that never
jump together gives exact statements about the geometry of learning. Monte Carlo witness (Weibull clocks of four shapes with
censoring, 400K draws): off-diagonal Fisher ≤ 4·10⁻⁴, diagonal = expected wins to 4·10⁻⁴; the which-only (softmax) Fisher has
eigenvalue 0. Algebraic core for exponential clocks machine-checked (`CreditTheory/RaceFisher.lean`).

## 1. Theorem 1 — Timing makes the race's Fisher information diagonal

Let clocks i = 1..K have intensities h_i(t) = e^{θ_i} k_i(t), where k_i is any nonnegative predictable function (it may depend on
elapsed time, on the observed history, on marks, on other parameters, but not on θ). Observe the process on [0, T] (first event
only, a whole sequence, or until censoring). The log-likelihood is ℓ = Σ_i [∫ log h_i dN_i − ∫ h_i dt], so

    ∂ℓ/∂θ_i = N_i(T) − Λ_i(T) =: M_i(T),     Λ_i(T) = ∫_0^T h_i(t) dt,

the compensated counting martingale of clock i. Because two clocks never fire at the same instant (continuous compensators),
⟨M_i, M_j⟩ = 0 for i ≠ j and ⟨M_i⟩ = Λ_i. Hence

    I(θ)_{ij} = E[M_i(T) M_j(T)] = δ_ij · E[Λ_i(T)] = δ_ij · E[N_i(T)].

**The Fisher information in log-scale coordinates is exactly diagonal, and its i-th entry is the expected number of events clock
i wins.** This holds for any hazard shapes, with marks (each mark-specific clock is a clock), history dependence, silence and
censoring. *Exponential special case* (first event, constant rates λ_i, Λ = Σλ): with p_i = λ_i/Λ, E[(1_i − λ_iτ)(1_j − λ_jτ)] =
δ_ij p_i − 2p_ip_j + 2p_ip_j = δ_ij p_i (machine-checked algebra).

**Contrast with a softmax over which-clock only.** Observing the winner but not the time gives Fisher diag(p) − ppᵀ, singular
along the shift direction (all logits +c). Observing *when* removes this gauge freedom: absolute rates become identifiable, and
the coupling between routes disappears.

## 2. Consequences for learning dynamics

1. **The natural gradient is local and closed-form.** For log-scale parameters, natural-gradient step = M_i / E[N_i]: each clock
   needs only its own score and its own expected win count, both locally computable. Per-event race learning with per-clock
   normalization by (a running estimate of) wins *is* natural-gradient learning for these coordinates, no matrix inverse, no
   communication. Adam-like per-parameter normalization approximates it; the exact preconditioner is the win count.
2. **Deeper parameters inherit a diagonal core.** For parameters φ entering through θ(φ), I(φ) = Jᵀ diag(E[N]) J with J = ∂θ/∂φ:
   a Gauss–Newton form whose core is diagonal. Natural-gradient approximations for the whole network therefore reduce to
   per-layer J products with a diagonal weighting by expected wins (a K-FAC-type factorization with an exact diagonal core).
3. **Credit precision equals activity.** By Cramér–Rao, after n sequences any unbiased estimate of θ_i has variance
   ≥ 1/(n E[N_i]). A clock's parameters are learned in proportion to how often it wins: the statistical price of capacity beyond
   activity. Dormant routes are not free to *learn*, even though they are free to *store and infer*.
4. **Learning rates for rare routes.** Plain gradient steps on rarely winning clocks have expected curvature E[N_i] ≪ 1, so they
   learn slowly at a fixed rate; the natural step scales them up by 1/E[N_i]. This is the formal reason exposure-normalized
   (win-normalized) updates are needed for large pools, and why dead or dormant capacity can stay unlearned under plain SGD.

## 3. Consequence for scaling: useful routed capacity is data-limited, with an explicit exponent

Suppose route usage follows a power law p_k ∝ k^{−a} (a > 1; Zipf-like usage is typical of routed memories and mixture experts).
A route's parameters are resolved to precision ε once its expected wins n p_k ≥ 1/ε² (Section 2.3). The number of usefully
learned routes after n events is therefore

    K_eff(n) = #{k : n p_k ≥ ε^{−2}}  ∝  (n ε²)^{1/a},

independent of the provisioned capacity K once K > K_eff. **Useful capacity grows as a power 1/a of the data**: capacity
provisioned beyond K_eff(n) is dormant for learning (though it costs nothing to store or skip, notes 154–155). Prediction for
scaling experiments on the capacity axis: loss improvements from adding routes saturate at K ≈ K_eff(n), and the saturation
point moves as n^{1/a}; with a = 1.2, doubling data adds ≈ 78% more useful routes. Measuring a (from route-win histograms) and
the saturation point gives a scaling law before any large run.

Combined with note 165 Theorem 3 (route credit costs O(1/θ) independent of K): **routed capacity is free in credit traffic but
not in data**, and the data price is quantified by the win distribution's tail exponent.

## 4. A further view: specialization as a symmetry-breaking phase transition (conjecture with a test)

At initialization identical routes form a symmetric fixed point. Under a race with effective temperature T (clock noise scale
relative to logit gaps), the symmetric point loses stability when the data's conditional covariance along the routes' feature
direction exceeds a critical value; for squared-error mixtures this is deterministic annealing's T_c = 2 λ_max (Rose 1998).
For races the analogous threshold should involve the Fisher core of Section 1: a route splits off when its expected wins times
the curvature of the conditional loss exceed the temperature. Predictions: routes specialize in a sequence of discrete splits
as training sharpens the race; the number of specialized routes at a given point equals the number of conditional-covariance
directions above threshold; hash- or content-addressed routing bypasses the transition (no symmetry to break), which may explain
why addressed writes learn quickly. Test: track per-route win entropy and pairwise route-parameter distance over training on R1
or the B1 learner; expect plateaus followed by abrupt splits.

## 5. What this adds to the programme

- A local, exact natural gradient for every race parameter, and the reason win-normalized updates matter for large pools.
- An identifiability statement: timing removes the softmax gauge (a concrete advantage of event models over which-only models).
- A data-scaling law for routed capacity, K_eff ∝ n^{1/a}, complementary to the credit-traffic laws of notes 165–166.
Attribution: counting-process martingales and the Fisher information of point processes (Andersen et al. 1993; Ogata 1978);
natural gradient (Amari 1998); K-FAC (Martens & Grosse 2015); deterministic annealing (Rose 1998). New here: the diagonal-Fisher
consequence for race-of-clocks learning (local natural gradient, win-count preconditioning), the gauge-removal contrast with
softmax routing, and the K_eff ∝ n^{1/a} capacity law for routed temporal memories. Post-boundary, unpublished.
