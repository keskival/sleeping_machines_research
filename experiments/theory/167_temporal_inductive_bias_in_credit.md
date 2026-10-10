# 167 — Where temporal structure helps credit assignment: the forward race predicts the credit's uncertainty

10 October 2026, AWS session, founder question: "Are there places in the credit assignment model where our temporal components
would produce a benefit? Where inductive biases are causal, within episodes?" Builds on notes 164 (adjoint as Bellman equation
with the memory's transport; optimal delay), 165 §§9–10 (surprise-adaptive audits need a calibrated surprise estimate) and 160 §2
(race posterior). Derivation; the measurement is proposed, not yet run.

## 1. Future credit is a Laplace transform of the forward model's own predicted gaps

For memory mode m with a_j = e^{(−r + iω)τ_j}, the adjoint is λ_t = c_t + Σ_{j≥1} (Π_{k≤j} a_{t+k}^*) c_{t+j}. Let the forward race
at state s_t predict the next-gap density f_t(τ) (its mixture of exponential and delayed clocks), with Laplace transform
L_t(s) = E_{f_t}[e^{−sτ}]. Under a renewal approximation (later gaps i.i.d. f_t, cotangents independent of gaps with mean c̄ and
variance σ² per mode):

    E[λ_t − c_t | s_t]   = c̄ · L_t(r + iω) / (1 − L_t(r + iω)),
    Var[λ_t − c_t | s_t] = σ² · L_t(2r) / (1 − L_t(2r)).

*Proof.* E[Π_{k≤j} a_{t+k}^*] = L_t(r + iω)^j and E[Π_{k≤j} |a_{t+k}|²] = L_t(2r)^j; sum the geometric series (|L_t| < 1 for
r > 0) and use independence of the cotangents. The variance line is exact for c̄ = 0; with c̄ ≠ 0 the gap
randomness adds Var[c̄ Σ_j Π_k a_{t+k}^*], which vanishes as the fitted model's mean score E[c] → 0 (§3). ∎

For an exponential clock of rate μ, L_t(s) = μ/(μ + s), so

    E[tail] = c̄ · μ/(r + iω),        Var[tail] = σ² · μ/(2r):

**the expected credit surprise of a mode is event rate × mode memory time**, read directly from the forward race's parameters.
Delayed (log-normal) clocks give L_t numerically by quadrature over their few components; the race mixture's Laplace transform
is the mixture of its components' transforms (for a race of independent clocks, through the survival-weighted first-arrival
density f(τ) = Σ_i h_i(τ) S(τ)).

## 2. Where this produces a benefit, causally within an episode

1. **Calibrated surprise for audit allocation.** Note 165 §10 allocates audits as p_i ∝ predicted ‖r_i‖/√c_i and needs that
   prediction before the audit. The Laplace variance gives it in closed form at every event, with no learned parameters
   beyond running per-mode σ² estimates: bursts (high predicted μ) raise the audit rate for slow modes, quiet periods lower it,
   and the mode's decay sets the scale. Miscalibration only adds variance, never bias (§10).
2. **State-dependent optimal delay.** Note 164 Theorem B's R(0) becomes R_t(0) = σ²L_t(2r)/(1 − L_t(2r)): credit for a mode is
   applied sooner during bursts (information arrives quickly) and later in quiet periods.
3. **Credit flows during silence.** The exposure part of the race credit (note 160 §2: ρ_i − H_i) accumulates at the predicted
   hazard between events; the credit memory evolves by the same lazy transport as the forward memory, so dormant modes need
   no computation until read, and silence is supervision for credit as it is for prediction.
4. **Surprise is self-exciting.** A large audited surprise indicates model error that persists over the episode's near future;
   a decaying-clock (Hawkes-type) estimator of recent audited surprise, Ŝ_t = Σ_{audits u < t} e^{−κ(t−u)}‖r_u‖, combined with
   the Laplace prior, adapts audit rates within an episode to local model error. This is the within-episode causal inductive
   bias the race substrate supplies natively (the same clock primitive, applied to credit errors).

## 3. Why mean prediction disappointed and magnitude prediction should not

For a well-fitted model, E[c] ≈ 0 (the expected score vanishes), so the mean tail above is near zero and window/mean credit
predictors removed only 20–40% of the residual in note 165 §7b. What timing structure predicts well is the credit's **magnitude**:
Var[tail] varies by orders of magnitude across modes (1/r) and over time (μ_t). Sparse credit needs exactly this, deciding where to
spend audits, not a better mean.

## 4. Test (no new training; extend the credit-spectrum measurement)

On saved DEV events, compute per event and mode the Laplace-predicted tail variance from the forward head's clock parameters and
compare with the realized squared tail |λ_t − c_t|²: rank correlation and calibration slope per mode; then the audit-cost
reduction of note 165 §10 with the Laplace allocation vs uniform allocation at equal variance. Prediction: strong positive rank
correlation (bursts and slow modes carry the large tails) and a cost reduction that grows with the spread of decay rates.

Attribution: Laplace transforms of renewal processes and discounted sums (standard renewal theory); Hawkes self-excitation
(Hawkes 1971). New here: using the forward race's predicted gap law as the credit model's uncertainty prior per memory mode, as
the input to surprise-adaptive sparse credit. Post-boundary, unpublished (IDF-08).
