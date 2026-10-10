# 165 — How much credit must a learning event carry? Scaling laws for sparse, forward-assisted credit

10 October 2026, AWS session, founder direction ("advance the formal theory analytically, in a way that supports our
progress; do things differently than usual"). Follows note 164 (credit as source coding with forward side information).
Question: as width, depth and routed capacity grow, how does the **credit information per learning event** needed for a
given learning fidelity scale, and when is it independent of model size? Results are derived; the combinatorial cores
are machine-checked in [`CreditTheory/Scaling.lean`](../lean/credit_theory/CreditTheory/Scaling.lean).

Conventions as in note 164 Part II: credit components are modelled as independent circular complex Gaussians (the
hardest source under squared error, so rates are upper bounds), rate in nats, distortion is the eligibility-weighted
squared credit error Σ_i w_i D_i, which bounds lost first-order learning progress (note 164, Theorem E).

## 1. Credit lives in state space, not parameter space

**Proposition 1.** If each parameter site knows its local sensitivity e_t (forward eligibility), every exact or
estimated gradient has the form Σ_t ⟨λ̂_t, e_t⟩, so the credit that must be *computed or communicated* per event is the
state cotangent: n numbers for n state units, not P parameters.
For a layer of width w with dense w × w maps, P/n = w: credit traffic is a factor w smaller than gradient traffic, and
the remaining contraction is local multiply-accumulate at the parameter (which a dense learner also pays). This is the
precondition for everything below: the object being coded has the dimension of the *state*, and the rest of the
argument is about how much of that state-sized object must actually be sent.

## 2. A phase transition at a summable residual credit spectrum

Let s_1 ≥ s_2 ≥ … ≥ s_n > 0 be the eigenvalues of the eligibility-weighted covariance of the **residual** credit
λ − λ̂ (after the forward pass's prediction). Optimal coding is reverse water-filling (note 164, Theorem E): components
above a level θ are sent at distortion θ, the rest are silent.

**Theorem 2 (credit rate under a power-law residual spectrum).** Let s_i = s_1 i^{−α}.
- **α > 1 (summable).** Sending the k largest components costs R ≤ α k nats and leaves relative distortion
  ε = D/V ≤ (α/(α−1)) k^{1−α}. Hence fidelity ε needs

      k(ε) = ⌈(α/((α−1)ε))^{1/(α−1)}⌉,     R(ε) ≤ α k(ε)   nats per event,

  **independent of the state dimension n** once n ≥ k(ε).
- **α = 1.** ε ≈ ln(n/k)/ln n, so k ≈ n^{1−ε}: polynomial in n, sub-linear.
- **α < 1.** ε ≈ 1 − (k/n)^{1−α}, so k ≈ n(1−ε)^{1/(1−α)}: linear in n; credit cost grows with model size.

*Proof (α > 1).* Water level θ = s_1 k^{−α}. Rate Σ_{i≤k} log(s_i/θ) = α Σ_{i≤k} log(k/i) = α(k log k − log k!) ≤ α k by
log k! ≥ k log k − k. Distortion kθ + Σ_{i>k} s_i ≤ s_1 k^{1−α} + s_1 k^{1−α}/(α−1) (integral bound on the tail), and
V ≥ s_1. The other regimes follow from Σ_{i≤n} i^{−α} ≍ n^{1−α}/(1−α) (α < 1) and ≍ ln n (α = 1). ∎

**Reading.** Whether sparse credit can make learning cost per event *size-independent* is decided by one measurable
exponent: the decay α of the residual credit spectrum. The forward pass's job, in these terms, is to remove the
**non-summable bulk** of the credit (make α > 1) with information it already has; what remains is a bounded number of
surprises per event. Dense backpropagation always pays n (or P), whatever α is.

## 3. Routed capacity costs no credit: the router spectrum is automatically summable

Consider a layer with K routes (experts, memory slots, clocks), winner-only selection of k routes per event, route
values of width d. Per event the exact credit consists of (i) content credit to the k selected routes, k·d numbers,
and (ii) one scalar of route credit per route (the race statistic; note 160 §2: ρ_j − H_j, or a forced-route return).

**Theorem 3 (credit to routes is bounded independently of K).** The route-credit covariance of one sampled winner
is diag(ρ) − ρρᵀ, whose eigenvalues v_j satisfy Σ_j v_j = 1 − Σ_j ρ_j² ≤ 1 (and v_j ≤ 1). With router score
eligibilities bounded by ‖e_j‖² ≤ W the weighted spectrum has total ≤ W; take W = 1 below (rescale θ by W otherwise).
For any water level 0 < θ ≤ 1:
1. at most 1/θ routes are active (#{j : v_j > θ} ≤ (Σ v_j)/θ ≤ 1/θ);
2. the route-credit rate is at most (1/θ)·ln(1/θ) nats;
3. the silent routes' distortion is at most Σ_{v_j ≤ θ} v_j ≤ 1.
None of these depends on K.

Together with content credit k·d: **credit per event for a routed layer is O(k·d + θ^{−1} ln θ^{−1})**, while a dense
mixture over K routes needs K·d. Capacity beyond activity therefore extends from inference to learning: adding routes
adds memory and options but, at fixed credit precision, no credit traffic. (Items 1–2 are machine-checked:
`count_above_level_le`, `rate_above_level_le`.)

*Interpretation for the families' design.* This is the learning-side counterpart of the inference result "selected
activity, not capacity, sets work" (notes 154–155). It also says how to spend counterfactual credit: losing routes with
ρ_j below the water level receive no message; their learning comes from the aggregate exposure terms, which the race
computes anyway.

## 4. Per-event credit should be extremely sparse: indirect source coding against data noise

What a learner actually needs is the *expected* gradient. One event's credit is u = s + ξ: signal s (its share of the
expected credit, variance S_i per component) plus example noise ξ (variance N_i). Coding u to learn s is an
**indirect (remote) source-coding problem** (Dobrushin–Tsybakov 1962; Witsenhausen 1980). For Gaussians the optimum
codes the Wiener estimate ŝ_i = S_i/(S_i+N_i)·u_i, whose variance is

    S̃_i = S_i · SNR_i/(1 + SNR_i),     SNR_i = S_i / N_i,

and water-fills on S̃_i; the unavoidable error Σ_i S_i/(1+SNR_i) is paid regardless of rate.

**Theorem 4 (silence below the noise floor).** Component i is sent only if S̃_i > θ. With per-event SNR_i ≪ 1, S̃_i ≈
S_i·SNR_i: the effective spectrum is the signal spectrum multiplied by its own SNR, hence steeper. If S_i ∝ i^{−β} and
N_i is flat, S̃_i ∝ i^{−2β}: **per-event credit has exponent 2β**, summable already for β > ½. Batching B events
divides N_i by B and flattens S̃ toward S: larger batches need more credit bits per update (but fewer per event).
Event-level learning is the regime where credit is sparsest, which is the regime our substrate runs in natively.

## 5. Time: per-mode credit and the scale-free memory

For a memory with decay rates r_m, the stationary per-event credit variance of mode m is σ²/(1 − e^{−2 r_m Δ}) ≈
σ²/(2 r_m Δ). If the rates are spread log-uniformly over [r_min, r_max] (multi-scale clocks), the sorted mode variances
decay **geometrically**, v_(i) ∝ (r_max/r_min)^{−i/n}, so k(ε) ≈ n·ln(1/ε)/ln(r_max/r_min): linear in n without a forward
predictor. With the transport shared and the innovation-only credit of note 164 Theorem C, every mode's credit variance
falls to σ² (no 1/r growth), and the time axis contributes no non-summable bulk. The remaining question for the memory
is the spectrum of the innovations themselves across modes, which is a property of the task, not of the memory.

## 6. Depth (conjecture with a test)

Credit at depth ℓ below the output is λ_ℓ = J_{ℓ+1}ᵀ ⋯ J_Lᵀ λ_L. For products of random Jacobians the singular values
spread exponentially with depth (Lyapunov spectrum; Newman 1986), so the credit covariance at lower layers concentrates
in fewer directions: its spectral exponent α_ℓ should **increase** with depth, unless the network is held at dynamical
isometry (Pennington et al. 2017), which flattens the spectrum to protect trainability. This exposes a genuine
trade-off, stated here as a conjecture: *conditioning* (flat Jacobian spectra) and *credit compressibility* (steep
spectra) pull in opposite directions, and a learner with forward-predicted credit can occupy the middle: keep layers
well-conditioned for the forward pass and let the predictor absorb the bulk of the backward spectrum. Empirical support
for low-dimensional gradients in deep nets: Gur-Ari, Roberts & Dyer (2018), "Gradient descent happens in a tiny
subspace".

## 7. What this changes for the program

1. **(Revised by §7b.) One number decides scalability of sparse credit, and dense writes fail it:** the residual spectral exponent α (after forward prediction)
   of each layer's credit. α > 1 ⇒ credit cost per event independent of width; α ≤ 1 ⇒ it grows. This is measurable
   from saved models without training: run the exact credit (traces) and the forward predictor on DEV events and
   eigendecompose the residual covariance per layer. That is the next analytic–empirical bridge; it gives a scaling
   prediction before any scaling run.
2. **Routing is free in credit** (Theorem 3): growing capacity K at fixed activity k is the scaling axis on which our
   learner's cost provably does not grow. Scaling experiments should exploit that axis first.
3. **Event-level learning is the sparsest regime** (Theorem 4): the per-event SNR squares the spectrum's decay.
4. **The forward predictor has a precise target:** maximize α of the residual (remove the non-summable bulk), not
   minimize total residual variance; these differ, and the first is what governs scaling.

**Attribution.** Reverse water-filling and Gaussian rate–distortion (Shannon 1959; Kolmogorov 1956); indirect source
coding (Dobrushin & Tsybakov 1962; Witsenhausen 1980; Wolf & Ziv 1970); Lyapunov spectra of random matrix products
(Newman 1986); dynamical isometry (Pennington, Schoenholz & Ganguli 2017); low-dimensional gradients (Gur-Ari et al.
2018); gradient compression in distributed training (Seide et al. 2014; Alistarh et al. 2017; Wangni et al. 2018). New
here: the state-space credit object with forward side information, the summability phase transition as the scaling
criterion for learning cost, the K-independence of route credit and the squaring of the spectrum at event level, as
properties of the race/temporal-memory substrate. Post-boundary and unpublished (IDF-08).

## 7b. Measurement: the dense-write memory has extensive credit (prediction not supported for this learner)

`credit_spectrum_taxi_n16_n64_r2` (10 Oct 11:05 UTC; Taxi DEV, 7,204 events; online_race `online_trace` learner retrained
with the exact recipe of `online1_taxi_online_trace_s0`, reproducing its DEV log-likelihood digit for digit; exact memory
adjoints; covariance of real 2n coordinates). Rows: residual after no prediction / arrived outcome with shared transport /
4-event exact window.

| Modes (dims) | Predictor | α (top half) | Participation ratio | k(ε = 0.1) | k(ε = 0.01) | Rate at ε = 0.01 |
|---|---|---|---|---|---|---|
| 16 (32) | none | 1.09 | 9.0 | 21 | 30 | 51.2 nats |
| 16 (32) | arrived outcome | 1.29 | 7.2 | 18 | 23 | 41.0 |
| 16 (32) | window 4 | 1.55 | 5.7 | 14 | 18 | 32.8 |
| 64 (128) | none | 1.19 | 25.5 | 60 | 87 | 153.3 |
| 64 (128) | arrived outcome | 1.27 | 23.1 | 57 | 76 | 137.8 |
| 64 (128) | window 4 | 1.19 | 19.3 | 48 | 68 | 119.7 |

**Result.** k(ε) is not flat in n: quadrupling the memory triples it (exponent ≈ 0.76 without prediction, 0.89–0.96 with the
window predictor). The top normalized eigenvalue falls from 0.24 to 0.11: the spectrum *rescales* with n rather than keeping
a fixed head, which is the α ≤ 1 regime of Theorem 2 in the scaling sense even though each spectrum's top half decays with
α ≈ 1.1–1.5. Credit is sublinear in the state dimension (k/2n falls from 0.66 to 0.47 at ε = 0.1) but extensive. The
forward predictors tested remove 20–40% of the components at fixed n; they do not change the scaling. (The 64-mode learner
also ends below the 16-mode one on DEV, 0.4699 vs 0.4812, at the same 20 passes.)

**Diagnosis.** This learner writes **every** memory mode at every event (dense `W_r u` write), so each event injects fresh
credit into all n modes; a fixed summable head cannot exist. Theorem 3's mechanism, selection limiting which components
receive credit, is what bounds credit traffic. **Size-independent credit is therefore a property of sparse addressed writes,
not of temporal memory as such.** For the dense-write diagnostic learner, credit cost grows as ≈ n^{0.8}; for the family's
addressed-write models (k selected slots of K), Theorem 3 predicts credit confined to written slots plus K route scalars with
bounded rate. Next test: the same measurement on an addressed-write memory (R1 keyed memory or a k-of-K write variant of the
Taxi learner) at K = 16 and 64 with fixed k; prediction: k(ε) flat in K.

**Pre-registered prediction for the addressed-write follow-up (`credit_spectrum_v2_taxi_K16_K64`, written 10 Oct 11:50 UTC,
before any v2 result exists).** Each event writes one content-addressed slot of b = 4 modes; addresses hash (mark, gap bucket)
over K slots. A slot j whose modes decay at rate r_j keeps eligibility above 1% of its fresh value for W_j ≈ ln(100)/(2 r_j Δ̄)
events. With roughly uniform addressing, slot j is eligibility-active with probability ≈ 1 − (1 − 1/K)^{W_j} ≤ min(1, W_j/K),
so the expected number of eligibility-active modes is b·Σ_j min(1, W_j/K) ≤ b·(K·W̄)/K = b·W̄: **independent of K** once
K ≫ max_j W_j is not required, only that the per-slot windows do not grow with K (they do not: rates are fixed log-uniform
over the same range). Predictions: (P1) the mean count of modes with eligibility above 1% grows by less than 25% from
K = 16 to K = 64; (P2) the eligibility-weighted k(ε = 0.1) (no predictor) at K = 64 is within 25% of K = 16, while
(P3) the unweighted k(ε = 0.1) grows by at least 1.5× (credit is still dense in state space because the readout reads every
mode). A failure of P2 with P1 holding would mean weighted credit concentrates on modes that eligibility does not
silence, i.e. the sparsity is in the wrong coordinates; a failure of P1 would mean addressing is not spreading writes as
assumed (hash collisions or a few dominant marks).

**Result of the pre-registered test (`credit_spectrum_v2_taxi_K16_K64_s1`, 10 Oct 14:00 UTC; addressed writes, one 4-mode slot
per event, Taxi DEV; DEV LL 0.467 at K = 16, 0.460 at K = 64).**

| Quantity | K = 16 (64 modes) | K = 64 (256 modes) | Ratio | Prediction | Verdict |
|---|---|---|---|---|---|
| modes with eligibility > 1% per event (P1) | 11.78 | 11.49 | 0.98 | < 1.25 | **holds** |
| eligibility-weighted k(ε = 0.1), pooled covariance (P2) | 21 | 78 | 3.71 | 0.75–1.25 | **fails** |
| unweighted k(ε = 0.1) (P3) | 64 | 107 | 1.67 | ≥ 1.5 | **holds** |
| (weighted, window-4 predictor) | 17 | 68 | 4.0 | — | — |
| (weighted participation ratio) | 5.0 | 8.7 | 1.73 | — | — |

**P2 failed as operationalized.** The interpretation pre-registered for "P2 fails while P1 holds" (credit concentrating where
eligibility does not silence it) does not fit: eligibility does silence all but ≈ 11.5 modes per event at both sizes. A
post-hoc explanation, stated as such: k(ε) is computed from the covariance pooled over all events, i.e. the dimension of the
*union* of per-event credit vectors; with addressed writes different events activate different slots, so the union grows with
K even when each event's support stays fixed. Theorem E's rate is per event with the support known to the decoder from
eligibility, which P1 measures. **The capacity-independence claim is therefore not yet established**: it needs the per-event
quantity measured directly (number of eligibility-weighted components above the water level per event, and the per-event rate),
which this driver did not record. Until then the evidence is: per-event active support flat in K (P1), pooled credit dimension
growing ≈ K^0.95 (P2).

## 8. Which credit method for which mode: a per-mode cost law

For mode m with per-event contraction ρ_m = e^{−r_mΔ}, P_m parameters feeding it, and a target relative credit error ε,
three methods are available (all with shared transport):

| Method | Relative residual variance | Credit work per event |
|---|---|---|
| truncation at H events | ρ_m^{2(H+1)} (geometric tail, note 164 Result 2) | ∝ H |
| exact forward trace | 0 | ∝ P_m (one trace per parameter, note 160 §§7, 12) |
| exact window H₀ + TD-learned tail, f features | ≤ ρ_m^{2(H₀+1)} · κ_f / (1 − ρ_m²) | ∝ H₀ + f |

The third row is the on-policy linear TD bound (Tsitsiklis & Van Roy 1997: ‖Φw* − V‖ ≤ ‖ΠV − V‖/√(1−γ²)) with the
discount replaced by the mode's contraction; κ_f is the relative error of the best predictor in the feature span.

**Theorem 8 (per-mode method choice).** Truncation reaches ε with H_m(ε) = ⌈ln(1/ε)/(2r_mΔ)⌉ − 1 events. It is
cheaper than an exact trace exactly when

    r_m > r_c(ε) = ln(1/ε) / (2Δ P_m),

and a learned tail beats both only when its features are good enough to survive the slow-mode amplification,
κ_f ≤ ε(1 − ρ_m²)ρ_m^{−2(H₀+1)}. Total credit work per event is therefore

    W(ε) = Σ_m min( ln(1/ε)/(2r_mΔ), P_m, H₀ + f [if κ_f admissible] ),

which, for decay rates spread log-uniformly over [r_min, r_max], gives exact traces only to the slowest modes, a
fraction ln(r_c/r_min)/ln(r_max/r_min) of them, and short windows to the rest.

*Reading.* Fast memories need almost no credit machinery; slow memories need exact traces (fixed cost, no history)
unless the forward pass supplies features that predict their long-horizon credit well, and the bar for those
features rises as 1/(1 − ρ²) ≈ 1/(2rΔ). The learned credit model is most valuable at intermediate time scales, not at
the slowest ones, the opposite of the naive intuition that learned credit is for "long" dependencies. Test: the
credit-spectrum measurement (`credit_spectrum.py`) records per-mode residuals for the window-4 predictor, which checks
the ρ^{2(H+1)} law per mode.

## 9. Does sparsity grow as the credit model learns the domain? (founder question, 10 Oct)

Write the residual credit variance of component i at training time t as v_i(t) = v_i^irr + a_i(t): v_i^irr = Var(λ_i | all
forward information) is unpredictable in principle (it contains the data noise of Theorem 4), and a_i(t) ≥ 0 is the credit
predictor's error, which falls as the predictor learns the domain's regularities. Under water-filling (note 164 Theorem E) the
active set is A(θ, t) = {i : w_i v_i(t) > θ}.

**Proposition 9.1 (sparsity grows with learning).** If every a_i(t) is non-increasing in t, so is |A(θ, t)|, and the credit
rate R(t) = Σ_{i∈A} log(w_i v_i(t)/θ) is non-increasing. *Proof:* each w_i v_i(t) is non-increasing; membership in A and each
log term can only fall. ∎

**Proposition 9.2 (savings grow with scale: conditions).** Savings S(n, t) = C_dense(n)/R(n, t) with C_dense ∝ n. If
(A) the predictor is a shared function of local forward features whose parameter count and approximation error do not grow
with n, and (B) the unpredictable spectrum is summable uniformly in n (sup_n Σ_i w_i v_i^irr < ∞ with a fixed head), then
R(n, t) → R_∞(θ) bounded in n as a_i → 0, and S(n, t) grows linearly in n. If (B) fails, S saturates at a constant factor
however good the predictor becomes (the dense-write learner of §7b, k ∝ n^0.8, is in this regime).

**The hidden-cost caveat.** A predictor whose small error ε is spread uniformly over n components has total residual n·ε.
Components with w_i ε < θ are silent (zero traffic), so measured traffic savings grow with n, but the omitted credit is a
*bias* of size n·ε in the update: the cost moves from communication to learning quality. Scale-growing savings are real only
if predictor errors concentrate (condition B for a_i too) or are corrected by unbiased sparse audits (notes 162–163), whose
variance must then be charged. Report distortion Σ_{i∉A} w_i v_i alongside traffic.

**Test (cheap, no new architecture):** record k(ε), residual distortion and traffic per epoch at two sizes during training
(credit_spectrum drivers evaluate after training only; per-epoch logging is the extension). Prediction under (A)+(B):
k(ε, t) falls over training and the K = 64 / K = 16 traffic ratio falls below the capacity ratio as training proceeds.

## 10. Credit should start heavy and become cheaper as surprise falls (founder principle, 10 Oct)

Use the predicted-credit-plus-audit form (notes 162–163): component i has predicted credit, surprise r_i = exact − predicted,
audit cost c_i and inclusion probability p_i; the unbiased estimator adds variance Σ(1/p_i − 1)‖r_i‖².

**Theorem 10.1 (cost scales with the square of surprise).** For any p_i > 0 with Σ‖r_i‖²/p_i ≤ V,

    Σ_i c_i p_i  ≥  (Σ_i ‖r_i‖ √c_i)² / V,

with equality at p_i ∝ ‖r_i‖/√c_i (when no p_i saturates at 1). Machine-checked: `audit_cost_lower_bound`
(Cauchy–Schwarz). Hence the optimal schedule has three phases:
1. **Heavy:** early, the credit model is ignorant, ‖r_i‖ is large, all p_i saturate at 1 and credit is exact (dense).
2. **Cheapening:** once below saturation, cost ∝ (surprise)²; halving typical surprise quarters the cost. If surprise falls
   like t^{−1/2} (estimation error of a predictor fitting a stationary target), per-event cost falls like 1/t and the
   **cumulative credit cost over training grows like log T**, against T for dense credit.
3. **Floor:** surprise cannot fall below the unpredictable part (data noise σ_i). With the variance budget matched to the
   noise SGD already tolerates (V = κΣσ_i²), the floor cost relative to dense is (Σσ_i√c_i)²/(κΣσ_i²Σc_i) ≤ 1/κ by
   Cauchy–Schwarz: small when the irreducible noise concentrates in few components, up to 1/κ when spread evenly
   (condition (B) of §9).
**Self-regulation.** Under a domain shift surprise rises, the p_i rise toward 1 and credit becomes heavy again until the model
re-learns: no schedule is set by hand. **Requirement:** the allocation needs ‖r_i‖ before the audit, so the credit model must
also predict its own surprise; that estimate must be calibrated, and allocation must use only information available before
the draw (an over-confident surprise estimate under-audits and inflates variance, never biases the mean).

## Formal verification (Lean 4 + Mathlib)

[`CreditTheory/Scaling.lean`](../lean/credit_theory/CreditTheory/Scaling.lean); every theorem depends only on
`propext`, `Classical.choice`, `Quot.sound` (no `sorry`):

| Lean theorem | Statement | Note 165 |
|---|---|---|
| `count_above_level_le` | #{v_j > θ} ≤ (Σ v)/θ | Theorem 3.1 |
| `rate_above_level_le` | Σ_{v_j>θ} log(v_j/θ) ≤ (1/θ) log(1/θ) when v_j ≤ 1, Σ v ≤ 1, 0 < θ ≤ 1 | Theorem 3.2 (K-independent route credit) |
| `sum_log_succ_eq_log_factorial`, `powerlaw_rate_le` | Σ_{i=1}^{k} log(k/i) = k log k − log k! ≤ k | Theorem 2 rate (R ≤ αk) |
| `audit_cost_lower_bound` (Surprise.lean) | Σc_i p_i ≥ (Σ‖r_i‖√c_i)²/V under Σ r_i²/p_i ≤ V | §10 cost ∝ surprise² |
| `powerlaw_tail_le` | Σ_{i=k}^{N−1} (i+1)^{−α} ≤ k^{1−α}/(α−1), α > 1, independent of N | Theorem 2 silent tail |

Together these give Theorem 2's size independence for α > 1 without appeal to asymptotics: the rate to send the top k
components is at most αk nats, and the silent tail is bounded by s₁k^{1−α}/(α−1) for every state dimension N.

