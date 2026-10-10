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

1. **One number decides scalability of sparse credit:** the residual spectral exponent α (after forward prediction)
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

## Formal verification (Lean 4 + Mathlib)

[`CreditTheory/Scaling.lean`](../lean/credit_theory/CreditTheory/Scaling.lean); every theorem depends only on
`propext`, `Classical.choice`, `Quot.sound` (no `sorry`):

| Lean theorem | Statement | Note 165 |
|---|---|---|
| `count_above_level_le` | #{v_j > θ} ≤ (Σ v)/θ | Theorem 3.1 |
| `rate_above_level_le` | Σ_{v_j>θ} log(v_j/θ) ≤ (1/θ) log(1/θ) when v_j ≤ 1, Σ v ≤ 1, 0 < θ ≤ 1 | Theorem 3.2 (K-independent route credit) |
| `sum_log_succ_eq_log_factorial`, `powerlaw_rate_le` | Σ_{i=1}^{k} log(k/i) = k log k − log k! ≤ k | Theorem 2 rate (R ≤ αk) |
| `powerlaw_tail_le` | Σ_{i=k}^{N−1} (i+1)^{−α} ≤ k^{1−α}/(α−1), α > 1, independent of N | Theorem 2 silent tail |

Together these give Theorem 2's size independence for α > 1 without appeal to asymptotics: the rate to send the top k
components is at most αk nats, and the silent tail is bounded by s₁k^{1−α}/(α−1) for every state dimension N.

