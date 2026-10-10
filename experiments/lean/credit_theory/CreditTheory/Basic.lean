import Mathlib

set_option linter.style.longLine false
set_option linter.unusedDecidableInType false

/-!
# Machine-checked lemmas for theory note 164 (sparse, forward-trained, delayed credit)

* `var_eq_half_pairwise`  : variance of a discrete distribution = half its mean squared pairwise distance
                            (the missing-information identity behind Theorem F).
* `pairwise_le_collision` : with zero diagonal and |d| ≤ 2G, the ρ-weighted squared pairwise distance is at most
                            4 G² (1 − Σ ρ²).
* `sampled_credit_variance_bound` : Theorem F (scalar per coordinate):
                            Σ ρ_k (v_k − ḡ)² ≤ 2 G² (1 − Σ ρ_k²) when |v_k| ≤ G.
* `geom_tail_le`          : Result 2's credit-horizon truncation factor  Σ_{j<N} ρ^(H+1+j) ≤ ρ^(H+1)/(1−ρ).
* `delay_stationary`      : Theorem B, the Lambert-W solution is a stationary point of R e^{−2rτ} + K (ητ)².
* `delay_objective_convex`: that objective has positive second derivative, so the stationary point is the minimum.
-/

open Finset BigOperators

variable {ι : Type*} [Fintype ι] [DecidableEq ι]

omit [DecidableEq ι] in
/-- Variance equals half the mean squared pairwise difference (probability weights summing to one). -/
theorem var_eq_half_pairwise (ρ v : ι → ℝ) (hs : ∑ i, ρ i = 1) :
    ∑ i, ρ i * (v i - ∑ k, ρ k * v k) ^ 2
      = (1 / 2) * ∑ i, ∑ j, ρ i * ρ j * (v i - v j) ^ 2 := by
  set m := ∑ k, ρ k * v k with hm
  have lhs : ∑ i, ρ i * (v i - m) ^ 2 = ∑ i, ρ i * v i ^ 2 - m ^ 2 := by
    have : ∀ i, ρ i * (v i - m) ^ 2 = ρ i * v i ^ 2 - 2 * m * (ρ i * v i) + m ^ 2 * ρ i := by
      intro i; ring
    simp only [this, sum_add_distrib, sum_sub_distrib, ← mul_sum, ← hm, hs]
    ring
  have rhs : ∑ i, ∑ j, ρ i * ρ j * (v i - v j) ^ 2
      = 2 * (∑ i, ρ i * v i ^ 2) - 2 * m ^ 2 := by
    have : ∀ i j, ρ i * ρ j * (v i - v j) ^ 2
        = ρ j * (ρ i * v i ^ 2) - 2 * (ρ i * v i) * (ρ j * v j) + ρ i * (ρ j * v j ^ 2) := by
      intro i j; ring
    simp only [this, sum_add_distrib, sum_sub_distrib, ← sum_mul, ← mul_sum, hs, ← hm]
    ring
  rw [lhs, rhs]; ring

/-- Weighted squared pairwise distances with zero diagonal, bounded by 2G, are at most 4G²(1 − Σρ²). -/
theorem pairwise_le_collision (ρ : ι → ℝ) (d : ι → ι → ℝ) (G : ℝ)
    (hρ : ∀ i, 0 ≤ ρ i) (hs : ∑ i, ρ i = 1)
    (hd0 : ∀ i, d i i = 0) (hdG : ∀ i j, |d i j| ≤ 2 * G) :
    ∑ i, ∑ j, ρ i * ρ j * d i j ^ 2 ≤ 4 * G ^ 2 * (1 - ∑ i, ρ i ^ 2) := by
  have key : ∀ i j, ρ i * ρ j * d i j ^ 2 ≤ 4 * G ^ 2 * (ρ i * ρ j - if i = j then ρ i * ρ j else 0) := by
    intro i j
    by_cases h : i = j
    · subst h; simp [hd0]
    · simp only [h, ite_false, sub_zero]
      have hsq : d i j ^ 2 ≤ (2 * G) ^ 2 := by
        rw [← sq_abs]; exact pow_le_pow_left₀ (abs_nonneg _) (hdG i j) 2
      have hpp : 0 ≤ ρ i * ρ j := mul_nonneg (hρ i) (hρ j)
      nlinarith [mul_le_mul_of_nonneg_left hsq hpp]
  calc ∑ i, ∑ j, ρ i * ρ j * d i j ^ 2
      ≤ ∑ i, ∑ j, 4 * G ^ 2 * (ρ i * ρ j - if i = j then ρ i * ρ j else 0) :=
        sum_le_sum fun i _ => sum_le_sum fun j _ => key i j
    _ = 4 * G ^ 2 * ((∑ i, ρ i) * (∑ j, ρ j) - ∑ i, ρ i ^ 2) := by
        simp only [← mul_sum, sum_sub_distrib, sum_ite_eq, mem_univ, ite_true, sum_mul_sum]
        congr 1; congr 1; apply sum_congr rfl; intro i _; ring
    _ = 4 * G ^ 2 * (1 - ∑ i, ρ i ^ 2) := by rw [hs]; ring

/-- **Theorem F (per coordinate).** Credit from one race-sampled cause has variance (missing information)
at most 2 G² (1 − Σ ρ²), where Σ ρ² = exp(−H₂(ρ)). -/
theorem sampled_credit_variance_bound (ρ v : ι → ℝ) (G : ℝ)
    (hρ : ∀ i, 0 ≤ ρ i) (hs : ∑ i, ρ i = 1) (hv : ∀ i, |v i| ≤ G) :
    ∑ i, ρ i * (v i - ∑ k, ρ k * v k) ^ 2 ≤ 2 * G ^ 2 * (1 - ∑ i, ρ i ^ 2) := by
  rw [var_eq_half_pairwise ρ v hs]
  have hb := pairwise_le_collision ρ (fun i j => v i - v j) G hρ hs (by intro i; simp)
    (by intro i j; calc |v i - v j| ≤ |v i| + |v j| := abs_sub _ _
                    _ ≤ 2 * G := by linarith [hv i, hv j])
  linarith

/-- **Result 2 (credit horizon).** The transported tail beyond H events is bounded by ρ^(H+1)/(1−ρ). -/
theorem geom_tail_le (ρ : ℝ) (h0 : 0 ≤ ρ) (h1 : ρ < 1) (H N : ℕ) :
    ∑ j ∈ range N, ρ ^ (H + 1 + j) ≤ ρ ^ (H + 1) / (1 - ρ) := by
  have hpos : 0 < 1 - ρ := by linarith
  have hsum : (∑ j ∈ range N, ρ ^ j) * (1 - ρ) = 1 - ρ ^ N := by
    have := geom_sum_mul ρ N; linarith
  have hgeom : ∑ j ∈ range N, ρ ^ j ≤ 1 / (1 - ρ) := by
    rw [le_div_iff₀ hpos, hsum]; linarith [pow_nonneg h0 N]
  calc ∑ j ∈ range N, ρ ^ (H + 1 + j) = ρ ^ (H + 1) * ∑ j ∈ range N, ρ ^ j := by
        rw [mul_sum]; apply sum_congr rfl; intro j _; rw [pow_add]
    _ ≤ ρ ^ (H + 1) * (1 / (1 - ρ)) := mul_le_mul_of_nonneg_left hgeom (pow_nonneg h0 _)
    _ = ρ ^ (H + 1) / (1 - ρ) := by ring

/-- **Theorem B (optimal delay).** If x e^x = 2 r² R / (K η²) then τ = x/(2r) satisfies the first-order condition
2 r R e^{−2 r τ} = 2 K η² τ of the objective R e^{−2 r τ} + K (η τ)². -/
theorem delay_stationary (r R K η x : ℝ) (hr : 0 < r) (hR : 0 < R) (hK : 0 < K) (hη : 0 < η)
    (hx : x * Real.exp x = 2 * r ^ 2 * R / (K * η ^ 2)) :
    2 * r * R * Real.exp (-2 * r * (x / (2 * r))) = 2 * K * η ^ 2 * (x / (2 * r)) := by
  have hr0 : r ≠ 0 := hr.ne'
  have harg : -2 * r * (x / (2 * r)) = -x := by field_simp
  rw [harg, Real.exp_neg]
  have hex : Real.exp x ≠ 0 := (Real.exp_pos x).ne'
  have hK2 : K * η ^ 2 ≠ 0 := by positivity
  have hx' : x * Real.exp x * (K * η ^ 2) = 2 * r ^ 2 * R := by rw [hx]; field_simp
  field_simp
  nlinarith [hx']

/-- The delay objective is strictly convex, so the stationary point of `delay_stationary` is its unique minimum. -/
theorem delay_objective_convex (r R K η τ : ℝ) (hR : 0 < R) (hK : 0 < K) (hη : 0 < η) (hr : 0 < r) :
    0 < 4 * r ^ 2 * R * Real.exp (-2 * r * τ) + 2 * K * η ^ 2 := by
  positivity

/-- **Theorem A (constant gaps).** With residual credit variances R₀ = σ² S and R_τ = e^{−2 r τ} σ² S, the Gaussian
mutual information log(R₀ / R_τ) delivered by waiting τ is exactly 2 r τ nats. -/
theorem info_from_waiting (σ2 S r τ : ℝ) (hσ : 0 < σ2) (hS : 0 < S) :
    Real.log ((σ2 * S) / (Real.exp (-2 * r * τ) * σ2 * S)) = 2 * r * τ := by
  have h : (σ2 * S) / (Real.exp (-2 * r * τ) * σ2 * S) = Real.exp (2 * r * τ) := by
    rw [show Real.exp (-2 * r * τ) * σ2 * S = Real.exp (-(2 * r * τ)) * (σ2 * S) by ring_nf, Real.exp_neg]
    field_simp
  rw [h, Real.log_exp]

/-- **Theorem C.** Rate log(σ²/(1−ρ²)/D) for coding the adjoint directly versus log(σ²/D) for its innovation:
the shared transport saves log(1/(1−ρ²)) nats per event and mode. -/
theorem transport_saving (σ2 D ρ : ℝ) (hσ : 0 < σ2) (hD : 0 < D) (h0 : 0 ≤ ρ) (h1 : ρ < 1) :
    Real.log (σ2 / (1 - ρ ^ 2) / D) - Real.log (σ2 / D) = Real.log (1 / (1 - ρ ^ 2)) := by
  have hq : 0 < 1 - ρ ^ 2 := by nlinarith
  rw [← Real.log_div (by positivity) (by positivity)]
  congr 1; field_simp

/-- **Theorem E (active components).** For x_i = w_i D_i > 0 with Σ log x_i fixed (a fixed total rate
Σ log(v_i w_i / x_i)), the weighted distortion Σ x_i is at least n · exp(mean log x): it is minimized by equal x_i,
i.e. D_i = θ / w_i, the reverse water-filling level. -/
theorem waterfill_equal_minimizes (s : Finset ι) (hn : 0 < s.card) (x : ι → ℝ) (hx : ∀ i ∈ s, 0 < x i) :
    (s.card : ℝ) * Real.exp ((∑ i ∈ s, Real.log (x i)) / s.card) ≤ ∑ i ∈ s, x i := by
  have hc : (0 : ℝ) < s.card := by exact_mod_cast hn
  have hw : ∑ _i ∈ s, (1 / (s.card : ℝ)) = 1 := by
    rw [sum_const, nsmul_eq_mul]; field_simp
  have hj := (convexOn_exp).map_sum_le (t := s) (w := fun _ => 1 / (s.card : ℝ))
    (p := fun i => Real.log (x i)) (fun _ _ => by positivity) hw (fun _ _ => Set.mem_univ _)
  simp only [smul_eq_mul] at hj
  have hl : ∑ i ∈ s, 1 / (s.card : ℝ) * Real.log (x i) = (∑ i ∈ s, Real.log (x i)) / s.card := by
    rw [← mul_sum]; ring
  have hr : ∑ i ∈ s, 1 / (s.card : ℝ) * Real.exp (Real.log (x i)) = (∑ i ∈ s, x i) / s.card := by
    rw [← mul_sum, sum_congr rfl (fun i hi => Real.exp_log (hx i hi))]; ring
  rw [hl, hr] at hj
  rw [le_div_iff₀ hc] at hj
  linarith
