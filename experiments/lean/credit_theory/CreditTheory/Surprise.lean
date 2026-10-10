import Mathlib

set_option linter.style.longLine false

/-!
# Surprise-adaptive credit cost (theory note 165 §10)

`audit_cost_lower_bound`: for audit probabilities p_i > 0 with residual-audit variance Σ r_i²/p_i ≤ V, the expected audit
cost Σ c_i p_i is at least (Σ |r_i| √c_i)² / V. Credit cost scales with the square of the surprise.
-/

open Finset BigOperators

theorem audit_cost_lower_bound {ι : Type*} (s : Finset ι) (r c p : ι → ℝ) (V : ℝ)
    (hc : ∀ i ∈ s, 0 ≤ c i) (hp : ∀ i ∈ s, 0 < p i) (hV : 0 < V)
    (hvar : ∑ i ∈ s, r i ^ 2 / p i ≤ V) :
    (∑ i ∈ s, |r i| * Real.sqrt (c i)) ^ 2 / V ≤ ∑ i ∈ s, c i * p i := by
  -- Cauchy–Schwarz with a_i = |r_i|/√p_i and b_i = √(c_i p_i)
  have hcs : (∑ i ∈ s, (|r i| / Real.sqrt (p i)) * Real.sqrt (c i * p i)) ^ 2
      ≤ (∑ i ∈ s, (|r i| / Real.sqrt (p i)) ^ 2) * (∑ i ∈ s, Real.sqrt (c i * p i) ^ 2) :=
    Finset.sum_mul_sq_le_sq_mul_sq s _ _
  have h1 : ∀ i ∈ s, (|r i| / Real.sqrt (p i)) * Real.sqrt (c i * p i) = |r i| * Real.sqrt (c i) := by
    intro i hi
    have hpi : 0 < Real.sqrt (p i) := Real.sqrt_pos.mpr (hp i hi)
    rw [Real.sqrt_mul (hc i hi)]
    field_simp
  have h2 : ∀ i ∈ s, (|r i| / Real.sqrt (p i)) ^ 2 = r i ^ 2 / p i := by
    intro i hi
    rw [div_pow, sq_abs, Real.sq_sqrt (hp i hi).le]
  have h3 : ∀ i ∈ s, Real.sqrt (c i * p i) ^ 2 = c i * p i := by
    intro i hi
    exact Real.sq_sqrt (mul_nonneg (hc i hi) (hp i hi).le)
  rw [sum_congr rfl h1, sum_congr rfl h2, sum_congr rfl h3] at hcs
  have hcp : 0 ≤ ∑ i ∈ s, c i * p i := sum_nonneg fun i hi => mul_nonneg (hc i hi) (hp i hi).le
  rw [div_le_iff₀ hV]
  calc (∑ i ∈ s, |r i| * Real.sqrt (c i)) ^ 2
      ≤ (∑ i ∈ s, r i ^ 2 / p i) * ∑ i ∈ s, c i * p i := hcs
    _ ≤ V * ∑ i ∈ s, c i * p i := mul_le_mul_of_nonneg_right hvar hcp
    _ = (∑ i ∈ s, c i * p i) * V := by ring
