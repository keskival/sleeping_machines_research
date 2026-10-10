import Mathlib

set_option linter.style.longLine false

/-!
# Exponential-race Fisher information is diagonal (theory note 168, algebraic core)

For constant rates λ_i > 0 with total Λ, first-event observation gives E[1_i] = p_i = λ_i/Λ, E[τ 1_i] = p_i/Λ,
E[τ²] = 2/Λ². The score covariance E[(1_i − λ_i τ)(1_j − λ_j τ)] equals δ_ij p_i:
`race_fisher_offdiag` (i ≠ j: E[1_i 1_j] = 0) and `race_fisher_diag` (i = j: E[1_i] = p_i).
-/

/-- Off-diagonal entry: 0 − λ_j E[τ1_i] − λ_i E[τ1_j] + λ_iλ_j E[τ²] = 0. -/
theorem race_fisher_offdiag (li lj Λ : ℝ) (hΛ : 0 < Λ) :
    0 - lj * ((li / Λ) / Λ) - li * ((lj / Λ) / Λ) + li * lj * (2 / Λ ^ 2) = 0 := by
  field_simp; ring

/-- Diagonal entry: E[1_i] − 2 λ_i E[τ1_i] + λ_i² E[τ²] = p_i. -/
theorem race_fisher_diag (li Λ : ℝ) (hΛ : 0 < Λ) :
    li / Λ - 2 * li * ((li / Λ) / Λ) + li ^ 2 * (2 / Λ ^ 2) = li / Λ := by
  field_simp; ring
