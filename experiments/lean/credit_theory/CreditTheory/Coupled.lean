import Mathlib

set_option linter.style.longLine false

/-!
# Machine-checked core of theory note 166 (coupled forward–credit learning)

`coupled_contraction`: for a nonnegative comparison system x' ≤ a x + b e, e' ≤ c x + d e with a, d < 1 and
(1 − a)(1 − d) > b c, there are w > 0 and q < 1 such that the potential x + w e contracts by q every step.
`coupled_converges`  : consequently the potential decays geometrically, Φ_t ≤ q^t Φ_0.
-/

theorem coupled_contraction (a b c d : ℝ) (ha0 : 0 ≤ a) (ha : a < 1) (hb : 0 < b) (hc : 0 ≤ c)
    (hd0 : 0 ≤ d) (hd : d < 1) (hdet : b * c < (1 - a) * (1 - d)) :
    ∃ w q : ℝ, 0 < w ∧ q < 1 ∧ 0 ≤ q ∧
      (∀ x e x' e' : ℝ, 0 ≤ x → 0 ≤ e → x' ≤ a * x + b * e → e' ≤ c * x + d * e →
        x' + w * e' ≤ q * (x + w * e)) := by
  have h1a : 0 < 1 - a := by linarith
  have h1d : 0 < 1 - d := by linarith
  -- feasible weights: b/(1-d) < w and c w < 1 - a; take w = (b/(1-d) + (1-a)/c)/2 when c > 0
  set lo := b / (1 - d) with hlo
  have hlo_pos : 0 < lo := div_pos hb h1d
  rcases eq_or_lt_of_le hc with hc0 | hcpos
  · -- c = 0: take w = 2 lo
    subst hc0
    refine ⟨2 * lo, max a ((b + d * (2 * lo)) / (2 * lo)), by positivity, ?_, le_max_of_le_left ha0, ?_⟩
    · apply max_lt ha
      rw [div_lt_one (by positivity)]
      have : b = lo * (1 - d) := by rw [hlo]; field_simp
      nlinarith
    · intro x e x' e' hx he hx' he'
      have hq1 : a ≤ max a ((b + d * (2 * lo)) / (2 * lo)) := le_max_left _ _
      have hq2 : (b + d * (2 * lo)) / (2 * lo) ≤ max a ((b + d * (2 * lo)) / (2 * lo)) := le_max_right _ _
      have hq2' : b + d * (2 * lo) ≤ max a ((b + d * (2 * lo)) / (2 * lo)) * (2 * lo) := by
        rwa [div_le_iff₀ (by positivity)] at hq2
      nlinarith [mul_le_mul_of_nonneg_left hq1 hx, mul_le_mul_of_nonneg_left hq2' he]
  · set hi := (1 - a) / c with hhi
    have hlohi : lo < hi := by
      rw [hlo, hhi, div_lt_div_iff₀ h1d hcpos]; linarith
    set w := (lo + hi) / 2 with hw
    have hw_pos : 0 < w := by rw [hw]; linarith
    have hcw : a + c * w < 1 := by
      have : w < hi := by rw [hw]; linarith
      have : c * w < c * hi := mul_lt_mul_of_pos_left this hcpos
      have : c * hi = 1 - a := by rw [hhi]; field_simp
      linarith
    have hbw : (b + d * w) / w < 1 := by
      rw [div_lt_one hw_pos]
      have hlw : lo < w := by rw [hw]; linarith
      have : b = lo * (1 - d) := by rw [hlo]; field_simp
      nlinarith
    refine ⟨w, max (a + c * w) ((b + d * w) / w), hw_pos, max_lt hcw hbw,
      le_max_of_le_left (by positivity), ?_⟩
    intro x e x' e' hx he hx' he'
    have hq1 : a + c * w ≤ max (a + c * w) ((b + d * w) / w) := le_max_left _ _
    have hq2 : (b + d * w) / w ≤ max (a + c * w) ((b + d * w) / w) := le_max_right _ _
    have hq2' : b + d * w ≤ max (a + c * w) ((b + d * w) / w) * w := by
      rwa [div_le_iff₀ hw_pos] at hq2
    nlinarith [mul_le_mul_of_nonneg_left hq1 hx, mul_le_mul_of_nonneg_left hq2' he,
      mul_le_mul_of_nonneg_left he' hw_pos.le]

/-- Geometric decay of the weighted potential under a one-step contraction. -/
theorem coupled_converges (q : ℝ) (hq0 : 0 ≤ q) (Φ : ℕ → ℝ) (hstep : ∀ t, Φ (t + 1) ≤ q * Φ t) :
    ∀ t, Φ t ≤ q ^ t * Φ 0 := by
  intro t
  induction t with
  | zero => simp
  | succ n ih =>
    calc Φ (n + 1) ≤ q * Φ n := hstep n
      _ ≤ q * (q ^ n * Φ 0) := mul_le_mul_of_nonneg_left ih hq0
      _ = q ^ (n + 1) * Φ 0 := by ring
