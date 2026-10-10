import Mathlib

set_option linter.style.longLine false

/-!
# Machine-checked cores of theory note 165 (credit-rate scaling)

* `count_above_level_le`  : Theorem 3.1, at most (Σ v)/θ components exceed the water level θ (route credit: ≤ 1/θ).
* `rate_above_level_le`   : Theorem 3.2, route-credit rate Σ_{v_j > θ} log(v_j/θ) ≤ (1/θ)·log(1/θ), independent of K.
* `powerlaw_rate_le`      : Theorem 2 rate step, Σ_{i=1}^{k} log(k/i) ≤ k (so R ≤ α k for a power-law spectrum).
-/

open Finset BigOperators

/-- At most (Σ v)/θ components of a nonnegative spectrum exceed the level θ. -/
theorem count_above_level_le {ι : Type*} (s : Finset ι) (v : ι → ℝ) (θ : ℝ) (hθ : 0 < θ)
    (hv : ∀ j ∈ s, 0 ≤ v j) :
    ((s.filter (fun j => θ < v j)).card : ℝ) ≤ (∑ j ∈ s, v j) / θ := by
  rw [le_div_iff₀ hθ]
  calc ((s.filter (fun j => θ < v j)).card : ℝ) * θ
      = ∑ _j ∈ s.filter (fun j => θ < v j), θ := by rw [sum_const, nsmul_eq_mul]
    _ ≤ ∑ j ∈ s.filter (fun j => θ < v j), v j := sum_le_sum fun j hj => (mem_filter.mp hj).2.le
    _ ≤ ∑ j ∈ s, v j := sum_le_sum_of_subset_of_nonneg (filter_subset _ _)
          (fun j hj _ => hv j hj)

/-- **Theorem 3.2.** If the route-credit variances are at most 1 and sum to at most 1, the rate spent above the water
level θ ∈ (0, 1] is at most (1/θ)·log(1/θ), whatever the number of routes. -/
theorem rate_above_level_le {ι : Type*} (s : Finset ι) (v : ι → ℝ) (θ : ℝ) (hθ : 0 < θ) (hθ1 : θ ≤ 1)
    (hv : ∀ j ∈ s, 0 ≤ v j) (hv1 : ∀ j ∈ s, v j ≤ 1) (hsum : ∑ j ∈ s, v j ≤ 1) :
    ∑ j ∈ s.filter (fun j => θ < v j), Real.log (v j / θ) ≤ (1 / θ) * Real.log (1 / θ) := by
  have hlog : 0 ≤ Real.log (1 / θ) := Real.log_nonneg (by rw [le_div_iff₀ hθ]; linarith)
  have hterm : ∀ j ∈ s.filter (fun j => θ < v j), Real.log (v j / θ) ≤ Real.log (1 / θ) := by
    intro j hj
    have hj' := mem_filter.mp hj
    apply Real.log_le_log (div_pos (by linarith [hj'.2]) hθ)
    exact div_le_div_of_nonneg_right (hv1 j hj'.1) hθ.le
  have hcount := count_above_level_le s v θ hθ hv
  have hcount' : ((s.filter (fun j => θ < v j)).card : ℝ) ≤ 1 / θ :=
    le_trans hcount (div_le_div_of_nonneg_right hsum hθ.le)
  calc ∑ j ∈ s.filter (fun j => θ < v j), Real.log (v j / θ)
      ≤ ∑ _j ∈ s.filter (fun j => θ < v j), Real.log (1 / θ) := sum_le_sum hterm
    _ = ((s.filter (fun j => θ < v j)).card : ℝ) * Real.log (1 / θ) := by rw [sum_const, nsmul_eq_mul]
    _ ≤ (1 / θ) * Real.log (1 / θ) := mul_le_mul_of_nonneg_right hcount' hlog

/-- Σ_{i<k} log(i+1) = log k!. -/
theorem sum_log_succ_eq_log_factorial (k : ℕ) :
    ∑ i ∈ range k, Real.log ((i : ℝ) + 1) = Real.log (k.factorial : ℝ) := by
  induction k with
  | zero => simp
  | succ n ih =>
    rw [sum_range_succ, ih, Nat.factorial_succ, Nat.cast_mul,
      Real.log_mul (by positivity) (by exact_mod_cast (Nat.factorial_pos n).ne')]
    push_cast; ring

/-- **Theorem 2, rate step.** Σ_{i=1}^{k} log(k/i) = k log k − log k! ≤ k, so a power-law spectrum water-filled at
its k-th component costs at most α k nats. -/
theorem powerlaw_rate_le (k : ℕ) :
    ∑ i ∈ range k, Real.log ((k : ℝ) / ((i : ℝ) + 1)) ≤ k := by
  rcases Nat.eq_zero_or_pos k with rfl | hk
  · simp
  have hkpos : (0 : ℝ) < k := by exact_mod_cast hk
  have hsplit : ∑ i ∈ range k, Real.log ((k : ℝ) / ((i : ℝ) + 1))
      = k * Real.log k - Real.log (k.factorial : ℝ) := by
    rw [← sum_log_succ_eq_log_factorial]
    have : ∀ i ∈ range k, Real.log ((k : ℝ) / ((i : ℝ) + 1)) = Real.log k - Real.log ((i : ℝ) + 1) :=
      fun i _ => Real.log_div hkpos.ne' (by positivity)
    rw [sum_congr rfl this, sum_sub_distrib, sum_const, card_range, nsmul_eq_mul]
  have hfac : (0 : ℝ) < k.factorial := by exact_mod_cast Nat.factorial_pos k
  have hexp : (k : ℝ) ^ k / k.factorial ≤ Real.exp k := Real.pow_div_factorial_le_exp _ hkpos.le k
  have hlog := Real.log_le_log (by positivity) hexp
  rw [Real.log_exp, Real.log_div (by positivity) hfac.ne', Real.log_pow] at hlog
  rw [hsplit]; linarith

/-- **Theorem 2, tail step.** For α > 1 and 1 ≤ k ≤ N, Σ_{i=k}^{N−1} (i+1)^{−α} ≤ k^{1−α}/(α−1): the silent tail of a
summable power-law credit spectrum is bounded independently of the state dimension N. -/
theorem powerlaw_tail_le (α : ℝ) (hα : 1 < α) (k N : ℕ) (hk : 1 ≤ k) (hkN : k ≤ N) :
    ∑ i ∈ Ico k N, ((i + 1 : ℕ) : ℝ) ^ (-α) ≤ (k : ℝ) ^ (1 - α) / (α - 1) := by
  have hk0 : (0 : ℝ) < k := by exact_mod_cast hk
  have hanti : AntitoneOn (fun x : ℝ => x ^ (-α)) (Set.Icc (k : ℝ) N) := by
    intro x hx y hy hxy
    have hx0 : 0 < x := lt_of_lt_of_le hk0 hx.1
    exact Real.rpow_le_rpow_of_nonpos hx0 hxy (by linarith)
  have hsum := AntitoneOn.sum_le_integral_Ico hkN hanti
  have hint : ∫ x in (k : ℝ)..(N : ℝ), x ^ (-α) = ((N : ℝ) ^ (-α + 1) - (k : ℝ) ^ (-α + 1)) / (-α + 1) := by
    apply integral_rpow
    right
    refine ⟨by linarith, ?_⟩
    rw [Set.uIcc_of_le (by exact_mod_cast hkN)]
    intro h; linarith [h.1]
  rw [hint] at hsum
  have hN0 : (0 : ℝ) ≤ (N : ℝ) ^ (-α + 1) := Real.rpow_nonneg (Nat.cast_nonneg N) _
  have hden : (-α + 1) < 0 := by linarith
  have hpos : (0 : ℝ) < α - 1 := by linarith
  have h1 : ((N : ℝ) ^ (-α + 1) - (k : ℝ) ^ (-α + 1)) / (-α + 1)
      ≤ (k : ℝ) ^ (1 - α) / (α - 1) := by
    have hre : ((N : ℝ) ^ (-α + 1) - (k : ℝ) ^ (-α + 1)) / (-α + 1)
        = ((k : ℝ) ^ (1 - α) - (N : ℝ) ^ (-α + 1)) / (α - 1) := by
      rw [show (1 - α) = (-α + 1) by ring, show (α - 1) = -(-α + 1) by ring, div_neg, ← neg_div]
      ring
    rw [hre]
    exact div_le_div_of_nonneg_right (by linarith) hpos.le
  exact le_trans hsum h1
