# Formal-analysis tutorials (internal training material, confidential)

All eight planned tutorials and both learning extensions are complete. Each is three pages.

Introductory tutorials to the mathematics needed to read our theory notes (`experiments/theory/160`–`168`) and the
machine-checked lemmas (`experiments/lean/credit_theory`). One topic per file; each has derivations, exercises with
solutions, and a "where this appears in our work" box. Build one tutorial from `tutorials/`: `latexmk -pdf <file>.tex` (shared style `tutorial.sty`). Rebuild tutorials 4–8 with `scripts/build_tutorials.py --tag <unique-tag>`; it checks text extraction, required sections and layout before publishing. The Lean PDF is instructional; a fresh proof build requires the pinned toolchain separately.
Post-boundary material: internal only until the priority filing (see `ip/`).

| # | Tutorial | Status |
|---|---|---|
| 1 | [Temporal point processes and races of clocks](01_point_processes_and_races.pdf) · [source](01_point_processes_and_races.tex) | done (3 pp.) |
| 2 | [Adjoints, eligibility traces and temporal credit](02_adjoints_traces_temporal_credit.pdf) · [source](02_adjoints_traces_temporal_credit.tex) | done (3 pp.) |
| 3 | [Gradient estimators and variance reduction](03_gradient_estimators_variance_reduction.pdf) · [source](03_gradient_estimators_variance_reduction.tex) | done (3 pp.) |
| 4 | [Information theory for learning signals](04_information_theory_for_learning_signals.pdf) · [source](04_information_theory_for_learning_signals.tex) | done (3 pp.) |
| 5 | [Learning dynamics: contraction, Bellman equations and TD](05_learning_dynamics_contraction_bellman_td.pdf) · [source](05_learning_dynamics_contraction_bellman_td.tex) | done (3 pp.) |
| 6 | [Counting-process martingales, Fisher information and natural gradients](06_counting_processes_fisher_natural_gradients.pdf) · [source](06_counting_processes_fisher_natural_gradients.tex) | done (3 pp.) |
| 7 | [Spectra, scaling and queues](07_spectra_scaling_and_queues.pdf) · [source](07_spectra_scaling_and_queues.tex) | done (3 pp.) |
| 8 | [Formal verification with Lean 4](08_formal_verification_with_lean4.pdf) · [source](08_formal_verification_with_lean4.tex) | done (3 pp.) |
| 9 | [Forward computation teaches credit](09_forward_teaches_credit.pdf) · [source](09_forward_teaches_credit.tex) | done (3 pp.) |
| 10 | [Forward feedback trains update functions](10_causal_update_functions.pdf) · [source](10_causal_update_functions.tex) | done (3 pp.) |

## Coverage and reading path

4. **Information theory for learning signals** — entropy, mutual information, Gaussian channels; rate–distortion of a Gaussian
   (½ log(v/D) per real Gaussian; log(v/D) per circular complex Gaussian), reverse water-filling (weighted form, sparsity as the optimum); side information and Wyner–Ziv (no rate loss for
   Gaussians; value of a forward statistic = I(λ;Y)); indirect source coding (Wiener filtering against data noise). Our use:
   note 164 Part II, note 165 §§2–4.
5. **Learning dynamics** — perturbed gradient descent on strongly convex losses; coupled two-system comparison and a weighted
   Lyapunov potential (2×2 nonnegative matrices, (1−a)(1−d) > bc); Bellman equations and TD(0)/n-step, contraction modulus,
   linear-TD error bound 1/√(1−γ²); the adjoint as a Bellman equation with a complex gap-dependent discount; Kolen–Pollack
   feedback alignment. Our use: notes 164 Result 2, 165 §8, 166.
6. **Counting-process martingales and Fisher geometry** — compensators, predictable covariation, orthogonality of processes
   without common jumps; Fisher information = E[⟨M⟩]; Cramér–Rao; natural gradient; Gauss–Newton structure; softmax gauge
   (singular Fisher) vs timed races (diagonal). Our use: note 168; witness `experiments/credit/race_fisher_mc.py`.
7. **Spectra, scaling and queues** — power-law spectra and summability (α > 1 vs ≤ 1), sums vs integrals, effective rank /
   participation ratio; pooled vs per-event dimension (the lesson of note 165 §7b); Laplace transforms of renewal processes
   (note 167); Little's law (note 166); Zipf usage and K_eff ∝ n^(1/a) (note 168).
8. **Formal verification with Lean 4** — what a proof assistant checks; reading a Mathlib statement; `field_simp`/`nlinarith`/
   `positivity`; `#print axioms`; walk through `CreditTheory/Basic.lean`, `Scaling.lean`, `Coupled.lean`, `Surprise.lean`,
   `RaceFisher.lean`; what is not formalized (Shannon theory) and why.
