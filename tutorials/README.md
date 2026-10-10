# Formal-analysis tutorials (internal training material, confidential)

Introductory tutorials to the mathematics needed to read our theory notes (`experiments/theory/160`–`168`) and the
machine-checked lemmas (`experiments/lean/credit_theory`). One topic per file; each has derivations, exercises with
solutions, and a "where this appears in our work" box. Build: `latexmk -pdf <file>.tex` (shared style `tutorial.sty`).
Post-boundary material: internal only until the priority filing (see `ip/`).

| # | Tutorial | Status |
|---|---|---|
| 1 | [Temporal point processes and races of clocks](01_point_processes_and_races.pdf) | done (3 pp.) |
| 2 | [Adjoints, eligibility traces and temporal credit](02_adjoints_traces_temporal_credit.pdf) | done (3 pp.) |
| 3 | [Gradient estimators and variance reduction](03_gradient_estimators_variance_reduction.pdf) | done (3 pp.) |
| 4 | Information theory for learning signals | to write |
| 5 | Learning dynamics: contraction, Bellman equations and TD | to write |
| 6 | Counting-process martingales, Fisher information and natural gradients | to write |
| 7 | Spectra, scaling and queues | to write |
| 8 | Formal verification with Lean 4 | to write |

## Outlines for the remaining tutorials (same format as 1–3: ~3 pages, exercises with solutions)

4. **Information theory for learning signals** — entropy, mutual information, Gaussian channels; rate–distortion of a Gaussian
   (log(v/D)), reverse water-filling (weighted form, sparsity as the optimum); side information and Wyner–Ziv (no rate loss for
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
