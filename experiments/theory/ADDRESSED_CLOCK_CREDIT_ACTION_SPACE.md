# Addressed clock credit: information and actuation must match

B10,13October2026. The native critic now reads individual mode theta/g/m/v but its retained action is still four uniform clock corrections. Better input identifiability does not remove a restricted output action space.

Let c in R64 add to the two clock-gradient legs of16modes in two layers. The retained action is c=Aa, a in R4, where each column of A is uniform1/sqrt16 within one layer/leg and zero elsewhere. A^T A=I. For actual clippedAdam plus refreshed producer state and eight future targets, define q=partial L/partial c at the issued cut. Then

    q_parallel = A A^T q
    q_omitted = q - q_parallel
    partial L/partial a = A^T q.

The last identity is a chain rule, not an approximation. Under a Euclidean gradient-action radius r, unrestricted optimal first-order loss reduction is r||q||; retained reduction is r||A^T q||. Parameter-step or forecast-curvature trust metrics require their corresponding metric projection, not this Euclidean formula. The actual optimizer's nonlinear global clip is included in measured q; full actor learning still has its ordinary gradients. Rank4 restricts only the extra learned correction, not the whole actor update.

## Completed native cut diagnostic

Source-bound curie_b10_clock_action_rank_20261013_v1: three real TRAIN learner cuts, same actor/Adam and observed state. Four-action derivative agrees with projected64-coordinate derivative within5.43e-20. Actual functional loss parity≤3.55e-11. Omitted squared clock-demand fractions0.937550/0.980824/0.962290. Actual small1e-5steps in the inaccessible negative-demand direction reduce future loss by1.05e-9/2.92e-8/9.77e-9. The first tiny difference is near the region where nonlinear/roundoff effects matter; no precision-optimal or practical gain claim is drawn from it. These are three local eight-target demand witnesses, not population coverage,64parameter total-learning dominance, whole-learner demand, delayed24target policy value or benchmark quality.

This explains a concrete expressive limitation; it does not prove the rank restriction caused every recent failed fit. A learned critic must acquire and generalize these demands before the larger action space is useful.

## Next integrated design

Retain current actor, temporal computation, hard races, private addressed critic banks, typed mode receipts, separate keys/values, actual actor clip/Adam and the exact null-parent contract. Add a zero-initialized bounded per-mode residual to the retained parent action. Route credit to selected clock targets; candidate target embeddings identify layer/mode/leg and local operator state. Shared temporal functions can transfer across target contexts, while address identity preserves mode differences.

First compare with fixed target support before learning admission. A sparse support's local rank can stay small while its union of possible addressed supports spans all64coordinates; four uniform columns cannot do so. Sparse updates alone do not eliminate candidate scoring or own-gradient/optimizer work. Charge discovery, losing-route teachers, message payloads, receipts, decoder/optimizer work and globalclip contraction separately. Availability64 and selected activity are distinct counters.

Numerical gate: zero residual preserves exact parent predictions/RNG; single addressed correction matches realAdam and future evaluation; pooled old actions reproduce old chain-rule derivatives. Integrated fit gate: same restored actor and source/data budget, paired future tapes, actual forecast utility, prefix/full prequential trajectories. Existing parent results retained until improved. Three-cut sensitivity is a design reason, never a predicted win.
