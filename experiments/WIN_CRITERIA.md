# What counts as a win

User direction (4 October 2026): beating Transformers or LSTMs at matched compute — training, inference or total — is a
win, and the agents developing and reporting models must say so. We also pursue pure-accuracy wins over the
strongest references by spending matched compute on our own models (more passes or more capacity). This file defines the
terms once, for every agent and every document. State each win plainly as a win; state its scope once, next to it, not as
a hedge in every sentence.

**Primary objective, user direction 5 October:** the strategic target is a
reproducible quality/work advantage over strong Transformers in larger-data
language regimes where they lead, followed by a confirmed scaling trend.
Small-scale LSTM/count wins still count within their declared scope; they do
not establish frontier advantage. LANGUAGE_CROSSOVER_PLAN.md defines the
equal-compute crossover study and the modern-protocol continuation.

## Win types

| Win | Definition | Example wording |
|---|---|---|
| **Matched-compute win (training)** | Same data, protocol and test set; our whole-fit compute ≤ the reference's; our quality better | "Wins against the 4-pass Transformer at 0.40× its training compute (1.888 vs 1.908 bpc)." |
| **Matched-compute win (inference)** | Same protocol/test; our per-position (or per-query) inference compute ≤ the reference's; our quality better | "Wins against LSTM-256 at matched inference compute (1.955 vs 2.171 bpc, 0.60 vs 0.68 MFLOPs/position)." |
| **Matched-compute win (total)** | Training plus inference over the declared deployment workload ≤ the reference's; quality better | State the workload (e.g. inference on N characters). |
| **Pure-accuracy win** | Better quality than the strongest saved reference on the same protocol, any compute | "Beats every saved 10M reference: X vs 1.799 bpc." |
| **Leaderboard win** | Better than every entry of a cited public leaderboard under its official protocol and metric | "New best on the NeuroBench primate-reaching leaderboard (R² X vs AEGRU 0.71)." |
| **Pareto win** | Better quality and lower compute on the same axis at once (strictly dominates the reference) | Implied by a matched-compute win with lower compute. |

A comparison where we use less compute but have worse quality, and no reference exists at our compute, is an
**efficiency point**: report the gap and the compute ratio, then close it — with a matched-compute reference or a native
run at the reference's compute. It is neither a win nor a loss.

## Attempt levels (always attach one to a benchmark result; user direction 6 October 2026)

| Level | Meaning | How to report |
|---|---|---|
| **First-pass variant** | An existing or quickly assembled member with at most knob-level tuning (width, pool, learning rate, decay), without task-specific design or error-analysis-driven iteration | "First-pass variant: X vs reference Y." Never a verdict on the family; never a front-door LOSS. |
| **In development** | Owned battle; design rationale written; iterating on development data | "In development: best X vs Y (development data)." |
| **Developed attempt** | The owner worked a documented improvement plan (design, error analysis, iterations) and ran the sealed protocol | Win at its evidence level, or **developed-attempt loss** with the diagnosis of what the family lacks there |

Win/loss wording applies to developed attempts. Earlier benchmark losses keep their numbers and are relabelled by
attempt level where that changes their interpretation (Mackey-Glass: about 20 knob-level development runs of one member; primate reaching: about 10 on one session;
banknote: fixed selection opportunities. All three are first-pass variants).

## Evidence levels (always attach one)

- **Win (single seed)** — one completed run; normal for development and the first report.
- **Confirmed win** — at least two seeds (or the official repeat protocol), all better than the reference, or a mean
  difference larger than the seed spread.
- **Leaderboard claims** require the official protocol run once with a configuration fixed beforehand (development on
  other data or validation), and the full official repeat/session set.

## Conventions, stated once per table or paragraph

- Native compute: traced arithmetic + special functions (fitting extrapolated from traced windows; inference from the
  exact winner-only trace or the evaluated member). References: the saved shape estimates (lm_training_flops.py) or the
  leaderboard's published counts. The conventions differ; say so once.
- Same data, same scored test targets, same evaluation window (state T).
- Optimizer updates are not a compute axis; state them when they differ (e.g. "with 1.5× its updates"), but they do not
  disqualify a compute win.
- Saved references are the project's E64 controls unless a published value is cited; say which.

## How developing agents should aim

1. Pick the reference and the axis before running (e.g. "beat LSTM-512 at ≤ 433 TFLOPs training").
2. Spend the matched budget on our own model: passes, width, credited capacity (sampled credit keeps pool training cost
   flat), depth — whichever the bottleneck diagnosis supports.
3. When a run wins, report it as a win at its evidence level and queue the confirming seed.
4. Report losses and efficiency points with the same clarity; do not soften a loss either.

## Current standing (keep updated)

10M text8, T = 256, single seeds (report: native language appendix, "Matched-compute comparisons"):
- **Wins:** Transformer-256×2 (training and inference budgets), Transformer-256×4 4 passes (training: 352 vs 889 TF;
  inference: 1.32 vs 7.41 MF/position; 1.888 vs 1.908), LSTM-256 (inference budget: 1.955 vs 2.171).
- **Losses:** LSTM-256 at its training budget (best within 20 TF: 2.326 vs 2.171); LSTM-512 6 passes at both budgets
  (1.888 vs 1.799 at 0.81× training compute).
- **90M:** efficiency point (1.857 vs 1.661 / 1.604 at 1/16–1/33 of the compute); multi-pass native runs at about half
  the LSTM's compute are queued to settle it.

## Information matching: FAS oracle-assisted diagnostics

A fair reference must receive the same available training and inference information as the evaluated model. Hidden TRAIN process/item identities used to learn routes or transition-gap distributions constitute privileged supervision even when test predictions are identity-free. FAS FIFO and the v2 timing-aware probe therefore belong in an oracle-assisted diagnostic table and are excluded from strongest-reference selection and win/loss verdicts. Retain their measurements and the native single-seed 0.600 vs 0.559 AUROC win against the six generic controls at N=256.
