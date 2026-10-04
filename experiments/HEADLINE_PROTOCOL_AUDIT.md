# P0 comparison and queue audit — 4 October 2026

This protects P0-1/2/5/6. It changes no fitted model, active training source, queue or saved metric.

The current table has **five quality/work wins against saved references**, at their stated evidence level. Four
Transformer comparisons use the same reset T256 windows and 999,936 targets as native. The LSTM inference win
compares against continuous-state scoring of 999,999 targets on the same interval. LSTM losses have the same context
difference. These remain explicit saved-reference comparisons; they are not identical-context LSTM evidence.
The original score/protocol assertions remain in older report artifacts; the new scoreboard records the correction.

The E64 `score()` LSTM branch carries `state` through 4096-token blocks. Its Transformer branch and native
`window_scores()` reset overlapping windows, scoring the first window whole and subsequent second halves. E174
already documented continuous-state LSTM and rescored all 999,999 targets for its own count-mixture protocol;
it is not a native-window rescore. The 63 tail targets are one difference; the retained context is another.
No direction or magnitude of the rescore change is assumed.

`reference_window_rescore.py` isolates the two pinned E64 model class definitions, loads the actual saved checkpoint
and resets state on every T256 lane. The first queue uses the saved 10M LSTM-512 weights, unchanged parent and raw
test bytes; identical model classes were checked against historical producer source. Parent/weights/data/source/
queue identity and inherited physical reservation precede numerical imports. The report admits a completed rescore
only with its matching manifest, parent, zero updates and exact target population; the original score remains recorded.
There is no numerical result yet. Missing 10M LSTM-256 and 90M checkpoints are not replaced by newly trained models.

The fourteen P0-6 queues all fit their budgets: A/B use completed native whole-fit work; C/D use projected four-pass
native work. Five LSTM arms require aligned validation and test scoring. Align validation before selecting the best
arm across architectures, and wait for the complete declared group. Existing runs/queues retain their original scoring.
This is a repair requirement, not a test-selected fallback or a prediction of a win.

Publication fixes: 90M budget eligibility is enforced before selecting a score; missing T256 and nonfinite results are
excluded; actual pass counts and optimizer windows replace fixed labels; official Mackey–Glass uses its canonical
collector's source/data/settings/repeat checks. Partial/duplicate/mismatched repeat sets cannot become leaderboard
wins. The scoreboard precedes the family chapter on PDF page 2 without deleting that chapter.
The P0-6 loader also requires every declared arm's completed provenance, exact queue settings, fitting count and
actual budget. It does not announce a tuned winner from the first finished arm or mix numerical producer versions.
The groups remain pending until all arms finish and validation/test contexts are aligned.

Completed, standard-library-only checks:

- [Headline checks](results/diagnostics/headline_comparisons_stdlib_20261004T214500Z.json): 12 groups covering
  budget/protocol/nonfinite/missing metrics, actual preflight, target geometry, historical model definitions and
  malformed official repeat sets. Numerical library imports are forbidden.
- [Tuned queue budgets](results/diagnostics/tuned_reference_budgets_stdlib_20261004T213500Z.json): all 14 within
  budget, five context mismatches explicitly identified. No numerical execution or queue launch.
- [Tuned-group admission](results/diagnostics/tuned_reference_admission_stdlib_20261004T214500Z.json): empty,
  partial and unaligned complete groups stay pending; eight malformed completed-arm cases and over-budget work
  are rejected before publication. These are synthetic JSON fixtures, not numerical fits.

Next physical action remains the product order: P0-1, P0-6 10M, then P0-3, with confirming seeds as ordered.
The [prepared rescore](queue/aws_lstm512_native_windows_20261004T213000Z/README.md) is a short inference-only
protocol repair admitted by the AWS owner through `run_safe.sh`, after the existing P0 owners, with one thread,
1.25 GB RSS cap, 3 GB address space cap and 8 GiB available floor. No fourth job and no ML work in Docker.
