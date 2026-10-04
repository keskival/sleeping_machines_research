# Public benchmark selection audit — 4 October 2026

Completed saved-prediction audit; no additional training or public win.

| Dataset | TEST accuracy mean, seeds6/7/8 | Seed SD, percentage points | Mean whole fit, GFLOPs estimated/run | Mean fitting MFLOPs/presented series |
| --- | ---: | ---: | ---: | ---: |
| ECG200 | 72.33% | 1.53 | 149.12 | 114.704 |
| JapaneseVowels | 93.96% | 2.73 | 173.15 | 29.150 |
| PenDigits | 95.96% | 0.82 | 257.73 | 0.955 |

All nine accuracies and negative log likelihoods were reconstructed from saved
probabilities and hash-verified official labels. The audit checked exact IDs,
completed prescribed seeds, data/source/configuration/selection lineage, selected
DEV epochs and fitting-work denominators. Three seeds reuse the same100/370/3498
TEST examples, respectively; seed SD is not a generalization confidence interval.
Footprint/active state, trained inference, candidate discovery, energy and matched
reference TEST/work comparisons remain separate confirmation requirements.

The selected large model’s paired DEV NLL gain over the small model was0.006670
on ECG200, with descriptive paired standard error0.058877, for11.354× estimated
whole-screen fitting work. On JapaneseVowels the gain was0.072779, standard
error0.072276, for11.116× work. DEV has20 and54 examples respectively and already
selected epochs/configurations, so these are descriptive comparisons without
post-selection significance. They do not show that the smaller final model
would have improved TEST accuracy. Full forty-epoch screen costs are compared
under the same denominator; no deployment saving is inferred.

Screens plus finals used approximately1,067.21GF on ECG200,952.68GF on
JapaneseVowels and5,121.28GF on PenDigits. Pilots, contracts, failed attempts,
evaluation, preprocessing and compilation are additional. All estimates use
eager-window extrapolation, with padding/length limitations preserved.

A prospective three-or-more-seed DEV selector now prefers lower fitting cost
when paired differences lie within a predeclared empirical tolerance. This is
an allocation heuristic, not a proof of noninferiority. Historical selections
are unchanged. Freeze fresh TRAIN-only split replications and compare same-epoch
with same-update refits before additional archive TEST comparisons. Existing
primate confirmation and AWS replay owners retain their allocations.

Source artifact:
[completed audit](../../experiments/results/diagnostics/public_campaign_evidence_20261004T025800Z.json).
Implementation:
[evidence audit](../../experiments/public_benchmarks/evidence_audit.py),
[prospective selector](../../experiments/public_benchmarks/selection_policy.py).
Interpretation and next comparisons:
[theory150](../../experiments/theory/150_public_selection_uncertainty_and_compute_allocation.md).
