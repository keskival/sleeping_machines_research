# Public selection uncertainty and compute allocation

4 October 2026. Completed evidence, prospective allocation policy; no new fit.

## Failure addressed

The fixed AWS archive campaign completed all nine final seed refits. Mean official
TEST accuracy is 72.3333% on ECG200, 93.9640% on JapaneseVowels and 95.9596% on
PenDigits. These results do not establish the target public advantage. A larger
model was selected on very small DEV sets for the first two tasks, while the
smaller model led PenDigits. Neither a tiny DEV lead nor a DEV–TEST gap identifies
a gradient implementation bug or a general expressive limitation.

`public_benchmarks/evidence_audit.py` reconstructed all nine accuracies and NLLs
from hash-verified official labels and saved probabilities. It checked exact
prediction IDs, normalized finite probabilities, completed three-seed coverage,
unchanged screen/refit programs and configurations, DEV-only parent selection,
all candidate/parent/data hashes, minimum-NLL epoch selection, full-refit
presentation counts and fitting-work denominators. Historical choices and
scores are unchanged. This complements the owner’s 030000Z refit audit.

Completed artifact:
`results/diagnostics/public_campaign_evidence_20261004T025800Z.json`.

Reproduce with a fresh, unused output filename (standard library only):

```sh
python3 -m experiments.public_benchmarks.evidence_audit \
  --selection experiments/results/public_benchmarks/aws_public_ECG200_selection_20261004T000100Z.json \
  --selection experiments/results/public_benchmarks/aws_public_JapaneseVowels_selection_20261004T000100Z.json \
  --selection experiments/results/public_benchmarks/aws_public_PenDigits_selection_20261004T000100Z.json \
  --output /tmp/public_campaign_audit_fresh.json
```

Archive files must match the committed data manifest. If absent, the existing
`python3 -m experiments.public_benchmarks.fetch` restores exact committed archive
bytes; it does not train or change the manifest.

## What the paired DEV observations say

Compare the selected p32/D4/pool4 with the p16/D2/pool2 screen, using the same
examples at each candidate’s selected epoch. Define each difference as selected
minus small per-example NLL. Negative favors the selected model.

| Task | DEV examples | Mean NLL difference | Paired standard error | Selected/small whole-screen work |
| --- | ---: | ---: | ---: | ---: |
| ECG200 | 20 | −0.006670 | 0.058877 | 11.354× |
| JapaneseVowels | 54 | −0.072779 | 0.072276 | 11.116× |

Both candidates have the same selected DEV accuracy within each task: 90% and
96.2963%, respectively. These are weak grounds for the large compute premium.
This does **not** establish that a small final refit would be more accurate;
that comparison has not been run. Forty-epoch screen costs are compared with
forty-epoch screen costs. They do not represent an inference saving.

DEV selected epochs/configurations already. The displayed standard errors and
normal intervals are descriptive and omit selection bias, seed uncertainty and
any example dependence. They are not post-selection significance certificates.
Three final seeds share the same TEST examples; their seed SD does not multiply
the number of independent observations by three.

Only one JapaneseVowels example is wrong for all three seeds, despite mean
error of 6.04%. Prediction disagreements are 6.22–11.62% across seed pairs. This
supports investigating training/routing variability. It does not demonstrate
that deployment ensembling beats a reference: no ensemble was selected or
scored here, and averaging predictions must pay all its inference work.

## Separation of error sources

Let R(q) be expected NLL for the fixed adapter and deployment protocol and Q the
available predictor family. Then, with minima replaced by infima if needed,

    R(q_hat) − H(Y|raw_input)
      = I(Y;raw_input|adapter(raw_input))
      + [inf(q in Q) R(q) − H(Y|adapter(raw_input))]
      + [R(q_hat) − inf(q in Q) R(q)].

The first term is adapter information loss (theory149), the second is family
approximation error, and the third includes fitting/optimization and finite-data
generalization. A small DEV selection does not measure any one term separately.
Deep-route message credit, persistent-write credit, truncated history,
regularization and data transfer can all affect the third term. Preserve their
existing contracts and negative findings rather than assigning the entire gap
to clipping or replacing the sparse temporal core.

Screen-to-full-refit also changes training dynamics. Holding epochs fixed changes
the number of Adam updates and normalization statistics. For ECG200 batch32,
80-example FIT uses three updates per epoch; full TRAIN100 uses four. Thirteen
epochs therefore change39→52 updates, not only the available data. JapaneseVowels
changes7→9 updates per epoch and PenDigits188→235. Same-epoch and same-update
controls are different questions; neither has yet isolated the cause of the
generalization gap. The owner’s proposed TRAIN-only refit controls remain apt.

## Prospective allocation policy

`public_benchmarks/selection_policy.py` implements a policy for **fresh,
prescribed multi-seed DEV replications**, not a replacement historical selection.
At least three specified unique seeds must complete for every candidate. Reject
TEST-bearing screens, missing/extra seeds, differing manifests/programs and
changed within-candidate configurations. Average per-example losses across seeds
to evaluate the single-seed model family, rather than average probabilities and
silently select a deployment ensemble.

For candidate c versus the lowest mean-DEV-NLL candidate b, compute paired
example differences averaged over seeds, and paired seed differences averaged
over examples. Let s_example and s_seed be their descriptive standard errors.
With multiplier λ frozen before the new screens, admit candidates satisfying

    mean_NLL(c) − mean_NLL(b) <= λ max(s_example, s_seed).

Select the lowest measured-estimated screening cost among admitted candidates;
parameters, mean NLL and name break ties. Default λ=1. This empirical rule
preserves a clearly better expensive candidate when its difference exceeds the
tolerance. It is an allocation heuristic: the max is neither a variance bound
nor a proof of noninferiority. Crossed seeds/examples, reused epoch selection and
overlapping split replications preclude interpreting it as a confidence level.
Freeze independent TRAIN-only splits and the rule before new numerical runs;
do not choose λ from the opened TEST outcomes.

## Allocation and retained architecture

1. Keep the existing curie primate DEV round and its bounded single-job owner.
   Tied p32/D2/pool4 reached R²0.740898 on indy_20170131_02, with44,826 parameters;
   the public AEGRU entry is0.71 **averaged across all sessions**. This is a
   promising exploratory session result, not a leaderboard win. Freeze shared
   settings using validation across sessions, save all weights/data/source
   identities, then confirm all six sessions and independent seeds. Record that
   this session’s TEST has already been consulted.
2. Compare single-stream and K-stream primate inference on identical weights.
   K=4 is already in the owner’s round2; charge approximately four core
   executions plus aggregation and state residency, rather than attach K1 work
   to the K4 score. Fit cost, selected updates, all-key discovery and state bytes
   remain separate. 44,826 FP32 parameters alone occupy179,304 bytes, before
   buffers/runtime state, so no smaller-than-AEGRU footprint claim is supported.
3. Complement the AWS owner’s TRAIN-only independent-split and refit controls
   with multi-seed selection; do not launch another broad public TEST-driven
   architecture sweep. Hold the architecture fixed first to isolate selection
   and budget transfer. Existing streaming replay fits retain their slots.
4. The completed90M p64/D4 native language model reaches T2561.857306bpc,
   showing continued learned scale progress. Its restricted E64/1M test and
   estimated241.219TF fitting work do not establish public text8 supremacy.
   Preserve the saved dense/count references and the trained sparse-evaluator
   parity/quality/resource gates.

No architectural substitution is made. Temporal delays/races, sparse private
state updates, separate keys/values, learned messages and counterfactual local
message credit remain. Full future-write replay, useful long credit,
silence-aware real-event coverage, trained serving costs and physical energy
remain open. SHD owner/admission work is retained without duplicate trainers.

Seventeen standard-library tests cover score corruption/IDs, paired alignment,
small-sample uncertainty, immutable lineage, missing seeds, cost denominators,
source/data changes and prospective selection. Numerical imports and model
execution are absent. Physical curie reservation is not visible in this Docker
workspace; AWS’s normal host reservation remains held by its existing owner.
No waiting trainer, new job or coordinator is started here.

Primary comparison source, checked4October:
[NeuroBench leaderboard](https://github.com/NeuroBench/neurobench/blob/main/leaderboard.rst).
