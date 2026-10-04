# Current AWS work

**P0-4 Mackey-Glass third batch unblocked (check 4 Oct, 23:55 UTC, Docker review host):** all 175 pinned sources in
`queue/aws_neurobench_admission_20261004T064000Z/manifest.json` match current main (sha256 recomputed). An actual import trace of
`experiments/mackey_glass_native.py` loads 14 sleeping_machines modules and **not** `expected_reception.py`, the file
behind the earlier needs_review. Admit `aws_mg_tau17_r1_from20` (repeats 20–29, unchanged pre-declared config) into the
next free slot after the P0-1/P0-6/P0-3 order. Re-verify the pins at admission. If they still fail, record which file and
whether the driver imports it. Arithmetic: 20/30 mean is 14.372; beating LSTM 13.37 needs a 10-repeat mean below about 11.37
(the first 10 averaged 13.60), so report the completed 30-repeat result plainly either way.

**Priority update (user, 4 October 21:00 UTC): tuned dense references (P0-6, [TUNED_BASELINES.md](TUNED_BASELINES.md)).**
Slot order: 90M rev. 4 p64 (P0-1) → the ten `queue/aws_tuned_ref_10M_*_20261004T210000Z.txt` → six-session primate (P0-3)
→ p96 rev. 4 → four `aws_tuned_ref_90M_*` → `aws_language_10M_r1_p64d4_4pass_linear_s7_20261004T210000Z` → `aws_language_10M_p64d4_pool8_sampled_4pass_s6_20261004T233500Z`
(P1-2/P2: quality beside the flat-traffic capacity cost model; 10M RSS for pool 8 is unmeasured, so start at the 10M caps and watch). One job per queue,
via run_safe.sh. Select tuned arms by validation, never by test. Wall time is not evidence (simulated hardware).

**Priority (user, 4 October 19:30): 90M matched-compute runs** `queue/aws_language_90M_r4_p64d4_4pass_linear_20261004T193000Z.txt`
then `..._r4_p96d4_4pass_...` (AWS_NATIVE_LANGUAGE_90M.md, revision 4), as soon as slots free.

**NeuroBench primate reaching protocol run (4 October 05:35 UTC):** six one-session queues
`queue/aws_primate_r1_*_20261004T053500Z.txt`; see [AWS_NEUROBENCH.md](AWS_NEUROBENCH.md). Admit as slots free up. Also three Mackey-Glass
tau 17 queues `queue/aws_mg_tau17_r1_from{0,10,20}_20261004T061500Z.txt` (same doc).

**Queued by the user's request (3 October): native language at 90M characters, revision 2 (10:30 UTC, with route
credit).** Protocol and admission: [AWS_NATIVE_LANGUAGE_90M.md](AWS_NATIVE_LANGUAGE_90M.md). Contracts and pilots
`queue/aws_language_90M_contracts_20261003T103000Z.txt`, then three one-job arms `aws_language_90M_r2_*_20261003T103000Z`
(p32/d4, p32/d8, p64/d4; all `--route-credit linear`, compiled, one pass, checkpointed), plus `aws_language_90M_r2_p32d4_pool4_linear_20261003T110000Z`
(pool 4, 2.343 at 10M). Admit the p64/d4 arm first (2.184 at 10M, .012 from the one-pass LSTM). Revision-1 queues
(`*_20261003T063000Z`) are superseded and must not be admitted.
Then the 90M pool-8 arm `aws_language_90M_r3_p32d4_pool8_linear_20261003T205500Z` (revision 3), then two 10M multi-pass supremacy arms `aws_language_10M_6pass_*_20261003T193000Z` (same doc, last section).

The prioritized active integrated experiment is
`gym/plans/aws_capacity_exposure_20261002T072141Z/manifest.json`:
common-source-seed native H2/d8/depth8, occupied sources64, private/shared
rules, matched8 fit queries/source/pass, four passes and seeds6/7/8.
See [the capacity protocol](AWS_CAPACITY_EXPOSURE.md). Three guarded CPU slots
on this AWS host only;4GiB RSS/6GiB VMS/job and8GiB available-memory floor.
Do not start a second worker or modify its frozen sources.

As of this review,15/18 stages are complete and published; three pilot fits are
active. Completed results are95.02%/98.34% private-rule seeds6/7 and99.90%
shared-rule seed6 on development. Keep remaining cells pending. Total data is
larger than the16-source references; this is not an iso-data supremacy result.
The new capacity-summary supervisor waits for full completion, validates every
result, and publishes a common-unit quality/work inventory automatically.

Completed predecessors remain preserved:

- Native fresh-data timing and shared+common-source confirmation both passed
  their prespecified gates. The independent chain and replication summary are
  complete; do not restart them.
- Crossed rule/source-seed pilots isolate the combined effect. Common seed
  explains most quality gain; shared rules alone did not improve mean accuracy.
  See [the full factorial findings](AWS_RULE_SEED_FINDINGS_20261002.md).
- The causal two-trace timing reference completed1024/1024; it uses known
  generator constants and is a diagnostic, not a learned competitive control.
- Banknote remains11/12 final comparisons. CatBoost seed8 was stopped after
  an anomalous3128s with no completed candidates; its failure is preserved.
  Logistic has lower reserved-test NLL than our variant. Do not rerun a completed
  tag or report the full banknote gate as complete.

Other hosts own current count/statistic-retrieval, late-projection and value-
credit language comparisons. Do not duplicate their queued variants. Review
LOCAL_HANDOFF and THEORY before assigning a new architectural comparison.
Large Transformer controls remain AWS work under their original protocols and
resource requirements; completed90M controls are historical evidence, not a
request to blindly rerun old queues. No additional newly assigned AWS battery
was found in this pull. Preserve invalid-protocol quarantine and pending cells.
