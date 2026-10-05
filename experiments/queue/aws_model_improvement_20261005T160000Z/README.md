# AWS model improvement program — 5 October 2026

Physical host ip-172-31-47-132; all numerical work uses unique one-job queues,
run_safe.sh, inherited ordinary host reservation, one thread and per-slot
locks. Three CPU slots maximum; 8 GiB available-memory floor.

## Concrete hypotheses and comparisons

1. **Write bandwidth.** The current sparse state update may inadequately
   refresh recent context. Corrected k-write v2 contracts test k1 parity,
   eager/compiled outputs and every parameter gradient, padding, inactive
   state and write counts. Pool4 k1 versus k2 keeps temporal races, private
   addressed memory, key/value separation and route credit; only earliest
   selected writes change. One winner still delivers. k2 retains sparsity
   (two of four slots), with all extra computation charged in the corrected
   fitting trace. The existing inference estimate is NOT established for
   this new policy. These equal-pass arms diagnose a mechanism; they are not
   an equal-compute advantage claim. The existing promotion gate is >=.02 BPC
   against matched k1; one seed is exploratory, not confirmation.
2. **Parameter regularization.** AdamW weight decay .01/.1 versus the saved
   Adam p64D4pool2 four-pass arm. Temporal computation, sparse addressed
   writes and route credit are retained. A three-window smoke must complete
   first with 1.5x RSS margin. Charge the actual optimizer work; a baseline
   nominal budget is not automatically the new actual fit budget.
3. **Capacity exposure.** Existing p32D4 sampled-credit one-pass queues: tied
   pool8, tied pool32, untied pool32. Shared proposal maps train on all
   examples while keys/state remain private. This tests exposure and useful
   capacity, with saved smaller-pool anchors. New matching three-window
   smokes precede each full fit. These compare combined sharing/credit
   choices; isolated causal claims need matched controls. The existing
   driver does not save full per-slot occupancy, so quality alone cannot
   establish successful use of every slot.

## Admission and publication

Slot1 resumes the original p96 90M fit from saved model/Adam/schedule/RNG
(window18000,147456000 presentations), with unchanged model/settings.
Slot2 runs both existing 90M C controls followed by the two D Transformer
controls. Slot3 contracts, two k-write smokes, weight-decay smoke, the bounded
k-write comparison, two decay arms and three capacity smoke/fit pairs.

Failed/missing/source-mismatched/oversized predecessors block descendants.
Unrelated arms and controls continue. Terminal failures remain recorded in
worker_recovery.status.json; no failed run is relabelled as passed. Results
and final weights publish on completion under the Git publication lock.

Original queues and frozen manifests are retained. transition.json records
unknown discarded pre-transition work <=4095999 presentations; the saved
checkpoint is immutable. prior_coordinator.py preserves the previous runner.
Pending rows supply no quality prediction. Fits remain CPU simulations, not
measurements of an asynchronous hardware speed or energy advantage.
