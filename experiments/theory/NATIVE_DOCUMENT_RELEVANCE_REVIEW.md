# Formal review: native memory relevance across documents

16October2026,R1. Review follows the producer-credit repair and source-bound4Mcontinuation. No new public result or ablation.

## Findings and selection

1. **Confirmed omission,repair selected:**NativeCauseBank.query returns same_document,but indexed_cause_nll never consumes it. The earlier addressed-memory member learned a cross-document bias−4.39(roughly80foldrate discount);raw persistence had performed worse than reset,and learned addressing restored useful persistence. This is prior evidence for providing the feature,not a prediction of its coefficient or gain in the new member. Preserve memory and let relevance learn;do not hard reset or force a negative coefficient.
2. **Producer-credit rule reviewed:**the local replay adds the last-input/residual contribution into the historical composer/mix;own-write eligibility supplies native write-parameter credit. Detaching native write coefficients in the local input VJP avoids double credit. Cached child views,route and timestamps stay fixed. Changing historical producer parameters makes this an approximate online teacher;it does not implement fullnetworkRTRL. Existing independent local derivative/forward/recovery contracts pass.
3. **Conditional probability and causality reviewed:**background/concept word laws retain full vocabulary support;repeat/continuation causes use actual stored token/successor and original age. append matures the previous successor using the current revealed input;the current record's unknown successor stays−1. Posterior inboxes are generated only after scoring a revealed outcome and enter its actual next input. Class partitions use knownTRAINstatistics,so TRAINloss is fitting evidence,not an untouched online claim.
4. **Architecture hypotheses retained,not silently promoted to defects:**query root rings/access policies,detached earlier child producers,missing recursive event/silence teaching and autonomous rate policies limit current integration. Fixed2querybindings are not proof of semantic parts,and output responsibility is not abstraction. These require a derived integration and task evidence;nonzero gradients or a finite-budget failure do not settle them.

## Minimal normalized repair

For candidate j,let c_j=1 when it is from another observed document. Add beta_repeat*c_j to its repeat log-rate and beta_continue*c_j to continuation. beta=(0,0) exactly recovers the original forward law and old parameter gradients. Two independent biases allow different behavior for copying and continuation. All memories/candidate sets remain available,including useful cross-document matches. Membership is known from already observed end-of-document tokens and does not depend on the predicted token.

For one cause group,write R for total rate and A for target-compatible weighted rate,R_c/A_c for their cross-document contributions. For logloss L=log R−log A,

    dL/dbeta = R_c/R − A_c/A.

The model discounts that group across documents if allocated prior rate exceeds posterior responsibility for correct predictions,and can increase it in the opposite case. Both signs are valid. The gate's two batches show differing continuation gradients;do not extrapolate them into a learned global discount or quality improvement.

Scaling positive rates preserves the existing exactly normalized categorical word/cause law. No vocabulary re-tokenization,latent target,history reset,external architecture or fresh baseline is introduced.

## Completed contracts and acquisition gate

`document_boundary_causes.py` wraps the original normalized law through an explicit process-local head view;the running source file stays untouched. A two-scalar parameter is added to the existing optimizer group after old model/Adam state loads. On own-checkpoint restore it is installed before loading the upgraded state,so parameter order and Adam references remain stable.

Two contracts pass:bitwise zero-biasforward and oldgradientparity;full small-vocabulary normalization plus document-bias gradient finite differences(error<1e−8). On a private77824checkpoint,the64TRAINtarget/twoupdategate plus32discardedrecoveryaudit passes:finite norms1.753/1.971,nonzero newbiasgradients,bitwise fulllearner recovery,2003176KiBRSS/23.949s including contracts/load. No DEV/TEST or quality result.

Checkpoint-migrate the current producer-credit learner with all old weights/Adam/receiver state/querybindings/producerrecords/index/inboxes/RNG/cursor preserved. Newbiases start zero with fresh Adam state. Keep4Mcumulative target,data order,batch/globalclip and fixed262K/1M/4Mselection/finalflagship protocol. Preserve sourcecheckpoint/log and record any discarded unsavedtail. A safe checkpoint boundary is preferred over restarting a prefix. Old imported sources stay frozen;only the explicit wrapper/new driver implement the repair. Fullprediction/learning/traffic work remains charged.
