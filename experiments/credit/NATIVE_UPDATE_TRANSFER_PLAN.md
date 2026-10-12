# B1 acquired native update policy: transfer and work to quality

12 October2026. Three seeds of whole-native future-taught update training improve final DEV NLL at every depth2/4/8 (nine comparisons). Each arm sees130200 actor targets; learned controller consumes existing next actor gradients, with no additional model replay. Coarse scheduled matching still costs7–10% more wall time. The next decision is whether the acquired180-parameter policy is reusable and whether conditioning adds value beyond fixed group scales.

## Fixed before fresh-seed fitting

Source policy: seed0 final checkpoint only, never best-seed selection. No actor parameters, optimizer moments or controller optimizer state transfer to the new actor. Same Taxi corpus; transfer is between actor initializations, not datasets. All1400TRAIN sequences cropped32events; first128DEV sequences; three passes,batch8,depth2/4/8,d16,modes8. Fresh model seed3. Controller stays frozen and uses current observed loss, phase and actual clipped gradient norms. It costs its forward evaluation but no optimizer/replay. Stable source hashes/checkpoint/data digests are checked.

Three arms share initial actor weights, data order, Adam moments/rules and update counts: fixedAdam at0.001; static group scales from the source policy's final TRAIN step; frozen conditional acquired function. Static scales are TRAIN-acquired constants, not hand-tuned DEV values. This distinguishes useful conditioning from fixed learning-rate changes. Full supported fitting operations and measured wall include all group-scale/controller work; score after batches44/88/132/175 each pass, with all validation costs retained. The reference target is fixedAdam's final three-pass DEV loss, declared before fitting. Record first scheduled crossing, final NLL, time/mark losses and whole fitting/validation wall.

## Acquisition and reuse accounting

Keep inherited acquisition separate from incremental reuse. For repeated matched-quality fits, total learned work is acquisition A plus sum of measured reuse work; fixed work is the matching sum of reference work. A amortized work claim requires that inequality to hold. Given per-fit savings S>0, the measured break-even reuse count is floor(A/S)+1, before smoke and additional selection costs. No saving gives no finite break-even. A includes the entire completed seed0 acquisition experiment in the conservative pipeline view; its per-depth learner-only acquisition is a second explicitly named boundary, not a replacement for the conservative total.

## Pipeline and scope

New-driver smoke completes0.935s/403660KiB, finite native updates for all three arms. Full run aws_native_update_transfer_20261012_s3 admitted slot3/1800s/RSS1.5GB/oneCPU through run_safe. No TEST, external dense architecture or new proxy grid. Dense native writes/reset between sequences remain driver scope. If frozen policy improves quality and measured incremental work at least at two depths, confirm unchanged on seeds4/5; report acquired and static comparisons separately. If static matches or beats conditional, reuse the cheaper static rule and develop conditional features against that stronger own-model reference. No family verdict from this development stage.
