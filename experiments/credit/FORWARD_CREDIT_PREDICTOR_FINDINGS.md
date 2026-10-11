# R1 forward-taught connected critic: first development iteration

Completed32-prefix/10pass corrected and uncorrected54,081parameter predictors,
source native2layer model fixed; eight new exact-scored private prefixes.
Corrected training approximate-label MSE2.682e-7 vszero4.035e-7 (33.5% lower).
Fresh exact MSE7.571e-7 vszero6.456e-7 (17.3% higher); corrected-over-zero
MSE gain -1.115e-7, paired95%[-1.643e-7,-5.587e-8]. Predictive gate fails.
Uncorrected fresh MSE7.567e-7; corrected-vs-uncorrected interval crosseszero.
Training mean cosine .538, fresh -.112; policies learn training alignment
but not transfer. This is a restricted predictor development result, not a
model-family/depth or future-task verdict. No actor/native parameter training.

Acquisition126.219s/105.768GF,95actual worlds/4560actual updates plus99,840
transported alternative updates. Both critic fits5.160s/2.891GF,320updates
perarm. Shared exact scoring48.970s/300.822GF,472worlds/22,656updates.
Whole180.526s,54,081parameters each. Sourcepretraining/contracts/smokes additional.
Inputs onlyprefix; critics predict48-target contrasts before futurelabels.
Teacher after-future calibration and prefix prediction are distinct tasks.

Diagnosis uses paid records, no new future acquisition.32independent returns
for a high-capacity connected function are a plausible data bottleneck, not a
proven cause. Before adding capacity or changing objectives, registered data
scaling reuses32records and adds96; same initialization/model/lr/10passes,
16NEW exact-scored prefixes. See FORWARD_CREDIT_SCALING_PROTOCOL.md.
No validation/checkpoint selection. This stage tests more data, not largerdepth.
If it fails, examine conditional future variance and representation/target
sufficiency; do not call post-observation transport accuracy prefix foresight.


## Completed128-prefix data stage

Same54,081parameters/initialization/learningrule/10passes,32paid records reused
and96newacquisitions.16new exact-scored prefixes. CorrectedMSE6.004e-7 versus
zero5.440e-7 (10.4% worse); improvement -5.642e-8, paired95%[-1.237e-7,+2.050e-8].
No confirmed predictive benefit. Different32/128scoring cohorts mean the
numerical error ratios are unpaired; do not claim a learning curve win.
New acquisition374.428s/317.926GF plus inherited126.219s/105.768GF; fitting
20.012s/11.563GF; scoring101.884s/660.937GF. Whole new stage496.566s/558.9MiB.
Further scaling of this prefixprior is deferred pending a representation/target
change. Next registeredboundarycritic observes8targets before predicting40
remaining learningeffects. Reuse128TRAIN/16DEV paid full48contrasts; source
andnativearchitecture unchanged. ReusedDEV marks development, not confirmation.
