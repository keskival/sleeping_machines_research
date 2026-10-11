# R1 subsequent forward losses teach a connected update function

Registered11October2026 before fitting/scoring. Motivation: fullblock conditional
curvature passes freshcredit calibration (80.3%lowerMSE vsdiag), while prefix/
boundary foresight predictors do not. Test actualonline learning of the update
function using observed future feedback, not forecast labels/backprop imitation.
Native source remains pretrained2layer keyedGPT2 model; newforwardlexical z64
learns within episode. Temporal/key state persistent; no externalarchitecture.

Connected two-layer64wide tanh function outputs step eta, damping lambda and
momentum beta from observednative hiddenstate, fastz/moment, currentgradient/
covariancediagonal, revealedloss andepisodeposition. Initialheads0 => eta.01,
lambda1,beta.9; eta bounded .001.. .1, lambda .1..10,beta0.. .99. Geometricstep is clipped to Euclideannorm.08 (the initialAdam.01step norm atdimension64), with the clipping derivative included. This changes the declared update and bounds saturation risk. Uniformembedding rank8 blockbasis;
currentforwardprobability supplies PSDblock covariance with exactdiagonal.

    m'=beta*m+(1-beta)*g; r'=beta*r
    z'=z-eta*(Hhat+lambda*mean(diagHhat)*I)^-1*m'/(1-r').

Keep currentz/m/r andHhat/g detached for the issued one-stepmeta objective.
Retain z' graph/creditparameter version. Nextprediction is made with z'; only
AFTER its target isrevealed compute g_next=E^T p_next-E_target. Backpropagate
that observed g_next through retained z' to train the updatefunction, then
issue the nextupdate. This derivative is exact for the declared one-step
futureloss holding incomingstate/forwardnative snapshot fixed. It does not
include all futuremomenteffects/BPTT. EOS resets fast/native episode state and
pendinggraph; globalcredit parameters/moments remain within that episode.

TRAINpretraining1024targets after64nativecontext at sourceboundary+720896,
onlyconnected updatefunction trained; factualfast learner adapts but is reset
before new scoring episodes. Seed99710, metaAdam.001, finalcheckpointonly.
Prequential private development4independent episodes512targets/64context,
sourceboundary+786432+1024*i. Same source andinitialfaststate for allarms:
standard lexicalAdam.01; fixed geometric eta.01/lambda1/beta.9; frozen pretrained
updatefunction; online updatefunction initialized from same pretrainedweights
andmetaAdamstate, trained from nextforwardfeedback. Primary online-vs-frozen
meanNLL gain with paired4episode10000bootstrap/95%; initialfixed/Adam secondary.
No heldoutselection. Both primary arms inherit fullpretrainingcost. BaselineAdam
has no extra learnedfunctionpretraining; no equaltrainingclaim against it.

Smoke64TRAIN/2episodes32targets on disjoint source+792576 offsets, no selection.
Full1024/4x512 after one-stepgradient/causality/EOS contracts and numericsmoke.
Whole nativeexecution shared across episodearms but perarm fastprediction,
geometry, function/solver/metaoptimizer work charged; nativecommon separately
allocatedidentically. Streaming oneevent perstep, no T x vocabulary storage.
Report inference/learning stages, selectedkeyactivity/capacity/exposures and
allbasis/pretraining/scoring costs. No publicDEV/TEST, fullnativelearning,
newTransformer/LSTM or fullBPTT/depth/sparsehardwareefficiencyclaim.
OneCPUthread/run_safe/RSS1.5GB/MemAvailable9GiB/2GiBcgroupreserve.
Measure smoke before choosing full timeout; longfit tmux, exactlyonehostjob.
