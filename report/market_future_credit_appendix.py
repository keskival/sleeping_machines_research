"""Current completed B10 forward-taught credit/data/work results."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]


def load(tag):
    path=ROOT/f'experiments/results/market/{tag}.json'
    result=json.loads(path.read_text());assert result['status']=='completed'
    for field in ('source_sha256','input_sha256'):
        for name,digest in result.get(field,{}).items():
            assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,name
    return result


def pages():
    if not (ROOT/'experiments/results/market/curie_b10_future_credit_confirmation_20261013_v1.json').exists():return []
    d=load('curie_b10_future_credit_evidence_20261013_v1')
    c=load('curie_b10_future_credit_confirmation_20261013_v1')
    action_path=ROOT/'experiments/results/market/curie_b10_action_geometry_20261013_v1_cohort.json'
    action=load('curie_b10_action_geometry_20261013_v1_cohort') if action_path.exists() else None
    quality=[[r['arm'].replace('_',' '),f"{r['nll']:.9f}"] for r in d['quality'] if r['arm']!='learned_refresh']
    quality.insert(0,['Native future credit, three critic fits',f"{c['mean_nll']:.9f} ± {c['sample_sd_nll']:.9f}"])
    work=[[r['arm'].replace('_',' '),f"{r['covered_arithmetic_flops']/1e6:.3f}",f"{r['covered_arithmetic_flops']/512/1e3:.3f}"] for r in d['work512']]
    curve=[[str(r['passes']),f"{r['dev_credit_mse']:.9f}"] for r in d['fixed_data_curve']]
    current = [
      [('h2','Native forward-taught credit improves actual online updates'),
       ('table',(['Learning rule','NLL / event'],quality,[100,100])),
       ('p',f"Win across three critic initializations: mean gain{c['mean_gain']['bptt_carry']:.6f}nats/event over exact eight-event factual BPTT and{c['mean_gain']['own_refresh']:.6f}over matched own-refresh. Every critic beats both controls and TRAIN-mean credit. Credit is taught from forward forecast contrasts; no BP fit labels. Native private memory evolves causally while its weights remain frozen during deployment."),
       ('small','Three independently initialized31,804parameter critics, one shared restored18,164parameter actor,128TRAIN receipts and16384reusedAug25DEV events. Producer8/future8, clip1/Adam.0003; refresh replays eight already observed events under updated weights. SampleSD across critic fits, not actor/data replications. The original signed-native control is for seed19124. New schedule/source acquisition; established32target5.737391headline unchanged. No sealedTEST/public leaderboard/profit claim.')],
      [('h2','Extra work is charged alongside the quality gain'),
       ('table',(['Rule','Whole prefix MFLOPs','kFLOPs / target'],work,[82,47,55])),
       ('p','Executed512DEVtarget/32update probes include127daywarmup events; all saved prefix forecasts reproduce exactly. Native future credit uses more arithmetic than exact factual BP. Original actor fitting and8192source forward-teacher targets, acquisition/critic learning and diagnostic trials are additional.'),
       ('small','Same units/denominator within columns. Torch mathematical formula and NumPy trace-kernel work; nonlinear functions/Pythonmath/random and uninstrumented native-history scalar operations remain separate. No16384work extrapolation, physical traffic or ASIC energy claim. User13October priority permits more compute for quality or data efficiency; matched compute is one evidence class, not a promotion requirement.')],
      [('h2','More learning compute at fixed credit data'),
       ('table',(['Native passes over128receipts','Frozen DEV credit MSE'],curve,[100,100])),
       ('p','Four passes improve fixed-actor credit error11.60%below zero, versus5.35%after one pass, using the same128TRAIN receipts and no new teacher worlds. Actual-update NLL worsens5.756599to5.756772: retain one-pass actor utility rather than the MSE winner. With64receipts, two passes reach0.001119539MSE versus0.001133650with128receipts/onepass; later passes overfit. This is an exploratory additional-credit-data point with shared actor pretraining and multiple DEV checkpoints, not half-data learning for the whole model.'),
       ('p','Random-ray teachers buy75%/50%fewer training worlds but give DEV MSE0.001343988/0.001246334 versus zero0.001197747; neither retained. Conditional gradient identity passes the native interface audit, but sampling variance defeats this128frame acquisition. Keep useful fitting compute and improve the actual-update teaching objective.'),
       ('p','Actual-utility follow-up: functional clip/Adam, refreshed producer state and future direct-parameter paths pass numerical parity. Offline2048DEV fits at critic learning rates0.0001/0.001 give5.703270648/5.703272409 versus frozen5.703270448. Causal online critic learning on16384DEVevents gives5.756685231 versus retained frozen5.756599087: this scheme loses0.000086144nats/event.1024future blocks teach only after factual forecasting; teacher loss parity1.43e-11. Additional16384teacher replay targets and8192refresh targets are paid. Retain original three-critic quality result; next priority optimizer-aligned action geometry and temporal learner credit.'),
       ('small','Online result/checkpoint were saved before a final-print KeyError (queue exit1). Completed rows/source hashes/numerical parity verified in curie_b10_actual_utility_20261013_v1_validation.json; no training rerun or evidence overwrite.'),
       ('small','Exact two-layer/addressed state interface:88coordinates,352eligibility floats/four directions; E^Tfuturecotangent matches full-history minus future-direct derivative2.22e-16. Delayed forward teacher activations remain native hard-race computation. Globalclip/Adam boundaries retained. Theory STATE_REFRESH_AND_ACTUAL_UPDATE_CREDIT.md derives the full optimizer/direct-parameter/refreshed-state teaching derivative.')]
    ]

    if action:
        current.append([('h2','Optimizer-conditioned credit: paired native development'),
          ('table',(['Critic initialization','Parent NLL','Conditioned NLL','Gain'],[[str(r['seed']),f"{r['parent_nll']:.9f}",f"{r['nll']:.9f}",f"{r['gain']:+.9f}"] for r in action['rows']],[39,48,54,44])),
          ('p',f"Three independently initialized parent critics, same restored actor and128additional TRAIN utility frames, frozen16384DEV forecasts. Mean conditioned NLL{action['mean_nll']:.9f}; mean paired gain{action['mean_parent_gain']:+.9f}nats/event. Report each paired outcome; the original three-critic recipe remains retained while this conditioning rule is developed."),
          ('p','Full refreshed-future parameter demand transported through AWS actual clip/Adam matches functional action derivatives≤2.22e-18. Sparse diagonal-plus-rank-one parameter-step metric matches four actual-proposal JVPs≤3.62e-18 on three native learner versions:64clock support entries,18,164actor parameters. Sparse metric avoids materializing an18,164by4tangent; full scalar clipping/moment contractions remain. Parameter-step metric is not forecast curvature.'),
          ('small','Reused DEV, one shared actor/data stream;36,820parameter conditioned critics versus31,804parents, additional2048teacher replay targets/128backwards plus1024source refresh targets perfit. Original forward-teacher acquisition and actor prefit also charged. Seed19124 used functional JVP metric;19129/19130 used contracted sparse formula. No TEST/profit/ASICenergy/full-work claim. Established32targetheadline unchanged.')])
    delayed_path=ROOT/'experiments/results/market/curie_b10_delayed_action_20261013_v1_analysis.json'
    if delayed_path.exists():
        delayed=load('curie_b10_delayed_action_20261013_v1_analysis')
        alias=load('curie_b10_optimizer_context_alias_20261013_v1')
        current.append([('h2','Delayed learner-consequence credit and input identifiability'),
          ('table',(['Teacher','TRAIN utility events','DEV prefix NLL'],[[label,str(row['train_utility_events']),f"{row['nll']:.9f}"] for label,row in zip(['Finite preference','Gain weighted','Gain weighted, more data'],delayed['curve'])],[70,52,62])),
          ('p',f"Native teacher recomputes future own gradients, Adam moments and native decisions in factual/alternative worlds. Gain weighting aligns the leading finite-probe signal with expected utility rather than probability of improvement. More data improves the2048DEVprefix, but unchanged frozen16384DEV scores{delayed['full_nll']:.9f}versus parent{delayed['full_parent_nll']:.9f}: loss{-delayed['full_gain']:.9f}. Retain the parent recipe. First64source receipts and full DEVprefix reproduce exactly."),
          ('p','Corrected eight same-horizon probes pair future race randomness: mean absolute optimizer-memory utility effect0.000003075nats/event, maximum0.000009337, first-future forecast difference0. No preference reversals. The earlier unpaired0.000003044 attribution is replaced; original result files remain. Immediate/delayed preference differences are not attributed to moments.'),
          ('p','Concrete input repair: jointly permuted per-mode clock moments preserve pooled19context≤6.51e-19 yet future demands change up to0.072669and reverse signs. Fixed-cut diagnostic, not observed acquisition alias frequency. Next use typed native mode/route response receipts instead of hiding associations inside pooled statistics.'),
          ('small','One actor/critic, reusedTRAIN/DEV; source8192events/256additional alternative worlds,6144scoredtargets+4096refresh targets;2048own-gradient targets are within scored targets. Originalactor/parentacquisition, native/optimizer/teacherfitting work additional. No TEST/scaling/wholework/hardwareclaim. Existing three-critic quality and32target marketheadline unchanged.')])
    receipt_path=ROOT/'experiments/results/market/curie_b10_native_receipt_20261013_v2_fit.json'
    if receipt_path.exists():
        current.append([('h2','Typed native optimizer receipts: integrated development'),
          ('p','A native residual critic receives32typed mode events with clock parameters, issued gradients, individual Adam moments, signed responses and observed state. It retains hard races and private persistent banks. Total63,608parameters,31,804trainable; bounded residual and frozen retained parent.'),
          ('p','Zero-residual contract reproduces48forecasts and factual race randomness exactly. Formerly aliased optimizer cuts produce distinct input tokens. Paired finite-consequence training on64receipts/2048TRAINevents gives2048DEV NLL5.703278097 versus parent5.703270448: loss0.000007650nats/event. Retain parent; no full continuation or investor quality headline.'),
          ('small','One actor/critic and reused DEV; additional32receipt events per decision and source counterfactual worlds are paid. No sealed TEST or ASIC efficiency claim. Earlier actual-utility, optimizer-conditioned and delayed-fit helpers evaluated an unused anchor that consumed extra race draws: their executed forecast scores remain, but small differences are not isolated under identical random trajectories. The new null contract and evaluator repair this pairing.')])
    signed_path=ROOT/'experiments/results/market/curie_b10_native_antithetic_20261013_v1_analysis.json'
    if signed_path.exists():
        current.append([('h2','Signed future-consequence teaching: optimizer calibration'),
          ('p','Paired positive/negative bounded actions teach the native residual critic from24future targets with recomputed later gradients, moments and decisions. Same native architecture/private race banks; raw contrast estimates a ball-smoothed decision gradient rather than fitting backpropagated targets.'),
          ('p','64TRAINreceipts/2048events score2048DEV5.703285203 vs parent5.703270448: loss0.000014756nats/event. Retain parent. Zero-action/actual-first-update smoke errors0.39of64critic gradients exceed norm1, mean3.859927/max36.468283; prior teacher mean0.097392. Clipping changes raw-estimator expectation; calibration is a concrete next diagnosis, not established cause of the loss.'),
          ('small','One reused actor/data stream/initialization.128extra counterfactual worlds3072score targets+2048refresh targets,1024own-gradient targets within scoring; actor/parent acquisition and32extra native receipt events perdecision additionally paid. No full continuation, sealedTEST, investor quality or ASIC efficiency claim. Exact portable fitted tensors and source-bound analysis retained.')])
    return current
