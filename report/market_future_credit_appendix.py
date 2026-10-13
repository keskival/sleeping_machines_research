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
    quality=[[r['arm'].replace('_',' '),f"{r['nll']:.9f}"] for r in d['quality'] if r['arm']!='learned_refresh']
    quality.insert(0,['Native future credit, three critic fits',f"{c['mean_nll']:.9f} ± {c['sample_sd_nll']:.9f}"])
    work=[[r['arm'].replace('_',' '),f"{r['covered_arithmetic_flops']/1e6:.3f}",f"{r['covered_arithmetic_flops']/512/1e3:.3f}"] for r in d['work512']]
    curve=[[str(r['passes']),f"{r['dev_credit_mse']:.9f}"] for r in d['fixed_data_curve']]
    return [
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
       ('small','Exact two-layer/addressed state interface:88coordinates,352eligibility floats/four directions; E^Tfuturecotangent matches full-history minus future-direct derivative2.22e-16. Delayed forward teacher activations remain native hard-race computation. Globalclip/Adam boundaries retained. Theory STATE_REFRESH_AND_ACTUAL_UPDATE_CREDIT.md derives the full optimizer/direct-parameter/refreshed-state teaching derivative.')]
    ]
