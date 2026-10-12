"""Current completed native future-taught quality/work, source-bound."""
import hashlib
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def blocks():
    records=[]
    for seed in range(3):
        path=ROOT/'experiments/results/credit'/f'aws_native_future_update_reuse_20261011_s{seed}_decision.json'
        if not path.exists():continue
        decision=json.loads(path.read_text());assert decision['status']=='completed'
        for source,digest in decision['source_sha256'].items():assert hashlib.sha256((ROOT/source).read_bytes()).hexdigest()==digest
        records.append((seed,decision))
    if not records:
        return [('p','Current integrated learning study aws_native_future_update_reuse_20261011_s0: full native actor updates atdepths2/4/8 against matched fixedAdam;180-parameter future-taught group scales reuse the next actor gradient with zero model replay. Smoke matches retained replay actor/controller parameters exactly. All1400TaxiTRAIN sequences cropped32events,threepasses/130200actor targets per arm/depth,128DEV sequences. Prediction-quality/work results pending; protocol experiments/credit/NATIVE_FUTURE_UPDATE_PLAN.md.')]
    rows=[]
    for seed,record in records:
        for r in record['rows']:
            rows.append([str(seed),str(r['depth']),f"{r['final_fixed_nll']:.6f}",f"{r['final_learned_nll']:.6f}",f"{r['final_quality_gain']:+.6f}",str(r['first_matched_quality_epoch'] or '-'),f"{r['fixed_over_learned_matched_work']:.3f}x" if r['fixed_over_learned_matched_work'] else '-'])
    return [('p','Whole-native future-taught learning: actor and controller learn from the same actual forward/backward passes. The controller chooses bounded scales for four Adam parameter groups from observed loss and gradient norms; the next actual loss gradient teaches its previous issued decision through the exact local chain rule.'),
            ('table',(['Seed','Depth','Adam NLL','Learned NLL','Quality gain','Match pass','Work ratio'],rows,[13,17,29,29,29,18,28])),
            ('p','Protocol: all1400TaxiTRAIN sequences cropped32events,threepasses/130200actor targets per arm/depth;128DEV sequences. Same native initialization/order/update counts, CPUfloat64,d16,modes8. Lower NLL is better; gain is Adam minus learned. Match pass is first scheduled learned pass reaching fixedAdam final quality. Work ratio is Adam total fit+allvalidation divided by learned cumulative fit to match+allvalidation. Ratios>1 favor learned work; smoke/failed admission are separately paid overhead. All controller/Adam work is included in fit wall; supported FLOPs exclude unsupported operations. Dense writes and state reset across sequences; development results, no publicTEST/hard-route/Transformer supremacy claim.'),
            ('p','Evidence: '+', '.join(f'aws_native_future_update_reuse_20261011_s{s}_decision' for s,_ in records)+'.')]


def transfer_blocks():
    records=[]
    for seed in [3,4,5]:
        path=ROOT/'experiments/results/credit'/f'aws_native_update_transfer_20261012_s{seed}_decision.json'
        if not path.exists():continue
        r=json.loads(path.read_text());assert r['status']=='completed'
        for source,digest in r['source_sha256'].items():assert hashlib.sha256((ROOT/source).read_bytes()).hexdigest()==digest
        records.append(r)
    if not records:return [('p','Acquired-policy reuse study aws_native_update_transfer_20261012_s3 is admitted: frozen seed0 controller vs fixedAdam and TRAIN-acquired constant group scales, fresh actor seed3, full1400TaxiTRAIN/128DEV, depths2/4/8. Fine checkpoints resolve incremental work to quality; inherited acquisition remains charged. Results pending. Protocol experiments/credit/NATIVE_UPDATE_TRANSFER_PLAN.md.')]
    rows=[]
    for record in records:
        for r in record['rows']:
            ratio=r['baseline_over_matched_work']
            parent_path=next(iter(record['input_sha256']))
            raw=json.loads((ROOT/parent_path).read_text())
            static=next(x for x in raw['rows'] if x['depth']==r['depth'] and x['arm']=='static_acquired_groups')
            frozen=next(x for x in raw['rows'] if x['depth']==r['depth'] and x['arm']=='frozen_acquired_policy')
            match=next((h for h in frozen['history'] if h['nll']<=static['history'][-1]['nll']),None)
            strong=(static['fit_wall_s']+static['evaluation_wall_s'])/(match['fit_wall_s']+frozen['evaluation_wall_s']) if match else None
            rows.append([str(record['seed']),str(r['depth']),f"{r['final_quality_gain']:+.6f}",f"{r['quality_gain_vs_static']:+.6f}",f"{ratio:.3f}x" if ratio else '-',f"{strong:.3f}x" if strong else '-',str(r['conservative_break_even_fit_count'] or '-')])
    return [('p','Acquired native update transfer: source seed0 controller frozen before fresh-model fitting. No controller optimizer/replay. Comparison with TRAIN-acquired static group scales isolates conditioning from fixed step-size changes.'),
            ('table',(['Seed','Depth','Gain vs Adam','Gain vs static','Adam/work','Static/work','Source break-even fits'],rows,[10,13,28,27,29,29,27])),
            ('p','Same1400TaxiTRAIN/128DEV,32events,threepasses,d16,modes8; lower NLL better. Quality gain is reference minus frozen. Work ratio is fixedAdam fit+allvalidation divided by frozen fit to first scheduled quality match+allvalidation. Ratios>1 favor incremental reuse. Static/work separately uses static final quality as the target. Across completed seeds3/4/5 it saves work1/9; eight cases cost1.6–2.7%more. The source acquisition experiment is additionally charged; break-even fits is the calculated repeated-fit count paying that source cost, not measured deployment saving. Other smoke/selection costs remain additional protocol charges. Dense writes/state reset; model-seed transfer on same corpus, not domain/publicTEST/Transformer supremacy.'),
            ('p','Evidence: '+', '.join(r['tag'] for r in records)+'.')]


native_learning_blocks=blocks

def blocks():
    return native_learning_blocks()+transfer_blocks()


def domain_blocks():
    records=[]
    for seed in [6,7,8]:
        path=ROOT/'experiments/results/credit'/f'aws_native_update_domain_20261012_s{seed}_decision.json'
        if not path.exists():continue
        r=json.loads(path.read_text());assert r['status']=='completed'
        for source,digest in r['source_sha256'].items():assert hashlib.sha256((ROOT/source).read_bytes()).hexdigest()==digest
        records.append(r)
    if not records:return [('p','Cross-domain learning transfer is admitted: Taxi seed0 controller frozen on StackOverflow, depths2/4/8,full1400TRAIN/128DEV,threepasses. Compared with fixedAdam and source-acquired static groups; source/target time-density units canonicalized from TRAIN medians. No target controller fit; completed quality/work decision pending. Protocol experiments/credit/NATIVE_UPDATE_DOMAIN_PLAN.md.')]
    rows=[]
    for record in records:
        for r in record['rows']:
            ratio=r['static_over_frozen_matched_work']
            rows.append([str(record['seed']),str(r['depth']),f"{r['final_quality_gain']:+.6f}",f"{r['quality_gain_vs_static']:+.6f}",f"{ratio:.3f}x" if ratio else '-'])
    return [('p','Taxi-to-StackOverflow update-policy transfer, without target policy refitting. Same native family and full-parameter learning; conditioning uses current observed loss/gradient norms and TRAIN-derived density-unit canonicalization.'),
            ('table',(['Seed','Depth','Gain vs Adam','Gain vs static','Static/matched work'],rows,[17,23,38,38,47])),
            ('p','All1400targetTRAIN sequences cropped32events,threepasses,128DEV. Gain is reference minus frozen NLL; lower NLL better. Stronger work ratio is static fit+allvalidation divided by frozen fit to first scheduled static-quality match+allvalidation. Acquisition remains separately charged; incremental reuse ratios do not represent whole-pipeline advantage. Dense writes/sequence reset; no TEST/Transformer claim.'),
            ('p','Evidence: '+', '.join(r['tag'] for r in records)+'.')]


seed_transfer_blocks=blocks

def blocks():
    return seed_transfer_blocks()+domain_blocks()
