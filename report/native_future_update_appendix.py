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
