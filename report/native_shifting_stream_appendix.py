"""Completed same-target nonstationary learning record."""
import hashlib
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def blocks():
    records=[]
    for seed in [9,10,11]:
        path=ROOT/'experiments/results/credit'/f'aws_native_shifting_stream_20261012_s{seed}_decision.json'
        if not path.exists():continue
        record=json.loads(path.read_text());assert record['status']=='completed'
        for source,digest in record['source_sha256'].items():assert hashlib.sha256((ROOT/source).read_bytes()).hexdigest()==digest
        records.append(record)
    if not records:return [('p','Long changing-stream learning admitted:12800distinct Retweet TRAIN sequences/396800scored targets,onepass,depth4/8. Original,timingx8,markrelabeling and gap-dependent mark regimes; no change labels given to learners. FixedAdam/static acquired/frozen/online acquired controls, pre-feedback scoring, carried actor/optimizer/controller state. Quality/adaptation/work results pending; acquisition charged. Protocol experiments/credit/NATIVE_SHIFTING_STREAM_PLAN.md.')]
    rows=[]
    for record in records:
        for r in record['rows']:
            for c in r['comparisons']:
                rows.append([str(record['seed']),str(r['depth']),c['reference'],f"{c['cumulative_quality_gain']:+.6f}",f"{c['first_postchange_window_gain']:+.6f}",f"{c['reference_over_online_fit_work']:.3f}x"])
    return [('p','Changing-stream learning: acquired update function continues learning from the next actual gradient, with no model replay. Score before feedback; same unique targets/order for every control. Larger positive cumulative/post-change gains favor online; work ratios>1 favor online fitting cost.'),
            ('table',(['Seed','Depth','Control','Cumulative gain','Post-change gain','Control/online work'],rows,[12,15,41,32,32,31])),
            ('p','Onepass/12800distinct RetweetTRAIN sequences cropped32events,396800targets;256earlier sequences calibrate vocabulary/time. Four constructed distribution/dependency regimes, not natural chronological drift. Actor/Adam/controller state carries across changes; forward state resets across independent sequences. First32updates after each of three changes define post-change windows. Source acquisition additionally charged; exact actor backprop retained. Dense writes; no TEST/Transformer/asymptotic scaling claim.'),
            ('p','Evidence: '+', '.join(r['tag'] for r in records)+'.')]
