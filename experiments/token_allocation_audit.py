"""Stdlib decision gate for a paired 2x2 token architecture experiment."""
import json
import math
from pathlib import Path


def audit(records, margin=.02):
    required={'private_local','private_future','shared_local','shared_future'}
    if set(records)!=required:
        raise ValueError('Complete predeclared 2x2 required')
    baseline=None
    rows={}
    varying={'tag','future_every','tie_pools'}
    for name,r in records.items():
        if r.get('status')!='completed':raise ValueError('Completed fits required')
        args=r['args'];settings={k:v for k,v in args.items() if k not in varying}
        identity=r['identity']
        bound=(settings,identity)
        if baseline is None:baseline=bound
        elif bound!=baseline:raise ValueError('Unmatched data/source/settings')
        shared=name.startswith('shared');future=name.endswith('future')
        if args['tie_pools']!=shared or args['future_every']!=int(future):
            raise ValueError('Treatment labels do not match actual policy')
        curve=r['curve']
        if curve[0]['step']!=0 or curve[-1]['step']!=args['steps']:
            raise ValueError('Full learning trajectory required')
        schedule=[x['step'] for x in curve]
        if rows and schedule!=next(iter(rows.values()))['evaluation_steps']:
            raise ValueError('Equal development selection opportunities required')
        for item in curve:
            if any(not math.isfinite(item[k]) for k in ('train_nll','dev_nll')):
                raise ValueError('Nonfinite quality trajectory')
        best=min(x['dev_nll'] for x in curve[1:])
        if not math.isclose(best,r['best_dev_nll'],rel_tol=1e-12):
            raise ValueError('Saved best does not match trajectory')
        if r['presentations_total']<=0 or r['train_wall_s_total']<=0:
            raise ValueError('Positive measured fitting work required')
        rows[name]=dict(best_dev_nll=best,final_dev_nll=curve[-1]['dev_nll'],
            initial_dev_nll=curve[0]['dev_nll'],learning_gain=curve[0]['dev_nll']-best,
            train_nll=curve[-1]['train_nll'],presentations=r['presentations_total'],
            fitting_wall_s=r['train_wall_s_total'],parameters=r['parameters'],
            core_parameters=r['core_parameters'],readout_parameters=r['readout_parameters'],
            peak_rss_kib=r['max_rss_kb'],evaluation_steps=schedule,
            receiver_usage=curve[-1].get('routing'),
            receiver_gradient_norms=curve[-1].get('receiver_gradient_norms'),
            route_entropy=curve[-1].get('route_entropy'),
            future_write_teacher=curve[-1].get('future_write_teacher'),
            whole_stage_wall_s=r.get('stage_wall_s'),
            memory_effective_rank=curve[-1].get('memory_effective_rank'))
    if len({x['presentations'] for x in rows.values()})!=1:
        raise ValueError('Equal fitting presentations required')
    gains={kind:rows[kind+'_local']['best_dev_nll']-rows[kind+'_future']['best_dev_nll']
           for kind in ('private','shared')}
    learning=[name for name,r in rows.items() if r['learning_gain']>=margin]
    selected=min(learning,key=lambda name:rows[name]['best_dev_nll']) if learning else None
    return dict(status='completed',evidence='single-seed development architecture allocation diagnostic',
        rows=rows,future_credit_gains=gains,
        sharing_credit_interaction=gains['shared']-gains['private'],
        provisional_member=selected,learning_margin=margin,
        future_credit_promotable_at_this_seed={k:v>=margin for k,v in gains.items()},
        scaling_ready=False,remaining_gates=['independent-seed confirmation',
           'complete fitting FLOP accounting','larger-data useful-capacity comparison'],
        scope='Matched data/presentations/selection cadence; future credit adds replay cost. '
              'CPU wall/RSS are measured operational resources, not asynchronous hardware efficiency. '
              'Occupancy/rank are diagnostics, not proof of predictive specialization. Public validation untouched.')
