"""Measured admission plan for the heavier paired AWS bank stage; no execution."""
import math
try:
    from .aws_private_bank_comparison import compare
except ImportError:
    from aws_private_bank_comparison import compare


def plan(small,large,mem_available_kb):
    evidence=compare(small,large)
    rows=evidence['rows']
    if any(r['admitted_train_tokens']>8192 for r in rows):raise ValueError('Bounded smoke pair required')
    if not any(r['learning_gain']>=.02 for r in rows) or any(r['context_gain']<=0 for r in rows):
        raise ValueError('Smoke learning/context gate failed')
    jobs=[]
    for arm,row in zip((small,large),rows):
        raw=arm[0];args=raw['args']
        fixed=dict(payload=24,depth=2,heads=2,chunk=64,credit_window=16,lanes=8,eval_lanes=8,dev_tokens=2048)
        if any(args[k]!=v for k,v in fixed.items()):raise ValueError('Declared fixed P24/D2/H2/batch64/credit16 recipe required')
        if args['future_site']!='uniform' or args['future_every']!=1 or args['future_score_positions']!=4:
            raise ValueError('Actual uniform-site K4 learning required')
        for point in raw['curve'][1:]:
            teacher=point.get('future_write_teacher')
            if not teacher or not math.isfinite(point['gradient_norm']):raise ValueError('Finite actual-future-credit logs required')
            if not math.isfinite(teacher['mean_suffix_loss_delta']) or any(not math.isfinite(q) or q<=0 for q in teacher['proposal_probability']):
                raise ValueError('Invalid future-write consequences or proposal support')
            if len(teacher['winner'])!=args['lanes'] or len(teacher['alternative'])!=args['lanes']:
                raise ValueError('Teacher lane count mismatch')
            if any(w==a for w,a in zip(teacher['winner'],teacher['alternative'])):
                raise ValueError('Teacher did not test losing alternatives')
        throughput=row['train_targets_per_s'];rss=row['max_rss_kb']
        if not math.isfinite(throughput) or throughput<=0 or not math.isfinite(rss) or rss<=0:
            raise ValueError('Measured ordinary throughput/RSS required')
        cap=max(800000,math.ceil(rss*1.5/65536)*65536)
        timeout=math.ceil(2*131056/throughput+120)
        if cap>4000000 or timeout>6000:raise ValueError('Measured workload needs revised bounded reservation')
        if mem_available_kb-cap<8*1024*1024:raise ValueError('Available-memory reserve insufficient')
        settings=dict(args)
        for key in ('tag','resume','stop_after_step'):settings.pop(key,None)
        settings.update(train_tokens=65536,steps=256,eval_every=64,chunk=64,credit_window=16,lanes=8,eval_lanes=8,dev_tokens=2048)
        for seed in (6,7):
            jobs.append(dict(pool=row['pool'],seed=seed,settings=dict(settings,seed=seed),
                timeout_s=timeout,rss_kb=cap,vms_kb=5000000,minimum_available_mb=8192))
    return dict(status='prepared',jobs=jobs,smoke_comparison=evidence,
        admitted=False,scope='Paired U4/U16 seeds6/7,64K GPT-2 TRAIN tokens,two passes,131056 fitted targets each. Resources derived from ordinary smoke measurements. This function launches no jobs; source/data binding and current host/slot checks remain required at execution.')
