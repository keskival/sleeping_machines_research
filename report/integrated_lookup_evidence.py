"""Completed full-fit token lookup engineering evidence."""
import hashlib
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]


def blocks():
    path=ROOT/'experiments/results/token_language/aws_integrated_p24_64k_lookup_decision_20261012_v1.json'
    if not path.exists():return []
    d=json.loads(path.read_text());assert d['status']=='completed'
    for name,digest in {**d['source_sha256'],**d['input_sha256']}.items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,name
    rows=[[r['arm'],f"{r['selected_dev_nll']:.6f}",f"{r['final_dev_nll']:.6f}",
           f"{r['training_wall_s']:.3f}",f"{r['engine_wall_s']:.3f}",f"{r['process_wall_s']:.3f}",
           f"{r['peak_rss_kib']/1024:.1f}"] for r in d['rows']]
    result = [('p','Integrated R1 execution: batching vocabulary gathers inside the existing credit windows removes repeated dense embedding-gradient accumulation. All temporal events, hard races, separate keys/values, persistent numerical state, state-gradient boundaries, counterfactual suffix teaching and optimizer updates are retained.'),
        ('table',(['Lookup','Selected NLL','Final NLL','Train seconds','Engine seconds','Process seconds','RSS MiB'],rows,[24,28,28,29,29,29,24])),
        ('p',f"Selected quality gain {d['quality_gain']:+.6f} nats; ordinary/batched training wall {d['ordinary_over_batched_training_wall']:.3f}x. Registered development gate (quality retained and at least5% training-wall saving): {'passed' if d['development_gate_passed'] else 'failed'}."),
        ('p',d['scope']),
        ('p','Numerical integration gate: three AdamW updates with carried state, repeated token IDs, EOS resets and two credit windows; maximum forward/state error9.54e-7, gradient error1.20e-7, parameter error1.06e-7, optimizer error7.46e-9; RNG states equal. Evidence aws_integrated_lookup_parity_20261012_v1 and '+path.stem+'.')]
    precision=ROOT/'experiments/results/token_language/aws_integrated_p24_64k_double_decision_20261012_v2.json'
    if precision.exists():
        q=json.loads(precision.read_text());assert q['status']=='completed'
        for name,digest in {**q['source_sha256'],**q['input_sha256']}.items():
            assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,name
        pr=[]
        for r in q['rows']:
            pr.append(['6',r['arm'],f"{r['selected_dev_nll']:.6f}",f"{r['training_wall_s']:.3f}",f"{r['engine_wall_s']:.3f}"])
        confirmations=[]
        for seed in (7,8):
            paths=[ROOT/'experiments/results/token_language'/f'aws_integrated_p24_64k_double_{arm}_s{seed}_20261012_v1.json' for arm in ('ordinary','batched')]
            if not all(p.exists() for p in paths):continue
            records=[json.loads(p.read_text()) for p in paths]
            assert all(r['status']=='completed' for r in records)
            for r in records:
                for name,digest in r['source_sha256'].items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,name
            assert records[0]['identity']==records[1]['identity']
            assert {k:v for k,v in records[0]['args'].items() if k!='tag'}=={k:v for k,v in records[1]['args'].items() if k!='tag'}
            assert [r['step'] for r in records[0]['curve']]==[r['step'] for r in records[1]['curve']]
            delta=max(abs(x['dev_nll']-y['dev_nll']) for x,y in zip(records[0]['curve'],records[1]['curve']))
            confirmations.append(f"seed{seed}: curve error{delta:.3g}, ordinary/batched training{records[0]['train_wall_s_total']/records[1]['train_wall_s_total']:.3f}x")
            for arm,r in zip(('ordinary','batched'),records):
                pr.append([str(seed),arm,f"{min(x['dev_nll'] for x in r['curve']):.6f}",f"{r['train_wall_s_total']:.3f}",f"{r['wall_s']:.3f}"])
        result.extend([('p','Precision-stabilized integrated execution: float64 retains the same model and learning operations. The32-update diagnosis agrees within2.3e-14NLL. Full64K measurements below include all factual/alternative learning and optimizer work.'),
            ('table',(['Seed','Lookup','Selected NLL','Training seconds','Engine seconds'],pr,[18,30,38,40,40])),
            ('p',f"Seed6 full DEV curve maximum error{q['max_dev_curve_error']:.3g}; ordinary/batched training{q['ordinary_over_batched_training_wall']:.3f}x. Against saved float32 ordinary: quality gain{q['versus_fp32']['quality_gain']:+.6f}, training ratio{q['versus_fp32']['ordinary_over_batched_training_wall']:.3f}x. Precision comparison changes arithmetic precision; same-precision comparison isolates execution. Full-curve parity criterion1e-8. Useful retained-quality/work gate: {'passed' if q['useful_quality_work_gate_passed'] else 'failed; original float32 retained'}."),
            ('p','Confirmations: '+('; '.join(confirmations) if confirmations else 'no further paired confirmation admitted after the retained-quality loss. An already-started seed7 ordinary fit completed; unstarted confirmation queues retired.'))])
    paths=[ROOT/'experiments/results/token_language'/f'aws_integrated_p24_unique_stream_decision_20261012_s{seed}_v1.json' for seed in (6,7,8)]
    paths.append(ROOT/'experiments/results/token_language/aws_integrated_p24_unique_512k_decision_20261012_s6_v1.json')
    for unique in paths:
        if not unique.exists():continue
        u=json.loads(unique.read_text());assert u['status']=='completed'
        for name,digest in {**u['source_sha256'],**u['input_sha256']}.items():
            p=ROOT/name
            # Checkpoints are host-local; the completed evaluator pins their hashes.
            if p.suffix=='.pt' and not p.exists():continue
            assert hashlib.sha256(p.read_bytes()).hexdigest()==digest,name
        ur=[[r['arm'],str(r['unique_tokens']),str(r['fitting_targets']),
             f"{r['initial_dev_nll']:.6f}",f"{r['selected_dev_nll']:.6f}",
             f"{r['context_gain']:.6f}",f"{r['training_wall_s']:.3f}"] for r in u['rows']]
        targets=u['rows'][0]['fitting_targets']; updates=u['rows'][0]['updates']
        result.extend([('p',f'Distinct natural experience at fixed learning-event budget: {targets}fitting targets/{updates}updates. Same original float32 integrated model; numeric state persists across the longer stream.'),
            ('table',(['Exposure','Unique tokens','Fit targets','Initial NLL','Selected NLL','Context gain','Train seconds'],ur,[25,28,28,30,30,30,30])),
            ('p',f"Distinct-experience quality gain{u['quality_gain']:+.6f}; initial frequency-prior gain{u['initial_prior_gain']:+.6f}; change in post-initialization learning gain{u['learning_gain_change']:+.6f}; change in contextual gain{u['context_gain_change']:+.6f}."),
            ('p',u['scope']),('p','Evidence '+unique.stem+'.')])
    return result
