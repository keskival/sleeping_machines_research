"""Completed source-bound local reciprocal teaching/update contracts."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
TITLE='Appendix. B1 reciprocal forward teaching and native actor-update closure'
RESULTS=['aws_reciprocal_tracking_20261010T2157Z','aws_reciprocal_cut_contract_20261010T2200Z',
         'aws_reciprocal_projection_20261010T2204Z']


def packet_blocks():
    tags=['aws_native_cached_jvp_20261011_v2','aws_native_cached_linearization_20261011_v1','aws_native_packet_actor_update_20261011_v1','aws_packet_actor_step_20261011_v1','aws_packet_actor_teaching_step_20261011_v1']
    if not all((ROOT/'experiments/results/credit'/f'{t}.json').exists() for t in tags):return []
    records=[]
    for tag in tags:
        r=json.loads((ROOT/'experiments/results/credit'/f'{tag}.json').read_text());assert r['status']=='completed'
        for source,sha in r['source_sha256'].items():assert hashlib.sha256((ROOT/source).read_bytes()).hexdigest()==sha
        records.append(r)
    jvp,vjp,actor,ordinary,teaching=records
    rows=[[str(f['events']),str(f['input_dimension']),f"{f['whole_speedup']:.2f}x",f"{b['whole_reverse_speedup']:.2f}x",str(b['cached_floats'])] for f,b in zip(jvp['rows'],vjp['rows'])]
    return [('p','A shared native primal packet now supports exact input tangents for forward teaching and exact input cotangents for producer credit, without constructing the dense cut Jacobian. Temporal decay/rotation, gated writes, key-independent local memory and normalization derivatives are retained. This is an economical exact local reference, not a learned or hard-route credit result.'),
            ('table',(['Events','Cut dimension','Forward probe speed','Reverse speed','Cached floats'],rows,[21,29,35,31,37])),
            ('p','Speed is measured whole capture plus16 batched directions/cotangents versus native batched autodiff (forward reference disables unnecessary reverse-graph recording), warmed5-repeat CPU medians. At32 events the cache contains7,744 floats; a dense1,024-dimensional Jacobian would contain1,048,576. Cache counts exclude shared model, input/dt and query buffers; autograd memory is unmeasured.'),
            ('p',f"Real Taxi TRAIN actor integration: four updates on one four-event prefix, all parameter gradients agree within {max(h['gradient_error'] for h in actor['history']):.2e} normalized error, Adam within {max(h['adam_error'] for h in actor['history']):.2e}. One tracked forward capture retains the layer/head parameter graph; its packet cotangent reconnects embedding/gap producers. No extra forward probe sweep. Gate wall {actor['wall_s']:.3f}s, peak RSS {actor['max_rss_kb']/1024:.1f}MiB; graph and packet work are charged."),
            ('p','Scope: dense-write native layer, fixed elapsed times, input derivatives only. Four actor updates score12 TRAIN target presentations per arm; no DEV/TEST quality endpoint, full fitting FLOPs, per-target fitting work or serving-work measurement. Parameter derivatives remain exact autodiff, with its graph retained. No whole-fit/depth/sparse-learning efficiency claim. Stale parameter versions are rejected. Next: use this exact packet reference for measured compact-credit teaching/audit integration.'),
            ('p','Complete actor-step measurement (aws_packet_actor_step_20261011_v1): reference/packet 2.432/3.063ms at B1,d8,n4,L4 and7.030/9.136ms at B4,d32,n16,L32. Retaining the parameter graph plus packet makes ordinary learning26–30% slower; native autodiff remains the ordinary-fit reference. Extra teaching-query gains are separate.'),
            ('p','Complete actor update plus16 forward teaching responses: '+', '.join(f"B{r['batch']},L{r['events']}: {r['reference_step_s']*1000:.3f}/{r['packet_step_s']*1000:.3f}ms reference/packet ({r['whole_step_speedup']:.2f}x)" for r in teaching['rows'])+'. Response parity <=3.56e-15 and parameter parity <=5.56e-17. Single-layer real Taxi TRAIN, five warmed timings; critic fitting/use is excluded. This is the label-production boundary, not whole-fit learning efficiency.'),
            ('p','Shared-packet evidence: '+', '.join(tags)+'.')]


def pages():
    records=[]
    for tag in RESULTS:
        path=ROOT/'experiments/results/credit'/f'{tag}.json'
        if not path.exists():return []
        r=json.loads(path.read_text());assert r['status']=='completed'
        for source,sha in r['source_sha256'].items():assert hashlib.sha256((ROOT/source).read_bytes()).hexdigest()==sha
        records.append(r)
    tracking,scalar,projected=records
    reduction=100*(1-tracking['mean_tracked_relative_error']/tracking['mean_frozen_relative_error'])
    rows=[]
    for label,r in [('Scalar teaching',scalar),('Packet projection',projected)]:
        rows.append([label,str(r['native_actor_updates']),str(r['forward_probes']),str(r['local_teaching_updates']),
                     f"{max(x['normalized_gradient_error'] for x in r['history']):.2e}"])
    return [[('h1',TITLE),
             ('p','Forward tangent packets now teach a backward transport map, which carries credit into native actor input parameters. '
              'Four successive parameter versions match the exact-gradient reference, including Adam state and parameter updates. '
              'The local teaching rule uses forward responses and outer products; it has no third learned teacher.'),
             ('table',(['Implementation','Actor steps','Probes','Teaching updates','Gradient error'],rows,[39,22,22,31,39])),
             ('p',f"Rank-one packet projection reduces teaching updates from {scalar['local_teaching_updates']:,} to {projected['local_teaching_updates']:,} with unchanged forward probes. "
              f"Maximum parameter difference {max(x['parameter_error'] for x in projected['history']):.2e}; maximum Adam-state difference {max(x['adam_state_error'] for x in projected['history']):.2e}. "
              f"In a separate 12-version tracking diagnostic, refreshing two input directions per version reduces mean relative squared transport error by {reduction:.1f}% versus a frozen map."),
             ('p',f"Accounting: projection gate forward teaching {projected['forward_teaching_s']:.3f}s; both actor/reference arms and audits {projected['two_arm_actor_and_audit_s']:.3f}s; "
              f"whole gate {projected['wall_s']:.3f}s, peak RSS {projected['max_rss_kb']/1024:.1f}MiB. Native forward loss is {projected['history'][0]['loss']:.6f} to {projected['history'][-1]['loss']:.6f} "
              'on the repeated TRAIN prefix. Dense cut transport stores 1,024 values; full fitting FLOPs and inference work are unmeasured.'),
             ('p','Scope: one four-event Taxi TRAIN prefix, one temporal layer. Cut credit trains embedding/gap producers; layer/head derivatives remain exact local operations. '
              'Every cut direction is refreshed in the actor-parity gate. This is a reciprocal-update contract, not a DEV-quality, sparse-learning, efficiency or benchmark win. '
              'Next: context-conditioned transport, cheaper probes and task-valued acquisition under moving weights.'),
             ('p','Evidence: '+', '.join(RESULTS)+'. Sources, data hashes, timings and all gate inputs are retained in completed JSON records.')] + packet_blocks()]
