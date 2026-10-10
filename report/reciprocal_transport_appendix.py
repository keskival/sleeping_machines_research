"""Completed source-bound local reciprocal teaching/update contracts."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
TITLE='Appendix. B1 reciprocal forward teaching and native actor-update closure'
RESULTS=['aws_reciprocal_tracking_20261010T2157Z','aws_reciprocal_cut_contract_20261010T2200Z',
         'aws_reciprocal_projection_20261010T2204Z']


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
             ('p','Evidence: '+', '.join(RESULTS)+'. Sources, data hashes, timings and all gate inputs are retained in completed JSON records.')]]
