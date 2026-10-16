"""Curie chain v15 (supersedes v14 while one-pass L8 runs; awaited unchanged). Resumable stages.

Route (15 Oct): depth must scale with an additive identity path everywhere.
1. Depth x data: L8 (running) → compile benchmark → L4 → part-whole levels on L8 → publication (v14 stage).
2. Concept contracts → plan-B L2 on 1024-token lanes (base for every 1024-lane arm).
3. Type-closed additive event stack: smoke (cost gate) → fit (P9e).
4. Concept hierarchy + tree read → without tree (P7) → recursive heads (P9a-b).
5. Concept decision, publication of the type-closed/recursive records, P6d, keyed 16.8M, B10.
"""
import json
import continue_market_depth_v2 as market
import continue_curie_deep_recovery_v5 as v5
from continue_curie_deep_recovery_v8 import completed, run_or_await
import continue_curie_deep_language as original
from continue_curie_deep_recovery import stable_host
import continue_curie_deep_recovery_v11 as v11
import continue_curie_deep_recovery_v12 as v12
import continue_curie_deep_recovery_v14 as v14

TC_SMOKE, TC = 'curie_r1_type_closed_4M_l1024_smoke_20261015_v1', 'curie_r1_type_closed_4M_l1024_s0_20261015_v1'
REC = 'curie_r1_recursive_4M_l1024_s0_20261015_v1'


def type_closed():
    if not (v11.result(TC_SMOKE) or {}).get('status') == 'smoke_completed':
        stable_host(); original.run(TC_SMOKE, 7200)
    smoke = v11.result(TC_SMOKE); print(json.dumps({'type_closed_smoke': {k: smoke[k] for k in ('s_per_step', 'max_rss_kb')}}), flush=True)
    if smoke['max_rss_kb'] > 2_800_000 or smoke['s_per_step'] > 12: raise RuntimeError('Type-closed smoke exceeds the admitted cost envelope')
    run_or_await(TC, 64800)


def publish_new_arms():
    paths = [f'experiments/results/token_language/{t}.json' for t in (TC_SMOKE, TC, REC) if v11.result(t)]
    if paths: market.publish(paths, 'R1 type-closed additive stack (P9e) and recursive-likelihood arm (P9a-b) records')


def main():
    stages = [('part_whole_v2', lambda: None if v12.diag('curie_part_whole_decision_20261015_v2') else v12.part_whole_v2()),
              ('depth_ladder_4m', v14.depth_ladder), ('contracts', v12.contracts), ('base_l1024', lambda: run_or_await(v12.BASE, 28800)),
              ('type_closed', type_closed), ('concept_tree', v12.concept_tree), ('concept_no_tree', lambda: run_or_await(v12.CN, 64800)),
              ('recursive', lambda: run_or_await(REC, 64800)), ('concept_decision', v12.concept_decision), ('publish_new_arms', publish_new_arms),
              ('nofuture', lambda: run_or_await(v11.NOFUT, 64800)),
              ('decision_p6d', lambda: v11.decide('curie_part_whole_decision_p6d_20261015_v2', [v11.L2, v11.FIT, v11.NOFUT, v11.L8], 'R1 part-whole hierarchy: level-matched future targets ablation (P6d)')),
              ('keyed_scaling', v5.keyed_scaling)]
    for name, stage in stages:
        try: stage()
        except Exception as error: print(json.dumps({name: 'failed', 'error': repr(error)}), flush=True)
    try:
        market.market_stage()
    except Exception as error:
        print(json.dumps({'market_stage': 'failed', 'error': repr(error)}), flush=True)
    print('Queue complete', flush=True)


if __name__ == '__main__':
    main()
