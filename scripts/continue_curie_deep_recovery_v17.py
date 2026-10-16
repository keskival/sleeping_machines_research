"""Curie chain v17 (supersedes v16 while 16M L8 runs; awaited unchanged). Resumable stages.

v16 lost one-pass L4 at update ~600 (run_safe cgroup-headroom STOP caused by a side process, 16 Oct 03:55); its exact
resume checkpoint (update 512) is resumed after the 16M pair.

Route (16 Oct 03:40): depth x data first. One pass at 4M: L8 beats L2 (flagship 5.8414 vs 5.8648, prequential 5.8095 vs
5.8362) and the gap grows through the pass; compiled blocks run 1.57x faster with loss agreement 9.5e-7.
1. 16M one-pass L8 (running, compiled) then L2 → L4 exact resume → part-whole levels on L8 → publication.
2. v15 stages unchanged: concept contracts → plan-B L2 on 1024 lanes → type-closed (P9e) → concept + tree → concept →
   recursive heads → concept decision → publications → P6d → keyed 16.8M → B10.
"""
import json
import continue_market_depth_v2 as market
from continue_curie_deep_recovery_v8 import completed, run_or_await
import continue_curie_deep_language as original
from continue_curie_deep_recovery import stable_host
import continue_curie_deep_recovery_v11 as v11
import continue_curie_deep_recovery_v12 as v12
import continue_curie_deep_recovery_v14 as v14
import continue_curie_deep_recovery_v15 as v15
import continue_curie_deep_recovery_v16 as v16


def depth_data():
    try:
        v16.smoke()
        for tag, timeout in ((v16.L8_16, 64800), (v16.L2_16, 43200)):
            try: run_or_await(tag, timeout)
            except Exception as error: print(json.dumps({'depth_16m': tag, 'failed': repr(error)}), flush=True)
    except Exception as error: print(json.dumps({'compile_smoke': 'failed', 'error': repr(error)}), flush=True)
    for tag, timeout in ((v14.L4, 36000), (v14.PW8, 64800)):
        try: run_or_await(tag, timeout)
        except Exception as error: print(json.dumps({'depth_ladder': tag, 'failed': repr(error)}), flush=True)
    done = [t for t in (v11.L2, v14.L4, v11.L8, v16.L8_16, v16.L2_16, v14.PW8) if completed(t)]
    extra = [f'experiments/results/diagnostics/{v14.BENCH}.json'] if v12.diag(v14.BENCH) else []
    market.publish([f'experiments/results/token_language/{t}.json' for t in done] + extra, 'R1 depth x data: one-pass deep keyed ladder at 4M and 16M fresh tokens, part-whole levels on L8')


def main():
    stages = [('depth_data', depth_data), ('contracts', v12.contracts), ('base_l1024', lambda: run_or_await(v12.BASE, 28800)),
              ('type_closed', v15.type_closed), ('concept_tree', v12.concept_tree), ('concept_no_tree', lambda: run_or_await(v12.CN, 64800)),
              ('recursive', lambda: run_or_await(v15.REC, 64800)), ('concept_decision', v12.concept_decision), ('publish_new_arms', v15.publish_new_arms),
              ('nofuture', lambda: run_or_await(v11.NOFUT, 64800)),
              ('decision_p6d', lambda: v11.decide('curie_part_whole_decision_p6d_20261015_v2', [v11.L2, v11.FIT, v11.NOFUT, v11.L8], 'R1 part-whole hierarchy: level-matched future targets ablation (P6d)')),
              ('keyed_scaling', v15.v5.keyed_scaling)]
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
