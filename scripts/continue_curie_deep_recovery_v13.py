"""Curie chain v13 (supersedes v12 at the end of the part-whole v2 fit, which is awaited unchanged). Resumable stages.

Strongest route (15 Oct 22:05): scale depth and data together with the composition that makes depth pay (plan-B blocks:
pre-norm residual, per-layer keyed race reads, full-lane credit; AWS 1M ladder L2 5.957 / L4 5.899 / L8 5.874).
1. Part-whole v2 fit (running) → decision v2 → publication.
2. Depth x data ladder at 4M fresh tokens, one pass: L8 then L4 (L2 done: 5.865) → publication.
3. Concept hierarchy (P7): contracts → L2 on 1024 lanes → concept + tree → concept without tree → curves + decision.
4. Part-whole no-future ablation (P6d), keyed 16.8M, B10 eight-layer stage.
"""
import json
import continue_market_depth_v2 as market
import continue_curie_deep_recovery_v5 as v5
from continue_curie_deep_recovery_v8 import completed, run_or_await
import continue_curie_deep_recovery_v11 as v11
import continue_curie_deep_recovery_v12 as v12

L4 = 'curie_r1_deep_keyed_stream_L4_4M_s0_20261015_v1'


def depth_ladder():
    for tag, timeout in ((v11.L8, 50400), (L4, 36000)):
        try: run_or_await(tag, timeout)
        except Exception as error: print(json.dumps({'depth_ladder': tag, 'failed': repr(error)}), flush=True)
    done = [t for t in (v11.L2, L4, v11.L8) if completed(t)]
    market.publish([f'experiments/results/token_language/{t}.json' for t in done], 'R1 depth x data: one-pass deep keyed ladder at 4M fresh tokens')


def main():
    stages = [('part_whole_v2', v12.part_whole_v2), ('depth_ladder_4m', depth_ladder), ('contracts', v12.contracts),
              ('base_l1024', lambda: run_or_await(v12.BASE, 28800)), ('concept_tree', v12.concept_tree),
              ('concept_no_tree', lambda: run_or_await(v12.CN, 64800)), ('concept_decision', v12.concept_decision),
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
        failure = market.ROOT / f'experiments/results/market/{market.FIT}_failure.json' if hasattr(market, 'ROOT') else None
        if failure:
            with failure.open('x') as f: json.dump(dict(status='failed', battle='B10', error=repr(error), smoke=market.SMOKE, fit=market.FIT), f, indent=2); f.write('\n')
    print('Queue complete', flush=True)


if __name__ == '__main__':
    main()
