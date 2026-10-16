"""Curie chain v16 (supersedes v15 while one-pass L4 runs; that job is awaited unchanged). Resumable stages.

Route (16 Oct 03:40): depth x data first. One pass at 4M: L8 beats L2 (flagship 5.8414 vs 5.8648, prequential 5.8095 vs
5.8362) and the gap grows through the pass; compiled blocks run 1.57x faster with loss agreement 9.5e-7.
1. L4 (running) → compiled-driver smoke → 16M one-pass L8 then L2 (compiled) → part-whole levels on L8 → publication.
2. v15 stages unchanged: concept contracts → plan-B L2 on 1024 lanes → type-closed (P9e) → concept + tree → concept →
   recursive heads → concept decision → publications → P6d → keyed 16.8M → B10.
"""
import json
from pathlib import Path
import continue_market_depth_v2 as market
from continue_curie_deep_recovery_v8 import completed, run_or_await
import continue_curie_deep_language as original
from continue_curie_deep_recovery import stable_host
import continue_curie_deep_recovery_v11 as v11
import continue_curie_deep_recovery_v12 as v12
import continue_curie_deep_recovery_v14 as v14
import continue_curie_deep_recovery_v15 as v15

ROOT = Path(__file__).resolve().parents[1]
SMOKE = 'curie_r1_deep_keyed_stream_L8_compile_smoke_20261016_v1'
L8_16, L2_16 = 'curie_r1_deep_keyed_stream_L8_16M_s0_20261016_v1', 'curie_r1_deep_keyed_stream_L2_16M_s0_20261016_v1'


def smoke():
    log = ROOT/f'experiments/queue/logs/{SMOKE}.log'
    lines = [l for l in log.read_text().splitlines() if l.startswith('{"smoke_steps"')] if log.exists() else []
    if not lines:
        stable_host(); original.run(SMOKE, 3600)
        lines = [l for l in log.read_text().splitlines() if l.startswith('{"smoke_steps"')]
    s = json.loads(lines[-1]); print(json.dumps({'compile_smoke': s}), flush=True)
    if s['max_rss_kb'] > 2_800_000 or s['s_per_step'] > 3.5: raise RuntimeError('Compiled L8 smoke exceeds the admitted cost envelope')


def depth_data():
    try: run_or_await(v14.L4, 36000)
    except Exception as error: print(json.dumps({'depth_ladder': v14.L4, 'failed': repr(error)}), flush=True)
    try:
        smoke()
        for tag, timeout in ((L8_16, 64800), (L2_16, 43200)):
            try: run_or_await(tag, timeout)
            except Exception as error: print(json.dumps({'depth_16m': tag, 'failed': repr(error)}), flush=True)
    except Exception as error: print(json.dumps({'compile_smoke': 'failed', 'error': repr(error)}), flush=True)
    try: run_or_await(v14.PW8, 64800)
    except Exception as error: print(json.dumps({'depth_ladder': v14.PW8, 'failed': repr(error)}), flush=True)
    done = [t for t in (v11.L2, v14.L4, v11.L8, L8_16, L2_16, v14.PW8) if completed(t)]
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
