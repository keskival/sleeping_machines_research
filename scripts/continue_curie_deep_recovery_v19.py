"""Curie chain v19 (supersedes v18 before any v18 stage started; 16M L8 is awaited unchanged). Resumable stages.

Route (16 Oct 12:30, user: achieve scaling supremacy and prove it): published PTB battle, the 16M L2 partner (P5d) with
the exact chunk-64 kernel (2x faster compiled), the depth x data analysis, then the native frontier extended to 64M
fresh tokens before older research arms.
1. 16M L8 (running) → PTB smoke → PTB d256 L8 fit → PTB validation scoring → publish (v18.ptb).
2. 16M L2 (chunk 64) → depth x data analysis (P5d) → publish.
3. 64M one-pass L8 (chunk 64, compiled): smoke → run → analysis → publish.
4. L4 4M exact resume → part-whole on L8 → publish; v15 stages unchanged.
"""
import json, subprocess, sys
from pathlib import Path
import continue_market_depth_v2 as market
from continue_curie_deep_recovery_v8 import completed, run_or_await
import continue_curie_deep_language as original
from continue_curie_deep_recovery import stable_host
import continue_curie_deep_recovery_v11 as v11
import continue_curie_deep_recovery_v12 as v12
import continue_curie_deep_recovery_v14 as v14
import continue_curie_deep_recovery_v15 as v15
import continue_curie_deep_recovery_v16 as v16
import continue_curie_deep_recovery_v18 as v18

ROOT = Path(__file__).resolve().parents[1]
L8_64_SMOKE, L8_64 = 'curie_r1_deep_keyed_stream_L8_64M_smoke_20261016_v1', 'curie_r1_deep_keyed_stream_L8_64M_s0_20261016_v1'


def analyse(tag, message):
    subprocess.run([sys.executable, 'analysis/depth_data_scaling.py', '--tag', tag], cwd=ROOT, check=True)
    done = [t for t in (v11.L2, v11.L8, v16.L8_16, v16.L2_16, L8_64) if completed(t)]
    market.publish([f'experiments/results/token_language/{t}.json' for t in done] + [f'experiments/results/diagnostics/{tag}.json'], message)


def frontier_64m():
    log = ROOT/f'experiments/queue/logs/{L8_64_SMOKE}.log'
    lines = [l for l in log.read_text().splitlines() if l.startswith('{"smoke_steps"')] if log.exists() else []
    if not lines:
        stable_host(); original.run(L8_64_SMOKE, 3600)
        lines = [l for l in log.read_text().splitlines() if l.startswith('{"smoke_steps"')]
    s = json.loads(lines[-1]); print(json.dumps({'l8_64m_smoke': s}), flush=True)
    if s['max_rss_kb'] > 2_800_000 or s['s_per_step'] > 3.0: raise RuntimeError('64M L8 smoke exceeds the admitted cost envelope')
    run_or_await(L8_64, 172800)
    analyse('curie_depth_data_scaling_64m_20261016_v1', 'R1 scaling frontier: one-pass L8 at 65.5M fresh tokens (chunk-64 kernel), depth x data analysis')


def depth_data():
    try: run_or_await(v16.L8_16, 64800)
    except Exception as error: print(json.dumps({'depth_16m': v16.L8_16, 'failed': repr(error)}), flush=True)
    try: v18.ptb()
    except Exception as error: print(json.dumps({'ptb': 'failed', 'error': repr(error)}), flush=True)
    try:
        run_or_await(v16.L2_16, 43200)
        analyse('curie_depth_data_scaling_p5d_20261016_v1', 'R1 depth x data: 16M one-pass L2 partner and the P5d decision (L2-L8 gap at 16M vs 4M)')
    except Exception as error: print(json.dumps({'p5d': 'failed', 'error': repr(error)}), flush=True)
    try: frontier_64m()
    except Exception as error: print(json.dumps({'frontier_64m': 'failed', 'error': repr(error)}), flush=True)
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
