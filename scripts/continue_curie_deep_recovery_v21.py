"""Curie chain v21 (supersedes v20 before any v20 stage started; 16M L8 is awaited unchanged).

Native first (AGENTS.md "Build our own models"): after 16M L8 the causal event race member's first full fit
(theory/169), then the cheap dense 16M L2 diagnostic partner.
"""
import json
import continue_market_depth_v2 as market
import continue_curie_deep_recovery_v16 as v16
import continue_curie_deep_recovery_v19 as v19
from continue_curie_deep_recovery_v8 import completed, run_or_await

CERL4 = 'curie_r1_cerl_4M_s0_20261016_v1'


def main():
    try: run_or_await(v16.L8_16, 64800)
    except Exception as error: print(json.dumps({'depth_16m': v16.L8_16, 'failed': repr(error)}), flush=True)
    try:
        run_or_await(CERL4, 21600)
        market.publish([f'experiments/results/token_language/{CERL4}.json', 'experiments/results/diagnostics/curie_cerl_contracts_20261016_v2.json'],
                       'R1 native causal event race member: first 4M one-pass development fit (native share, active work, dense diagnostic comparison)')
    except Exception as error: print(json.dumps({'cerl_4m': 'failed', 'error': repr(error)}), flush=True)
    try:
        run_or_await(v16.L2_16, 43200)
        v19.analyse('curie_depth_data_scaling_p5d_20261016_v1', 'R1 diagnostic control (plan-B, dense): 16M one-pass L2 partner and the P5d depth x data decision')
    except Exception as error: print(json.dumps({'p5d': 'failed', 'error': repr(error)}), flush=True)
    print('Queue complete', flush=True)


if __name__ == '__main__':
    main()
