"""Curie chain v23 (supersedes v22 while CERL 4M runs; awaited unchanged).

Compositional depth first (founder 16 Oct): CERL 4M (running) → CERL-H two-level 4M (theory 169 section 8) → PTB CERL round 1
→ dense 16M L2 diagnostic partner.
"""
import json
import continue_market_depth_v2 as market
import continue_curie_deep_recovery_v16 as v16
import continue_curie_deep_recovery_v19 as v19
import continue_curie_deep_recovery_v21 as v21
import continue_curie_deep_recovery_v22 as v22
from continue_curie_deep_recovery_v8 import run_or_await

CERLH2 = 'curie_r1_cerlh_L2lev_4M_s0_20261016_v1'


def main():
    try:
        run_or_await(v21.CERL4, 21600)
        market.publish([f'experiments/results/token_language/{v21.CERL4}.json', 'experiments/results/diagnostics/curie_cerl_contracts_20261016_v9.json'],
                       'R1 native causal event race member: first 4M one-pass development fit (native share, active work, dense diagnostic comparison)')
    except Exception as error: print(json.dumps({'cerl_4m': 'failed', 'error': repr(error)}), flush=True)
    try:
        run_or_await(CERLH2, 21600)
        market.publish([f'experiments/results/token_language/{CERLH2}.json', 'experiments/results/diagnostics/curie_cerlh_contracts_20261016_v1.json'],
                       'R1 native compositional depth: two-level causal event race hierarchy (CERL-H) at 4M against level-1 CERL and the dense diagnostic')
    except Exception as error: print(json.dumps({'cerlh_4m': 'failed', 'error': repr(error)}), flush=True)
    try: v22.ptb_cerl()
    except Exception as error: print(json.dumps({'ptb_cerl': 'failed', 'error': repr(error)}), flush=True)
    try:
        run_or_await(v16.L2_16, 43200)
        v19.analyse('curie_depth_data_scaling_p5d_20261016_v1', 'R1 diagnostic control (plan-B, dense): 16M one-pass L2 partner and the P5d depth x data decision')
    except Exception as error: print(json.dumps({'p5d': 'failed', 'error': repr(error)}), flush=True)
    print('Queue complete', flush=True)


if __name__ == '__main__':
    main()
