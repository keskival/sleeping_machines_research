"""Curie chain v26 (supersedes v25 while CERL-H 4M runs). Heavier runs delegated to AWS (founder 16 Oct): CERL 16M moves to AWS.
CERL-H 4M (running) → RMR K=4 dense / residual (ReZero-gated old-slot corrections) / sparse at 4M → PTB CERL round 1 → dense 16M L2.
"""
import json
import continue_market_depth_v2 as market
import continue_curie_deep_recovery_v16 as v16
import continue_curie_deep_recovery_v19 as v19
import continue_curie_deep_recovery_v22 as v22
import continue_curie_deep_recovery_v23 as v23
import continue_curie_deep_recovery_v25 as v25
from continue_curie_deep_recovery_v8 import run_or_await


def main():
    jobs = [(v23.CERLH2, 21600, 'R1 native compositional depth: two-level causal event race hierarchy (CERL-H) at 4M')] + \
           [(t, 21600, f'R1 native depth via residual memory: {t.split("_")[3]} 4-level variant at 4M against level-1 CERL') for t in v25.RMR]
    for tag, timeout, msg in jobs:
        try:
            run_or_await(tag, timeout); market.publish([f'experiments/results/token_language/{tag}.json'], msg)
        except Exception as error: print(json.dumps({tag: 'failed', 'error': repr(error)}), flush=True)
    try: v22.ptb_cerl()
    except Exception as error: print(json.dumps({'ptb_cerl': 'failed', 'error': repr(error)}), flush=True)
    try:
        run_or_await(v16.L2_16, 43200)
        v19.analyse('curie_depth_data_scaling_p5d_20261016_v1', 'R1 diagnostic control (plan-B, dense): 16M one-pass L2 partner and the P5d depth x data decision')
    except Exception as error: print(json.dumps({'p5d': 'failed', 'error': repr(error)}), flush=True)
    print('Queue complete', flush=True)


if __name__ == '__main__':
    main()
