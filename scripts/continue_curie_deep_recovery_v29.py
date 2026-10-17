"""Curie chain v29 (supersedes v28 while RMR residual K4 16M runs; awaited unchanged). PTB/WT2 moved to AWS slots.
RMR residual K4 16M (running) → RMR dense K4 4M final scoring (exact resume at its last step) → dense 16M L2 diagnostic (P5d).
"""
import json
import continue_market_depth_v2 as market
import continue_curie_deep_recovery_v16 as v16
import continue_curie_deep_recovery_v19 as v19
import continue_curie_deep_recovery_v28 as v28
from continue_curie_deep_recovery_v8 import run_or_await


def main():
    for tag, timeout, msg in v28.JOBS:
        try:
            run_or_await(tag, timeout); market.publish([f'experiments/results/token_language/{tag}.json'], msg)
        except Exception as error: print(json.dumps({tag: 'failed', 'error': repr(error)}), flush=True)
    try:
        run_or_await(v16.L2_16, 43200)
        v19.analyse('curie_depth_data_scaling_p5d_20261016_v1', 'R1 diagnostic control (plan-B, dense): 16M one-pass L2 partner and the P5d depth x data decision')
    except Exception as error: print(json.dumps({'p5d': 'failed', 'error': repr(error)}), flush=True)
    print('Queue complete', flush=True)


if __name__ == '__main__':
    main()
