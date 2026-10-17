"""Curie chain v30 (supersedes v29 at the dense K4 4M final scoring). RMR dense K4 4M final scoring (running/next) → RMR residual K4
16M with decaying old-slot corrections (long-lane repair) → dense 16M L2 diagnostic (P5d)."""
import json
import continue_market_depth_v2 as market
import continue_curie_deep_recovery_v16 as v16
import continue_curie_deep_recovery_v19 as v19
from continue_curie_deep_recovery_v8 import run_or_await

JOBS = [('curie_r1_rmr_dense_K4_4M_s0_20261016_v1', 10800, 'R1 native depth via residual memory: dense 4-level variant at 4M against level-1 CERL (final scoring after resume)'),
        ('curie_r1_rmr_residdecay_K4_16M_s0_20261017_v1', 64800, 'R1 native depth at 16M: 4-level residual memory with decaying old-slot refinement (long-lane repair)')]


def main():
    for tag, timeout, msg in JOBS:
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
