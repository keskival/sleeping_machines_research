"""Curie chain v28 (supersedes v27). RMR memory fix (sparse old-slot writes: 8192-token scoring 576 MB instead of > 1.5 GB):
RMR residual K4 16M (exact resume from its last checkpoint with the fixed code) → RMR dense K4 4M final scoring (exact resume at its
last step) → PTB CERL round 1 → dense 16M L2. CERL-H 4M final scoring waits for a memory-efficient level-2 evaluation.
"""
import json
import continue_market_depth_v2 as market
import continue_curie_deep_recovery_v16 as v16
import continue_curie_deep_recovery_v19 as v19
import continue_curie_deep_recovery_v22 as v22
from continue_curie_deep_recovery_v8 import run_or_await

JOBS = [('curie_r1_rmr_resid_K4_16M_s0_20261016_v1', 64800, 'R1 native depth at 16M: 4-level residual memory with gated old-slot refinement against CERL 16M and dense L8 16M'),
        ('curie_r1_rmr_dense_K4_4M_s0_20261016_v1', 10800, 'R1 native depth via residual memory: dense 4-level variant at 4M against level-1 CERL (final scoring after resume)')]


def main():
    for tag, timeout, msg in JOBS:
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
