"""Curie chain v24 (supersedes v23 while CERL-H 4M runs; awaited unchanged).

Native scaling first: CERL-H two-level 4M (running) → CERL 16M (native data scaling vs dense L8 16M 5.2183) → PTB CERL round 1
→ dense 16M L2 diagnostic partner.
"""
import json
import continue_market_depth_v2 as market
import continue_curie_deep_recovery_v16 as v16
import continue_curie_deep_recovery_v19 as v19
import continue_curie_deep_recovery_v22 as v22
import continue_curie_deep_recovery_v23 as v23
from continue_curie_deep_recovery_v8 import run_or_await

CERL16 = 'curie_r1_cerl_16M_s0_20261016_v1'


def main():
    for tag, timeout, msg in ((v23.CERLH2, 21600, 'R1 native compositional depth: two-level causal event race hierarchy (CERL-H) at 4M against level-1 CERL and the dense diagnostic'),
                              (CERL16, 43200, 'R1 native data scaling: causal event race member at 16M fresh tokens against the dense hybrid L8 16M')):
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
