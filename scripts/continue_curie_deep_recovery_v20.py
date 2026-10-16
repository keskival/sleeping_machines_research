"""Curie chain v20 (supersedes v19 before any v19 stage started; 16M L8 is awaited unchanged).

Founder direction 16 Oct (AGENTS.md "Build our own models"): the plan-B member is a diagnostic control, never a flagship.
Dense spending stops after the cheap 16M L2 partner that completes the registered P5d diagnostic. The dense PTB fit, the
dense 64M L8 run and the plan-B-based v15 stages are withdrawn. The native causal event race member
(theory/169_causal_race_language.md) is admitted in its own queue after its contracts pass.
"""
import json
import continue_curie_deep_recovery_v16 as v16
import continue_curie_deep_recovery_v19 as v19
from continue_curie_deep_recovery_v8 import run_or_await


def main():
    try: run_or_await(v16.L8_16, 64800)
    except Exception as error: print(json.dumps({'depth_16m': v16.L8_16, 'failed': repr(error)}), flush=True)
    try:
        run_or_await(v16.L2_16, 43200)
        v19.analyse('curie_depth_data_scaling_p5d_20261016_v1', 'R1 diagnostic control (plan-B, dense): 16M one-pass L2 partner and the P5d depth x data decision')
    except Exception as error: print(json.dumps({'p5d': 'failed', 'error': repr(error)}), flush=True)
    print('Queue complete: dense diagnostics done; native CERL jobs are admitted in their own queue', flush=True)


if __name__ == '__main__':
    main()
