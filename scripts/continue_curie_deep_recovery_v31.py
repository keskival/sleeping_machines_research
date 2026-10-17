"""Curie chain v31 (supersedes v30 while the PTB fit runs; awaited unchanged). The PTB CERL d256 fit (launched by chain v28's PTB stage) is running: await it, run its validation scoring
(frozen + causal online adaptation), then RMR residual K4 16M with decaying old-slot corrections (long-lane repair), then the
dense 16M L2 diagnostic (P5d)."""
import json, subprocess, time
import continue_market_depth_v2 as market
import continue_curie_deep_language as original
from continue_curie_deep_recovery import stable_host
import continue_curie_deep_recovery_v16 as v16
import continue_curie_deep_recovery_v19 as v19
import continue_curie_deep_recovery_v22 as v22
from continue_curie_deep_recovery_v18 import word_done
from continue_curie_deep_recovery_v8 import run_or_await

DECAY = 'curie_r1_rmr_residdecay_K4_16M_s0_20261017_v1'


def ptb():
    pat = f'^bash experiments/queue/run_safe.sh experiments/queue/r1/{v22.FIT}.txt'
    while subprocess.run(['pgrep', '-f', pat], capture_output=True).stdout: time.sleep(60)
    if not word_done(v22.FIT): raise RuntimeError('PTB fit ended without a completed result')
    if not word_done(v22.VALID): stable_host(); original.run(v22.VALID, 21600)
    market.publish([f'experiments/results/word_language/{t}.json' for t in (v22.FIT, v22.VALID)] + ['experiments/results/diagnostics/curie_cerl_contracts_20261016_v9.json'],
                   'R1 PTB development round 1 with the native causal event race member: validation frozen and causal online adaptation')


def main():
    try: ptb()
    except Exception as error: print(json.dumps({'ptb': 'failed', 'error': repr(error)}), flush=True)
    for tag in ('curie_rmr16M_vs_cerl16M_error_analysis_20261017_v1', 'curie_cerl16M_vs_denseL8_16M_error_analysis_20261017_v1'):   # analysis jobs (diagnostics JSON)
        try:
            from pathlib import Path
            if not (Path(__file__).resolve().parents[1]/f'experiments/results/diagnostics/{tag}.json').exists(): stable_host(); original.run(tag, 7200)
            market.publish([f'experiments/results/diagnostics/{tag}.json'], f'R1 error analysis {tag}')
        except Exception as error: print(json.dumps({tag: 'failed', 'error': repr(error)}), flush=True)
    try:
        run_or_await(DECAY, 64800); market.publish([f'experiments/results/token_language/{DECAY}.json'], 'R1 native depth at 16M: 4-level residual memory with decaying old-slot refinement (long-lane repair)')
    except Exception as error: print(json.dumps({DECAY: 'failed', 'error': repr(error)}), flush=True)
    try:
        run_or_await(v16.L2_16, 43200)
        v19.analyse('curie_depth_data_scaling_p5d_20261016_v1', 'R1 diagnostic control (plan-B, dense): 16M one-pass L2 partner and the P5d depth x data decision')
    except Exception as error: print(json.dumps({'p5d': 'failed', 'error': repr(error)}), flush=True)
    print('Queue complete', flush=True)


if __name__ == '__main__':
    main()
