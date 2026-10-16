"""Curie chain v22 (supersedes v21 before any v21 stage started; 16M L8 is awaited unchanged).

Native first (AGENTS.md "Build our own models"): 16M L8 (running) → CERL 4M FineWeb fit → PTB with CERL: smoke → d256 fit
(validation only) → validation scoring (frozen + causal online adaptation) → dense 16M L2 diagnostic partner (P5d).
"""
import json
from pathlib import Path
import continue_market_depth_v2 as market
import continue_curie_deep_language as original
from continue_curie_deep_recovery import stable_host
import continue_curie_deep_recovery_v16 as v16
import continue_curie_deep_recovery_v19 as v19
import continue_curie_deep_recovery_v21 as v21
from continue_curie_deep_recovery_v18 import word_done
from continue_curie_deep_recovery_v8 import run_or_await

ROOT = Path(__file__).resolve().parents[1]
SMOKE, FIT, VALID = 'curie_ptb_cerl_smoke_20261016_v1', 'curie_ptb_cerl_d256_s0_20261016_v1', 'curie_ptb_cerl_d256_s0_valid_score_20261016_v1'


def ptb_cerl():
    log = ROOT/f'experiments/queue/logs/{SMOKE}.log'
    lines = [l for l in log.read_text().splitlines() if l.startswith('{"smoke_steps"')] if log.exists() else []
    if not lines:
        stable_host(); original.run(SMOKE, 3600); lines = [l for l in log.read_text().splitlines() if l.startswith('{"smoke_steps"')]
    s = json.loads(lines[-1]); print(json.dumps({'ptb_cerl_smoke': s}), flush=True)
    if s['max_rss_kb'] > 2_600_000: raise RuntimeError('PTB CERL smoke exceeds the RSS envelope')
    timeout = int(s['s_per_step']*453*12*1.6 + 3600*3)                                        # 12 epochs of 453 updates + validation, 60% margin
    if not word_done(FIT): stable_host(); original.run(FIT, timeout)
    if not word_done(VALID): stable_host(); original.run(VALID, 21600)
    market.publish([f'experiments/results/word_language/{t}.json' for t in (FIT, VALID)] + ['experiments/results/diagnostics/curie_cerl_contracts_20261016_v3.json'],
                   'R1 PTB development round 1 with the native causal event race member: validation frozen and causal online adaptation')


def main():
    try: run_or_await(v16.L8_16, 64800)
    except Exception as error: print(json.dumps({'depth_16m': v16.L8_16, 'failed': repr(error)}), flush=True)
    try:
        run_or_await(v21.CERL4, 21600)
        market.publish([f'experiments/results/token_language/{v21.CERL4}.json', 'experiments/results/diagnostics/curie_cerl_contracts_20261016_v3.json'],
                       'R1 native causal event race member: first 4M one-pass development fit (native share, active work, dense diagnostic comparison)')
    except Exception as error: print(json.dumps({'cerl_4m': 'failed', 'error': repr(error)}), flush=True)
    try: ptb_cerl()
    except Exception as error: print(json.dumps({'ptb_cerl': 'failed', 'error': repr(error)}), flush=True)
    try:
        run_or_await(v16.L2_16, 43200)
        v19.analyse('curie_depth_data_scaling_p5d_20261016_v1', 'R1 diagnostic control (plan-B, dense): 16M one-pass L2 partner and the P5d depth x data decision')
    except Exception as error: print(json.dumps({'p5d': 'failed', 'error': repr(error)}), flush=True)
    print('Queue complete', flush=True)


if __name__ == '__main__':
    main()
