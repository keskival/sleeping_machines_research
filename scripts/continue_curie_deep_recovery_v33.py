"""Curie chain v33 (supersedes v32 while the K4 decay run is running; awaited unchanged). After it: the native e2e smoke, the
sparse native gate (RSS/time), the sparse native 16M run when it fits Curie's memory, then the PTB fit resume + validation
scoring, then the dense 16M L2 diagnostic."""
import json, re
from pathlib import Path
import continue_market_depth_v2 as market
import continue_curie_deep_language as original
from continue_curie_deep_recovery import stable_host
import continue_curie_deep_recovery_v16 as v16
import continue_curie_deep_recovery_v19 as v19
import continue_curie_deep_recovery_v22 as v22
from continue_curie_deep_recovery_v18 import word_done
from continue_curie_deep_recovery_v8 import run_or_await, completed
from continue_curie_deep_recovery_v31 import DECAY

ROOT = Path(__file__).resolve().parents[1]
E2E, GATE, SPARSE = 'curie_r1_rmr_native_e2e_smoke_20261017_v1', 'curie_r1_rmr_nativesparse_K4_16M_s0_20261017_v1_gate', 'curie_r1_rmr_nativesparse_K4_16M_s0_20261017_v1'
RSS_MAX_KB = 2_400_000


def gate():
    log = (ROOT/f'experiments/queue/logs/{GATE}.log').read_text()
    m = re.findall(r'\{"smoke_steps".*\}', log); assert m, 'gate printed no smoke line'
    g = json.loads(m[-1]); return g['max_rss_kb'], g['s_per_step']


def main():
    try:
        run_or_await(DECAY, 64800); market.publish([f'experiments/results/token_language/{DECAY}.json'], 'R1 native depth at 16M: 4-level residual memory with decaying old-slot refinement (long-lane repair)')
    except Exception as error: print(json.dumps({DECAY: 'failed', 'error': repr(error)}), flush=True)
    try:
        if not completed(E2E): stable_host(); original.run(E2E, 3600)
        assert completed(E2E), 'e2e smoke did not complete'
        stable_host(); original.run(GATE, 3600); rss, sps = gate(); print(json.dumps(dict(gate=GATE, max_rss_kb=rss, s_per_step=sps)), flush=True)
        if rss < RSS_MAX_KB:
            run_or_await(SPARSE, int(16384*sps*1.4 + 4*3600))
            market.publish([f'experiments/results/token_language/{SPARSE}.json', f'experiments/results/token_language/{E2E}.json'], 'R1 sparse native member at 16M (all native mechanisms)')
        else: print(json.dumps({SPARSE: 'skipped', 'reason': f'gate RSS {rss} KB above {RSS_MAX_KB}'}), flush=True)
    except Exception as error: print(json.dumps({SPARSE: 'failed', 'error': repr(error)}), flush=True)
    try:
        if not word_done(v22.FIT): stable_host(); original.run(v22.FIT, 21600)
        if not word_done(v22.FIT): raise RuntimeError('PTB fit resume ended without a completed result')
        if not word_done(v22.VALID): stable_host(); original.run(v22.VALID, 21600)
        market.publish([f'experiments/results/word_language/{t}.json' for t in (v22.FIT, v22.VALID)] + ['experiments/results/diagnostics/curie_cerl_contracts_20261016_v9.json'],
                       'R1 PTB development round 1 with the native causal event race member: validation frozen and causal online adaptation')
    except Exception as error: print(json.dumps({'ptb': 'failed', 'error': repr(error)}), flush=True)
    try:
        run_or_await(v16.L2_16, 43200)
        v19.analyse('curie_depth_data_scaling_p5d_20261016_v1', 'R1 diagnostic control (plan-B, dense): 16M one-pass L2 partner and the P5d depth x data decision')
    except Exception as error: print(json.dumps({'p5d': 'failed', 'error': repr(error)}), flush=True)
    print('Queue complete', flush=True)


if __name__ == '__main__':
    main()
